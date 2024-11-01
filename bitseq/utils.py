import os
import random
from math import log
import numpy as np

import torch
from torch.distributions.categorical import Categorical

from scipy.stats import spearmanr


def set_random_seeds(seed, deterministic=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic


def construct_M(n, b, H, M_size, seed=0):
    np.random.seed(seed)
    M = []
    for i in range(M_size):
        M.append("".join([np.random.choice(H) for _ in range(n // b)]))
        assert len(M[-1]) == n
    return M


def distance(s1, s2):
    assert len(s1) == len(s2)
    return sum([int(s1[i] != s2[i]) for i in range(len(s1))])


def M_distance(s, M):
    return min([distance(s, ms) for ms in M])


def construct_test_set(M, seed=0):
    np.random.seed(seed)
    test_set = []
    for s in M:
        test_set.append(s)
        for cnt in range(1, len(s)):
            new_s = list(s)
            subset = np.random.choice(list(range(len(s))), size=cnt, replace=False)
            for i in subset:
                new_s[i] = "0" if s[i] == "1" else "1"
            test_set.append("".join(new_s))
            assert len(test_set[-1]) == len(s)
            assert distance(test_set[-1], s) == cnt
    return test_set


def log_reward(s, M):
    return -M_distance(s, M)


def reward(s, M):
    return np.exp(log_reward(s, M))


def token_seq_to_str(seq, k):
    return "".join([bin(int(v))[2:].zfill(k) for v in seq])


def batch_rewards(batch, M, k):
    batch_np = batch.cpu().numpy()
    rewards = [reward(token_seq_to_str(batch_np[i], k), M) for i in range(batch_np.shape[0])]
    return torch.tensor(rewards)


def batch_log_rewards(batch, M, k):
    batch_np = batch.cpu().numpy()
    log_rewards = [log_reward(token_seq_to_str(batch_np[i], k), M) for i in range(batch_np.shape[0])]
    return torch.tensor(log_rewards)


def process_logits(all_logits, pos_mask, args):
    # Model predicts positional logits p_i and word logits for each position w_ij.
    # The logits used to sample pairs of positions and word (i, j) are computed as p_i + w_ij.
    pos_logits = all_logits[0, :, -(args.n // args.k + 1) :]  # [batch_size, n/k + 1]
    pos_logits[pos_mask] = -torch.inf
    word_logits = all_logits[:, :, : 2**args.k]  # [n/k + 1, batch_size, 2^k]
    sum_logits = torch.moveaxis(word_logits, 1, 0) + pos_logits[:, :, None]  # [batch_size, n/k + 1, 2^k]
    sum_logits = sum_logits.reshape(
        pos_logits.shape[0], (args.n // args.k + 1) * (2**args.k)
    )  # [batch_size, (n/k + 1) * 2^k]
    return pos_logits, word_logits, sum_logits


def compute_correlation_wpb(model, M, test_set, args, rounds=10, batch_size=180):
    model.eval()
    assert len(test_set) % batch_size == 0
    logP_sums = torch.zeros(len(test_set), rounds).to(args.device)

    for round in range(rounds):
        for batch_idx in range(len(test_set) // batch_size):
            batch = []
            for j in range(batch_size):
                s = str(test_set[batch_idx * batch_size + j])
                current_word = [2**args.k + 1] + [
                    int(s[args.k * i : args.k * (i + 1)], base=2) for i in range(args.n // args.k)
                ]
                batch.append(current_word)
            batch = torch.tensor(batch, device=args.device)

            for i in range(args.n // args.k):
                with torch.no_grad():
                    _, pb_logits = model(batch.T)
                    pb_logits[batch >= (2**args.k)] = -torch.inf

                    while True:
                        positions = Categorical(logits=pb_logits).sample().to(args.device)
                        if (batch[range(batch_size), positions] != 2**args.k).sum() == batch_size:
                            break
                    logPb = Categorical(logits=pb_logits).log_prob(positions)

                    assert positions.min() >= 1
                    assert positions.max() <= args.n // args.k

                    actions = positions * (2**args.k) + batch[range(batch_size), positions]
                    batch[range(batch_size), positions] = 2**args.k

                    pos_mask = batch != 2**args.k
                    forward_logits, _ = model(batch.T)
                    _, _, forward_logits_sum = process_logits(forward_logits, pos_mask, args)
                    logPf = Categorical(logits=forward_logits_sum).log_prob(actions)

                    logP_sums[batch_idx * batch_size : (batch_idx + 1) * batch_size, round] += logPf - logPb

    logP_sum = torch.logsumexp(logP_sums, dim=-1)
    log_rewards = np.array([log_reward(s, M) for s in test_set])
    return spearmanr((args.reward_exponent * log_rewards), (logP_sum.detach().cpu().numpy())).statistic


def compute_correlation(model, M, test_set, args, rounds=10, batch_size=180):
    # Sampling a trajectory from PB(tau | x) when PB is uniform over parents
    # in this case is equvalent to starting at s0 and randomly choosing the order
    # in which we replace empty words with words at corresponding positions from x.
    # Thus we can sample trajectories and compute PF(tau) in parallel.
    model.eval()
    assert len(test_set) % batch_size == 0
    p_forward_sums = torch.zeros(len(test_set), rounds).to(args.device)

    for round in range(rounds):
        for batch_idx in range(len(test_set) // batch_size):
            batch = torch.tensor(
                [[2**args.k + 1] + ([2**args.k] * (args.n // args.k)) for i in range(batch_size)]
            ).to(args.device)
            for i in range(args.n // args.k):
                with torch.no_grad():
                    pos_mask = batch != 2**args.k
                    all_logits, _ = model(batch.T)
                    pos_logits, word_logits, sum_logits = process_logits(all_logits, pos_mask, args)

                    # There is a bug in pytorch that allows to sample objects that has 0 probability (happens very rarely but still happens).
                    # This loop basically resamples until everything is correct.
                    while True:
                        uniform_probs = torch.zeros(batch_size, args.n // args.k + 1) + 1 / (args.n // args.k - i)
                        uniform_probs[pos_mask] = 0.0
                        positions = Categorical(probs=uniform_probs).sample().to(args.device)
                        if (batch[range(batch_size), positions] == 2**args.k).sum() == batch_size:
                            break

                    assert positions.min() >= 1
                    assert positions.max() <= args.n // args.k

                    words = []
                    for j in range(batch_size):
                        s = test_set[batch_idx * batch_size + j]
                        word = int(s[(positions[j] - 1) * args.k : positions[j] * args.k], base=2)
                        words.append(word)
                    words = torch.tensor(words).to(args.device)

                    batch_cl = batch.clone()
                    batch_cl[range(batch_size), positions] = words
                    batch = batch_cl

                    actions = positions * (2**args.k) + words
                    log_pf = sum_logits[range(batch_size), actions] / args.entropy_coeff - torch.logsumexp(
                        sum_logits / args.entropy_coeff, dim=-1
                    )
                    p_forward_sums[batch_idx * batch_size : (batch_idx + 1) * batch_size, round] += log_pf

    p_forward_sum = torch.logsumexp(p_forward_sums, dim=-1)
    log_rewards = np.array([log_reward(s, M) for s in test_set])
    return spearmanr((args.reward_exponent * log_rewards), (p_forward_sum.detach().cpu().numpy())).statistic
