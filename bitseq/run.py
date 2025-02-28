import argparse
import random
import os
import time
import numpy as np
from math import log
from copy import deepcopy

import torch
from torch.distributions.categorical import Categorical

from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from tensordict import TensorDict

from model import TransformerModel
from utils import (
    set_random_seeds,
    batch_log_rewards,
    batch_rewards,
    compute_correlation,
    compute_correlation_wpb,
    construct_M,
    construct_test_set,
    distance,
    process_logits,
    token_seq_to_str,
)

parser = argparse.ArgumentParser()

# Environment params
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--n", default=120, type=int)
parser.add_argument("--k", default=8, type=int)
parser.add_argument("--M_size", default=60, type=int)
parser.add_argument("--mode_threshold", default=30, type=int)
parser.add_argument("--reward_exponent", default=2.0, type=float)

parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--print_every", default=100, type=int)
parser.add_argument("--validate_every", default=2000, type=int)
parser.add_argument("--print_modes", default=False, action="store_true")
parser.add_argument("--log_grad_norm", default=False, action="store_true")

# Base training params
parser.add_argument("--num_iterations", default=50000, type=int)
parser.add_argument("--rand_action_prob", default=0.001, type=float)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--blr", type=float)
parser.add_argument("--gamma", default=0.9999, type=float)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument(
    "--backward_approach", default="uniform", choices=["uniform", "tlm", "naive", "pessimistic"], type=str
)
parser.add_argument("--uniform_init", action="store_false")

parser.add_argument("--objective", choices=["tb", "db", "subtb", "dqn"], type=str)
parser.add_argument("--z_lr", default=0.001, type=float)
parser.add_argument("--subtb_lambda", default=0.9, type=float)
parser.add_argument("--leaf_coeff", default=5.0, type=float)
parser.add_argument("--update_target_every", default=5, type=int)
parser.add_argument("--tau", default=0.1, type=float)
parser.add_argument("--corr_num_rounds", default=10, type=int)

# SoftDQN params
parser.add_argument("--start_learning", default=50, type=int)
parser.add_argument("--softdqn_loss", default="Huber", type=str)

# Replay buffer parameters
parser.add_argument("--rb_size", default=100000, type=int)
parser.add_argument("--rb_batch_size", default=256, type=int)
parser.add_argument("--per_alpha", default=0.9, type=float)
parser.add_argument("--per_beta", default=0.1, type=float)
parser.add_argument("--anneal_per_beta", default=False, action="store_true")

# Munchausen DQN parameters
parser.add_argument("--m_alpha", default=0.15, type=float)
parser.add_argument("--entropy_coeff", default=1.0, type=float)
parser.add_argument("--m_l0", default=-25.0, type=float)

pessimistic_buffer = []
pessimistic_size = 20


def sample_forward(sum_logits, sum_uniform, batch, args):
    # There is a bug in pytorch that allows to sample objects that has 0 probability (happens very rarely but still happens).
    # This loop basically resamples until everything is correct.
    while True:
        actions = Categorical(logits=sum_logits.clone()).sample()
        uniform_actions = Categorical(logits=sum_uniform).sample().to(args.device)
        uniform_mask = torch.rand(args.batch_size) < args.rand_action_prob
        actions[uniform_mask] = uniform_actions[uniform_mask]
        positions = actions // (2**args.k)
        if (batch[range(args.batch_size), positions] == 2**args.k).sum() == args.batch_size:
            break
    assert positions.min() >= 1
    assert positions.max() <= args.n // args.k
    words = actions % (2**args.k)
    return actions, positions, words


def TB_train_step(model, target_model, logZ, optimizer, Z_optimizer, pb_optimizer, M, args):
    global pessimistic_buffer
    # This code is pretty simple because all trajectories in our graph have the same length.
    model.train()

    # The seqence has length n/k + 1 and at the beginning looks like [2^k + 1, 2^k, 2^k, ..., 2^k].
    # 2^k + 1: [BOS] token, 2^k: token for "empty" word.
    batch = torch.tensor([[2**args.k + 1] + ([2**args.k] * (args.n // args.k)) for i in range(args.batch_size)]).to(
        args.device
    )
    history = []

    for i in range(args.n // args.k):
        pos_mask = batch != 2**args.k
        all_logits, _ = model(batch.T)

        pos_logits, word_logits, sum_logits = process_logits(all_logits, pos_mask, args)
        with torch.no_grad():
            _, _, sum_uniform = process_logits(0.0 * all_logits.clone(), pos_mask, args)

            actions, positions, words = sample_forward(sum_logits, sum_uniform, batch, args)

            batch_cl = batch.clone()
            batch_cl[range(args.batch_size), positions] = words
            history.append([batch.clone(), batch_cl.clone(), positions.clone(), actions.clone()])
            batch = batch_cl

    pessimistic_buffer.append(history)
    pessimistic_buffer = pessimistic_buffer[-pessimistic_size:]

    if args.backward_approach == "uniform":
        pb_loss, pb_deviation = 0.0, 0.0
    else:
        pb_loss = torch.zeros(args.batch_size).to(args.device)
        pb_deviation = 0.0

        if args.backward_approach == "pessimistic":
            history_for_pb_update = history
        else:
            history_for_pb_update = random.choice(pessimistic_buffer)

        for i, (_, batch, positions, _) in enumerate(history_for_pb_update):
            _, pb_logits = model(batch.T)
            mask = batch < (2**args.k)
            pb_logits[~mask] = -torch.inf
            logPb = pb_logits - torch.logsumexp(pb_logits, dim=-1).unsqueeze(-1)
            pb_loss += logPb[range(args.batch_size), positions]
            pb_deviation += torch.abs(logPb[mask] - log(1 / (i + 1))).mean().cpu().item()
        if args.backward_approach in ["tlm", "pessimistic"]:
            pb_loss = -torch.mean(pb_loss) / (args.n // args.k)
            pb_loss.backward()
            pb_optimizer.step()
            pb_optimizer.zero_grad()
            pb_deviation /= len(history)
            pb_loss = pb_loss.cpu().item()
        else:
            pb_loss = 0.0

    sumlogPf = torch.zeros(args.batch_size).to(args.device)
    sumlogPb = torch.zeros(args.batch_size).to(args.device)
    for i, (prev_batch, batch, positions, actions) in enumerate(history):
        pos_mask = prev_batch != 2**args.k
        all_logits, pb_logits = model(prev_batch.T)
        if args.backward_approach == "tlm":
            _, pb_logits = target_model(prev_batch.T)

        _, _, sum_logits = process_logits(all_logits, pos_mask, args)

        sumlogPf += sum_logits[range(args.batch_size), actions] - torch.logsumexp(sum_logits, dim=-1)

        if args.backward_approach == "uniform":
            sumlogPb += torch.log(torch.tensor(1 / (i + 1))).to(args.device)
        else:
            pb_logits[batch >= (2**args.k)] = -torch.inf
            sumlogPb += pb_logits[range(args.batch_size), positions] - torch.logsumexp(pb_logits, dim=-1)

    log_rewards = args.reward_exponent * batch_log_rewards(batch[:, 1:], M, args.k).to(args.device).detach()
    loss = (logZ.sum() + sumlogPf - sumlogPb - log_rewards).pow(2).mean() / (args.n // args.k)
    loss.backward()
    optimizer.step()
    Z_optimizer.step()
    optimizer.zero_grad()
    Z_optimizer.zero_grad()

    assert batch[:, 1:].max() < 2**args.k
    return loss.cpu().item(), batch[:, 1:].cpu(), pb_loss, pb_deviation


def DB_train_step(model, target_model, optimizer, pb_optimizer, M, args):
    global pessimistic_buffer
    # This code is pretty simple because all trajectories in our graph have the same length.
    model.train()

    # The seqence has length n/k + 1 and at the beginning looks like [2^k + 1, 2^k, 2^k, ..., 2^k].
    # 2^k + 1: [BOS] token, 2^k: token for "empty" word.
    batch = torch.tensor([[2**args.k + 1] + ([2**args.k] * (args.n // args.k)) for i in range(args.batch_size)]).to(
        args.device
    )
    history = []

    for i in range(args.n // args.k):
        pos_mask = batch != 2**args.k
        all_logits, _ = model(batch.T)
        pos_logits, word_logits, sum_logits = process_logits(all_logits, pos_mask, args)

        with torch.no_grad():
            _, _, sum_uniform = process_logits(0.0 * all_logits.clone(), pos_mask, args)

            try:
                actions, positions, words = sample_forward(sum_logits, sum_uniform, batch, args)
            except:
                print(sum_logits)
                exit(0)

            batch_cl = batch.clone()
            batch_cl[range(args.batch_size), positions] = words
            history.append([batch.clone(), batch_cl.clone(), positions.clone(), actions.clone()])
            batch = batch_cl

    pessimistic_buffer.append(history)
    pessimistic_buffer = pessimistic_buffer[-pessimistic_size:]

    pb_grad_norm = 0.0
    if args.backward_approach == "uniform":
        pb_loss, pb_deviation = 0.0, 0.0
    else:
        pb_loss = torch.zeros(args.batch_size).to(args.device)
        pb_deviation = 0.0

        if args.backward_approach == "pessimistic":
            history_for_pb_update = history
        else:
            history_for_pb_update = random.choice(pessimistic_buffer)

        for i, (_, batch, positions, _) in enumerate(history_for_pb_update):
            _, pb_logits = model(batch.T)
            mask = batch < (2**args.k)
            pb_logits[~mask] = -torch.inf
            logPb = pb_logits - torch.logsumexp(pb_logits, dim=-1).unsqueeze(-1)
            pb_loss += logPb[range(args.batch_size), positions]
            pb_deviation += torch.abs(logPb[mask] - log(1 / (i + 1))).mean().cpu().item()
        if args.backward_approach in ["tlm", "pessimistic"]:
            pb_loss = -torch.mean(pb_loss) / (args.n // args.k)
            pb_loss.backward()
            for p in list(filter(lambda p: p.grad is not None, model.parameters())):
                pb_grad_norm += p.grad.data.norm(2).item()
            pb_optimizer.step()
            pb_optimizer.zero_grad()
            pb_deviation /= len(history)
            pb_loss = pb_loss.cpu().item()
        else:
            pb_loss = 0.0

    loss = torch.tensor(0.0).to(args.device)
    for i, (prev_batch, batch, positions, actions) in enumerate(history):
        pos_mask = prev_batch != 2**args.k
        all_logits, _ = model(prev_batch.T)
        pos_logits, word_logits, sum_logits = process_logits(all_logits, pos_mask, args)
        prev_logF = all_logits[0, :, 2**args.k]
        prev_logPf = sum_logits[range(args.batch_size), actions] - torch.logsumexp(sum_logits, dim=-1)

        all_logits, pb_logits = model(batch.T)
        if args.backward_approach == "tlm":
            _, pb_logits = target_model(batch.T)
        logF = all_logits[0, :, 2**args.k]
        if i + 1 == len(history):
            logF = args.reward_exponent * batch_log_rewards(batch[:, 1:], M, args.k).to(args.device).detach()
        if args.backward_approach == "uniform":
            logPb = torch.log(torch.tensor(1 / (i + 1))).to(args.device)
        else:
            pb_logits[batch >= (2**args.k)] = -torch.inf
            logPb = pb_logits[range(args.batch_size), positions] - torch.logsumexp(pb_logits, dim=-1)

        loss += (prev_logF + prev_logPf - logF - logPb).pow(2).mean()

    loss = loss / (args.n // args.k)
    loss.backward()
    grad_norm = 0.0
    for p in list(filter(lambda p: p.grad is not None, model.parameters())):
        grad_norm += p.grad.data.norm(2).item()
    optimizer.step()
    optimizer.zero_grad()

    assert batch[:, 1:].max() < 2**args.k
    return loss.cpu().item(), batch[:, 1:].cpu(), pb_loss, pb_deviation, grad_norm, pb_grad_norm


def SubTB_train_step(model, target_model, optimizer, pb_optimizer, M, args):
    global pessimistic_buffer
    # This code is pretty simple because all trajectories in our graph have the same length.
    model.train()

    # The seqence has length n/k + 1 and at the beginning looks like [2^k + 1, 2^k, 2^k, ..., 2^k].
    # 2^k + 1: [BOS] token, 2^k: token for "empty" word.
    batch = torch.tensor([[2**args.k + 1] + ([2**args.k] * (args.n // args.k)) for i in range(args.batch_size)]).to(
        args.device
    )
    history = []

    for i in range(args.n // args.k):
        pos_mask = batch != 2**args.k
        all_logits, _ = model(batch.T)
        pos_logits, word_logits, sum_logits = process_logits(all_logits, pos_mask, args)

        with torch.no_grad():
            _, _, sum_uniform = process_logits(0.0 * all_logits.clone(), pos_mask, args)

            actions, positions, words = sample_forward(sum_logits, sum_uniform, batch, args)

            batch_cl = batch.clone()
            batch_cl[range(args.batch_size), positions] = words
            history.append([batch.clone(), batch_cl.clone(), positions.clone(), actions.clone()])
            batch = batch_cl

    pessimistic_buffer.append(history)
    pessimistic_buffer = pessimistic_buffer[-pessimistic_size:]

    if args.backward_approach == "uniform":
        pb_loss, pb_deviation = 0.0, 0.0
    else:
        pb_loss = torch.zeros(args.batch_size).to(args.device)
        pb_deviation = 0.0

        if args.backward_approach == "pessimistic":
            history_for_pb_update = history
        else:
            history_for_pb_update = random.choice(pessimistic_buffer)

        for i, (_, batch, positions, _) in enumerate(history_for_pb_update):
            _, pb_logits = model(batch.T)
            mask = batch < (2**args.k)
            pb_logits[~mask] = -torch.inf
            logPb = pb_logits - torch.logsumexp(pb_logits, dim=-1).unsqueeze(-1)
            pb_loss += logPb[range(args.batch_size), positions]
            pb_deviation += torch.abs(logPb[mask] - log(1 / (i + 1))).mean().cpu().item()
        if args.backward_approach in ["tlm", "pessimistic"]:
            pb_loss = -torch.mean(pb_loss) / (args.n // args.k)
            pb_loss.backward()
            pb_optimizer.step()
            pb_optimizer.zero_grad()
            pb_deviation /= len(history)
            pb_loss = pb_loss.cpu().item()
        else:
            pb_loss = 0.0

    log_pfs = torch.zeros(args.n // args.k + 1, args.batch_size).to(args.device)
    log_pbs = torch.zeros(args.n // args.k + 1, args.batch_size).to(args.device)
    log_fs = torch.zeros(args.n // args.k + 1, args.batch_size).to(args.device)
    for i, (prev_batch, batch, positions, actions) in enumerate(history):
        pos_mask = prev_batch != 2**args.k
        all_logits, _ = model(prev_batch.T)
        pos_logits, word_logits, sum_logits = process_logits(all_logits, pos_mask, args)
        log_fs[i] = all_logits[0, :, 2**args.k]
        log_pfs[i] = sum_logits[range(args.batch_size), actions] - torch.logsumexp(sum_logits, dim=-1)

        pos_mask = batch != 2**args.k
        if args.backward_approach == "tlm":
            _, pb_logits = target_model(batch.T)
        else:
            _, pb_logits = model(batch.T)
        if args.backward_approach == "uniform":
            log_pbs[i + 1] = torch.log(torch.tensor(1 / (i + 1))).to(args.device)
        else:
            pb_logits[batch >= (2**args.k)] = -torch.inf
            log_pbs[i + 1] = pb_logits[range(args.batch_size), positions] - torch.logsumexp(pb_logits, dim=-1)

        if i + 1 == len(history):
            log_fs[-1] = args.reward_exponent * batch_log_rewards(batch[:, 1:], M, args.k).to(args.device).detach()

    loss = torch.tensor(0.0).to(args.device)
    total_lambda = torch.tensor(0.0).to(args.device)
    for i in range(log_fs.shape[0]):
        for j in range(i + 1, log_fs.shape[0]):
            lmbd = args.subtb_lambda ** (j - i)
            loss += (
                lmbd
                * (log_fs[i, :] + log_pfs[i:j, :].sum(dim=0) - log_fs[j, :] - log_pbs[i + 1 : j + 1, :].sum(dim=0))
                .pow(2)
                .mean()
            )
            total_lambda += lmbd
    loss /= total_lambda * (args.n // args.k)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    assert batch[:, 1:].max() < 2**args.k
    return loss.cpu().item(), batch[:, 1:].cpu(), pb_loss, pb_deviation


def SoftDQN_collect_experience(rb, model, target_model, pb_optimizer, M, args, is_learning_started):
    global pessimistic_buffer
    # This code is pretty simple because all trajectories in our graph have the same length.

    # The seqence has length n/k + 1 and at the beginning looks like [2^k + 1, 2^k, 2^k, ..., 2^k].
    # 2^k + 1: [BOS] token, 2^k: token for "empty" word.
    batch = torch.tensor([[2**args.k + 1] + ([2**args.k] * (args.n // args.k)) for i in range(args.batch_size)]).to(
        args.device
    )
    history = []
    with torch.no_grad():
        for i in range(args.n // args.k):
            pos_mask = batch != 2**args.k

            all_logits, _ = model(batch.T)
            _, _, sum_logits = process_logits(all_logits, pos_mask, args)
            _, _, sum_uniform = process_logits(0.0 * all_logits.clone(), pos_mask, args)

            actions, positions, words = sample_forward(sum_logits, sum_uniform, batch, args)

            next_batch = batch.clone()
            next_batch[range(args.batch_size), positions] = words

            if args.backward_approach == "uniform":
                rewards = torch.log(torch.tensor([1 / (i + 1)] * args.batch_size).to(args.device))
            else:
                if args.backward_approach == "tlm":
                    _, pb_logits = target_model(next_batch.T)
                else:
                    _, pb_logits = model(next_batch.T)
                pb_logits[next_batch >= (2**args.k)] = -torch.inf
                rewards = pb_logits[range(args.batch_size), positions] - torch.logsumexp(pb_logits, dim=-1)

            history.append([next_batch.clone(), positions.clone()])

            # The last added word
            if i + 1 == args.n // args.k:
                rewards += args.reward_exponent * batch_log_rewards(next_batch[:, 1:], M, args.k).to(args.device)
                is_done = torch.tensor([1.0] * args.batch_size).to(args.device)
            else:
                is_done = torch.tensor([0.0] * args.batch_size).to(args.device)

            rb_record = TensorDict(
                {
                    "state": batch,
                    "action": actions,
                    "next_state": next_batch,
                    "rewards": rewards,
                    "is_done": is_done,
                },
                batch_size=args.batch_size,
            )
            rb.extend(rb_record)  # add record to replay buffer
            batch = next_batch

    pessimistic_buffer.append(history)
    pessimistic_buffer = pessimistic_buffer[-pessimistic_size:]

    if not is_learning_started or args.backward_approach == "uniform":
        pb_loss, pb_deviation = 0.0, 0.0
    else:
        pb_loss = torch.zeros(args.batch_size).to(args.device)
        pb_deviation = 0.0

        if args.backward_approach == "pessimistic":
            history_for_pb_update = history
        else:
            history_for_pb_update = random.choice(pessimistic_buffer)

        for i, (observed_batch, positions) in enumerate(history_for_pb_update):
            _, pb_logits = model(observed_batch.T)
            mask = observed_batch < (2**args.k)
            pb_logits[~mask] = -torch.inf
            logPb = pb_logits - torch.logsumexp(pb_logits, dim=-1).unsqueeze(-1)
            pb_loss += logPb[range(args.batch_size), positions]
            pb_deviation += torch.abs(logPb[mask] - log(1 / (i + 1))).mean().cpu().item()
        if args.backward_approach in ["tlm", "pessimistic"]:
            pb_loss = -torch.mean(pb_loss) / (args.n // args.k)
            pb_loss.backward()
            pb_optimizer.step()
            pb_optimizer.zero_grad()
            pb_deviation /= len(history)
            pb_loss = pb_loss.cpu().item()
        else:
            pb_loss = 0.0

    assert batch[:, 1:].max() < 2**args.k
    return batch[:, 1:].cpu(), pb_loss, pb_deviation


def SoftDQN_learn_rb(progress, rb, model, target_model, optimizer, M, args):
    # Select loss function
    if args.softdqn_loss == "Huber":
        loss_fn = torch.nn.HuberLoss(reduction="none")
    else:
        loss_fn = torch.nn.MSELoss(reduction="none")
    if args.anneal_per_beta:
        # Update beta parameter of experience replay
        add_beta = (1.0 - args.per_beta) * progress
        rb._sampler._beta = args.per_beta + add_beta

    model.train()
    optimizer.zero_grad()

    # Sample from replay buffer
    rb_batch = rb.sample().to(args.device)
    # Compute td-loss
    pos_mask = rb_batch["state"] != 2**args.k
    all_logits, _ = model(rb_batch["state"].T)
    _, _, sum_logits = process_logits(all_logits, pos_mask, args)
    if args.m_alpha > 0:
        all_target_logits, _ = target_model(rb_batch["state"].T)
        _, _, sum_target_logits = process_logits(all_target_logits, pos_mask, args)
        norm_target_logits = sum_target_logits / args.entropy_coeff

    q_values = sum_logits[range(args.rb_batch_size), rb_batch["action"]]

    with torch.no_grad():
        pos_mask = rb_batch["next_state"] != 2**args.k
        all_target_logits, _ = target_model(rb_batch["next_state"].T)
        _, _, sum_target_logits = process_logits(all_target_logits, pos_mask, args)
        target_v_next_values = args.entropy_coeff * torch.logsumexp(sum_target_logits / args.entropy_coeff, dim=-1)
        target_v_next_values[rb_batch["is_done"].bool()] = 0.0
        td_target = rb_batch["rewards"] + target_v_next_values

        if args.m_alpha > 0:
            target_log_policy = norm_target_logits[range(args.rb_batch_size), rb_batch["action"]] - torch.logsumexp(
                norm_target_logits, dim=-1
            )
            munchausen_penalty = torch.clamp(args.entropy_coeff * target_log_policy, min=args.m_l0, max=1)
            td_target += args.m_alpha * munchausen_penalty

    td_errors = loss_fn(q_values, td_target)
    td_errors[rb_batch["is_done"].bool()] *= args.leaf_coeff

    # Update PER
    rb_batch["td_error"] = td_errors
    rb.update_tensordict_priority(rb_batch)

    # Compute loss with IS correction
    loss = (td_errors * rb_batch["_weight"]).mean()
    loss.backward()
    optimizer.step()

    return loss.cpu().item()


def main(args):
    device = args.device
    assert args.n % args.k == 0
    assert args.validate_every % args.print_every == 0
    H = ["00000000", "11111111", "11110000", "00001111", "00111100"]
    assert args.n % len(H[0]) == 0
    M = construct_M(args.n, len(H[0]), H, args.M_size, seed=42)
    test_set = construct_test_set(M, seed=42)
    print(f"test set size: {len(test_set)}")
    set_random_seeds(args.seed, False)

    experiment_name = f"results/{args.seed}_n={args.n}_k={args.k}_rexp={args.reward_exponent}_{args.objective}"
    if args.objective == "dqn" and args.m_alpha > 0:
        experiment_name += f"_m_alpha={args.m_alpha}"
    elif args.objective == "subtb":
        experiment_name += f"_lambda={args.subtb_lambda}"
    blr = args.lr if args.blr is None else args.lr
    gamma = 1 if args.backward_approach == "pessimistic" else args.gamma
    experiment_name += f"_lr={args.lr}_blr={blr}_lrg={gamma}_pb={args.backward_approach}"
    if args.backward_approach == "tlm":
        experiment_name += f"_tau={args.tau}"
    os.makedirs(experiment_name, exist_ok=True)
    print(experiment_name)

    model = TransformerModel(
        ntoken=2**args.k + 2,
        d_model=64,
        d_hid=64,
        nhead=8,
        nlayers=3,
        seq_len=args.n // args.k,
        dropout=args.dropout,
        uniform_init=args.uniform_init,
    ).to(device)
    target_model = deepcopy(model)
    target_model.load_state_dict(model.state_dict())

    logZ = torch.nn.Parameter(torch.tensor(np.ones(64) * 0.0 / 64, requires_grad=True, device=device))

    if args.backward_approach in ["tlm", "pessimistic"]:
        f_params = [v for k, v in dict(model.named_parameters()).items() if not "pb_linear" in k]
        optimizer = torch.optim.Adam(f_params, args.lr, weight_decay=1e-5)
        if args.objective == "tb":
            b_params = [v for k, v in dict(model.named_parameters()).items() if "pb_linear" in k]
        else:
            b_params = [v for k, v in dict(model.named_parameters()).items() if not "pf_linear" in k]

        pb_optimizer = torch.optim.Adam(b_params, blr, weight_decay=1e-5)
        b_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(pb_optimizer, gamma)
    else:
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-5)
        pb_optimizer = None
        b_lr_scheduler = None
    Z_optimizer = torch.optim.Adam([logZ], args.z_lr, weight_decay=1e-5)

    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(args.rb_size),
        sampler=PrioritizedSampler(
            max_capacity=args.rb_size,
            alpha=args.per_alpha,
            beta=args.per_beta,
        ),
        batch_size=args.rb_batch_size,
        priority_key="td_error",
    )

    modes = [False] * len(M)
    sum_rewards = 0.0

    corr_nums = []
    mode_nums = []
    if args.objective == "dqn":
        # Renormalize entropy for Munchausen DQN
        args.entropy_coeff *= 1 / (1 - args.m_alpha)

    previous_time = time.time()
    logs_to_save = {
        "loss": [],
        "pb_loss": [],
        "pb_deviation": [],
        "num_modes": [],
        "corr_w_uniform": [],
        "corr_w_naive": [],
        "grad_norm": [],
        "pb_grad_norm": [],
    }
    for it in range(args.num_iterations + 1):
        progress = float(it) / args.num_iterations
        if args.objective == "tb":
            loss, batch, pb_loss, pb_deviation = TB_train_step(
                model, target_model, logZ, optimizer, Z_optimizer, pb_optimizer, M, args
            )
        elif args.objective == "db":
            loss, batch, pb_loss, pb_deviation, grad_norm, pb_grad_norm = DB_train_step(
                model, target_model, optimizer, pb_optimizer, M, args
            )
        elif args.objective == "subtb":
            loss, batch, pb_loss, pb_deviation = SubTB_train_step(model, target_model, optimizer, pb_optimizer, M, args)
        elif args.objective == "dqn":
            # First, collect experiences for experience replay
            batch, pb_loss, pb_deviation = SoftDQN_collect_experience(
                rb, model, target_model, pb_optimizer, M, args, it > args.start_learning
            )
            # Next, sample transitions from the buffer and calculate the loss
            if it > args.start_learning:
                loss = SoftDQN_learn_rb(progress, rb, model, target_model, optimizer, M, args)
            else:
                loss = 0.0

        if args.objective == "dqn":
            if it % args.update_target_every == 0:
                target_model.load_state_dict(model.state_dict())
        else:
            with torch.no_grad():
                for param, target_param in zip(model.parameters(), target_model.parameters()):
                    target_param.data.mul_(1 - args.tau)
                    torch.add(target_param.data, param.data, alpha=args.tau, out=target_param.data)

        if b_lr_scheduler:
            b_lr_scheduler.step()

        sum_rewards += (batch_rewards(batch, M, args.k) ** args.reward_exponent).sum().item() / args.batch_size

        batch_strings = [token_seq_to_str(seq, args.k) for seq in batch]
        for m in range(len(M)):
            if modes[m]:
                continue
            for i in range(args.batch_size):
                if distance(M[m], batch_strings[i]) <= args.mode_threshold:
                    modes[m] = True
                    break

        logs_to_save["loss"].append(loss)
        logs_to_save["pb_loss"].append(pb_loss)
        logs_to_save["pb_deviation"].append(pb_deviation)
        logs_to_save["num_modes"].append(sum(modes))
        if args.log_grad_norm:
            logs_to_save["grad_norm"].append(grad_norm)
            logs_to_save["pb_grad_norm"].append(pb_grad_norm)

        if it > 0 and it % args.print_every == 0:
            blr = b_lr_scheduler.get_last_lr()[0] if args.backward_approach == "tlm" else 0
            print(
                f"{it=}\tloss: {loss:.4f}\tpb_loss: {pb_loss:.4f}\tpb_deviation: {pb_deviation:.6f}\t"
                f"num_modes: {sum(modes)}\tavg_reward: {sum_rewards / args.print_every}\t"
                f"logZ: {logZ.sum().cpu().item():.6f}\tblr: {blr:.6f}"
            )
            np.save(f"{experiment_name}/loss.npy", logs_to_save["loss"])
            np.save(f"{experiment_name}/pb_loss.npy", logs_to_save["pb_loss"])
            np.save(f"{experiment_name}/pb_deviation.npy", logs_to_save["pb_deviation"])
            np.save(f"{experiment_name}/num_modes.npy", logs_to_save["num_modes"])
            np.save(f"{experiment_name}/grad_norm.npy", logs_to_save["grad_norm"])
            np.save(f"{experiment_name}/pb_grad_norm.npy", logs_to_save["pb_grad_norm"])
            sum_rewards = 0.0

        if it > 0 and it % args.validate_every == 0:
            if args.print_modes:
                print("found modes:")
                for m in range(len(M)):
                    if modes[m]:
                        print(M[m])
            mode_nums.append(sum(modes))

            try:
                corr = compute_correlation(target_model, M, test_set, args, rounds=args.corr_num_rounds)
            except:
                corr = 0
            print(f"reward correlation with uniform backward:\t{corr:.3f}")
            corr_nums.append(corr)
            logs_to_save["corr_w_uniform"].append(corr)
            np.save(f"{experiment_name}/corr_w_uniform.npy", logs_to_save["corr_w_uniform"])
            if args.backward_approach != "uniform":
                try:
                    corr = compute_correlation_wpb(target_model, M, test_set, args, rounds=args.corr_num_rounds)
                except:
                    corr = 0
                print(f"reward correlation with naive backward:\t{corr:.3f}")
                logs_to_save["corr_w_naive"].append(corr)
                np.save(f"{experiment_name}/corr_w_naive.npy", logs_to_save["corr_w_naive"])
            else:
                np.save(f"{experiment_name}/corr_w_naive.npy", logs_to_save["corr_w_uniform"])

            print(f"spent minutes:\t{(time.time() - previous_time) / 60:.2f}")
            previous_time = time.time()

            np.save(f"{experiment_name}/num_modes.npy", logs_to_save["num_modes"])


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
