import argparse
import random
import os
import time
import numpy as np
from math import log
from copy import deepcopy

import torch
from torch.distributions.categorical import Categorical
from torch.utils.data import Dataset, DataLoader
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from tensordict import TensorDict

from model import TransformerModel
from utils import (
    construct_action_list,
    transpile_action_sequence,
    sequence_to_unitary,
    unitary_distance,
    log_reward,
    process_logits,
    batch_log_rewards,
    batch_rewards,
    compute_correlation,
    compute_correlation_wpb
)

parser = argparse.ArgumentParser()

# Environment params
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--n_qubits", type=int, default=3)
parser.add_argument("--num_actions", type=int, default=18)
parser.add_argument("--max_length", type=int, default=12)
parser.add_argument("--data_path", type=str, default=None,
                    help="Path to .npz dataset for offline quantum-circuit training")
parser.add_argument("--offline_epochs", type=int, default=10,
                    help="Number of epochs for offline pretraining")
parser.add_argument("--offline_batch_size", type=int, default=64,
                    help="Batch size for offline pretraining")

parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--print_every", default=100, type=int)
parser.add_argument("--validate_every", default=2000, type=int)
parser.add_argument("--print_modes", default=False, action="store_true")
parser.add_argument("--log_grad_norm", default=False, action="store_true")

# Base training params
parser.add_argument("--num_iterations", default=10000, type=int)
parser.add_argument("--rand_action_prob", default=0.001, type=float)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--blr", type=float)
parser.add_argument("--gamma", default=0.9999, type=float)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument(
    "--backward_approach", default="tlm", choices=["uniform", "tlm", "naive", "pessimistic"], type=str
)
parser.add_argument("--uniform_init", action="store_false")

parser.add_argument("--objective", default="tb", choices=["tb", "db", "subtb", "dqn"], type=str)
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

def load_data(path):
    """
    Load pre-generated gate sequences and target unitaries from an .npz file.
    Assumes keys 'sequences' (array of token lists) and 'U' (array of matrices).
    """
    npz = np.load(path, allow_pickle=True)
    return npz['seqs'], npz['unitaries']

class OfflineCircuitDataset(Dataset):
    def __init__(self, path):
        self.seqs, self.unitaries = load_data(path)
        self.pad = args.num_actions
        self.max_len = self.seqs.shape[1]

    def __len__(self):
        return self.seqs.shape[0]

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        inp = [self.pad+1] + list(seq) + [self.pad] * (self.max_len - len(seq))
        action = len(seq) * (args.num_actions) + int(seq[len(seq)-1])
        return torch.tensor(inp, dtype=torch.long), torch.tensor(action, dtype=torch.long)


def TB_train_step(model, target_model, logZ, optimizer, Z_optimizer, pb_optimizer, batch_seqs, unitaries, action_list, args):
    global pessimistic_buffer
    # This code is pretty simple because all trajectories in our graph have the same length.
    model.train()
    B,L = batch_seqs.shape
    # The seqence has length n/k + 1 and at the beginning looks like [2^k + 1, 2^k, 2^k, ..., 2^k].
    # 2^k + 1: [BOS] token, 2^k: token for "empty" word.
    sos_idx = args.num_actions + 1
    pad_idx = args.num_actions
    inp = torch.full((B, L+1), pad_idx, dtype=torch.long, device=args.device)
    inp[:, 0] = sos_idx
    inp[:, 1:1+L] = batch_seqs

    all_logits, _ = model(inp.T)

    sumlogPf = 0.0
    for t in range(1, L+1):
        logits_t = all_logits[t].T            # [B, V]
        a_t      = batch_seqs[:, t-1]         # [B]
        logp     = logits_t[range(B), a_t] \
                 - torch.logsumexp(logits_t, dim=-1)
        sumlogPf += logp

    log_rewards = args.reward_exponent * \
        batch_log_rewards(batch_seqs, unitaries, action_list, args.n_qubits) \
        .to(args.device).detach()         # [B]

    sumlogPb = 0.0
    if args.backward_approach == "uniform":
        # uniform backward: each step에 1/(t+1)
        for t in range(L):
            sumlogPb += torch.log(1.0/(t+1))
    else:
        for t in range(1, L+1):
            # prefix including the chosen token at t
            prefix = inp[:, :t+1]            # [B, t+1]
            # if tlm, use target_model, otherwise use model
            net = target_model if args.backward_approach=="tlm" else model
            _, pb_all_logits = net(prefix.T)  # [t+1, B, V] but we only need pos t
            logits_t = pb_all_logits[t].T     # [B, V]

            # mask out invalid tokens (PAD/SOS) if needed
            mask = torch.ones_like(logits_t, dtype=torch.bool)
            mask[:, pad_idx] = False
            mask[:, sos_idx] = False
            logits_t[~mask] = -float('inf')

            a_t  = batch_seqs[:, t-1]         # [B]
            logpb = logits_t[range(B), a_t] \
                  - torch.logsumexp(logits_t, dim=-1)
            sumlogPb += logpb

    target = log_rewards - logZ.sum()    # [B]
    loss = ((sumlogPf - sumlogPb - target)**2).mean() / L

    optimizer.zero_grad()
    Z_optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    Z_optimizer.step()

    return loss.item()


def main(args):
    device = args.device
    assert args.validate_every % args.print_every == 0
    action_list = construct_action_list(args.n_qubits)
    print(action_list)
    args.num_actions = len(action_list)
    #U = sequence_to_unitary([12,13,4,7,6,5,0,4,3,2,1,0],action_list, args.n_qubits)
    test_set = OfflineCircuitDataset(args.data_path)
    loader = DataLoader(
        test_set,
        batch_size=16,
        shuffle=True,             # 매 epoch마다 섞어 주면 좋습니다
        drop_last=True,           # 마지막 미니배치가 작으면 버릴 수도 있고
    )
    experiment_name = f"results/{args.seed}_n={args.n_qubits}"
    blr = args.lr if args.blr is None else args.lr
    gamma = args.gamma
    experiment_name += f"_lr={args.lr}_blr={blr}_lrg={gamma}_pb={args.backward_approach}"
    if args.backward_approach == "tlm":
        experiment_name += f"_tau={args.tau}"
    os.makedirs(experiment_name, exist_ok=True)
    print(experiment_name)

    model = TransformerModel(
        ntoken=args.num_actions + 2,
        d_model=64,
        d_hid=64,
        nhead=8,
        nlayers=3,
        seq_len=args.max_length,
        dropout=args.dropout,
        uniform_init=args.uniform_init,
    ).to(device)
    target_model = deepcopy(model)
    target_model.load_state_dict(model.state_dict())

    logZ = torch.nn.Parameter(torch.tensor(np.ones(64) * 0.0 / 64, requires_grad=True, device=device))

    f_params = [v for k, v in dict(model.named_parameters()).items() if not "pb_linear" in k]
    optimizer = torch.optim.Adam(f_params, args.lr, weight_decay=1e-5)
    b_params = [v for k, v in dict(model.named_parameters()).items() if "pb_linear" in k]

    pb_optimizer = torch.optim.Adam(b_params, blr, weight_decay=1e-5)
    b_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(pb_optimizer, gamma)
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

    modes = [False]
    sum_rewards = 0.0

    corr_nums = []
    mode_nums = []

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
        for seqs, unitaries in loader:
            if args.objective == "tb":
                loss, batch, pb_loss, pb_deviation = TB_train_step(
                    model, target_model, logZ, optimizer, Z_optimizer, pb_optimizer, seqs, unitaries, action_list, args
                )
            else:
                with torch.no_grad():
                    for param, target_param in zip(model.parameters(), target_model.parameters()):
                        target_param.data.mul_(1 - args.tau)
                        torch.add(target_param.data, param.data, alpha=args.tau, out=target_param.data)

            if b_lr_scheduler:
                b_lr_scheduler.step()

            sum_rewards += (batch_rewards(seqs, unitaries, action_list, args.n_qubits) ** args.reward_exponent).sum().item() / args.batch_size

            batch_strings = [seq for seq in seqs]
            if modes:
                continue
            for i in range(args.batch_size):
                if unitary_distance(test_set.unitaries[i], sequence_to_unitary(batch_strings[i], action_list, args.n_qubits)) <= 1e-6:
                    modes = True
                    break

            logs_to_save["loss"].append(loss)
            logs_to_save["pb_loss"].append(pb_loss)
            logs_to_save["pb_deviation"].append(pb_deviation)
            logs_to_save["num_modes"].append(sum(modes))

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
                    if modes:
                        print("found modes")
                mode_nums.append(sum(modes))

                try:
                    corr = compute_correlation(target_model, U, action_list, args.n_qubits, test_set, args, rounds=args.corr_num_rounds)
                except:
                    corr = 0
                print(f"reward correlation with uniform backward:\t{corr:.3f}")
                corr_nums.append(corr)
                logs_to_save["corr_w_uniform"].append(corr)
                np.save(f"{experiment_name}/corr_w_uniform.npy", logs_to_save["corr_w_uniform"])
                if args.backward_approach != "uniform":
                    try:
                        corr = compute_correlation_wpb(target_model, U, action_list, args.n_qubits, test_set, args, rounds=args.corr_num_rounds)
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
