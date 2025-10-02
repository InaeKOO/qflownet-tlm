"""
Quantum‑circuit synthesis with GFlowNet.
Focus: strict tensor‑shape consistency, clear separation of forward/backward
probabilities, and robust special‑token handling.
"""
import argparse
import os
import random
import time
from copy import deepcopy
from math import log

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from tensordict import TensorDict

from model import TransformerModel
from utils import (
    batch_length_rewards,
    batch_log_rewards,
    compute_correlation,
    compute_correlation_wpb,
    construct_action_list,
    sequence_to_unitary,
    unitary_distance,
)

# ────────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ────────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()

# Environment
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--n_qubits", type=int, default=3)
parser.add_argument("--max_length", type=int, default=12)
parser.add_argument("--reward_exponent", type=float, default=2.0)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--offline_epochs", type=int, default=10)
parser.add_argument("--offline_batch_size", type=int, default=64)
parser.add_argument("--device", type=str, default="cuda")

# Training
parser.add_argument("--num_iterations", type=int, default=10_000)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--z_lr", type=float, default=1e-3)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument(
    "--backward_approach", choices=["uniform", "tlm", "naive", "pessimistic"], default="tlm"
)
parser.add_argument("--tau", type=float, default=0.1)  # target‑network update rate (for TLM)
parser.add_argument("--print_every", type=int, default=100)
parser.add_argument("--validate_every", type=int, default=2_000)

# Replay‑buffer
parser.add_argument("--rb_size", type=int, default=100_000)
parser.add_argument("--rb_batch_size", type=int, default=256)
parser.add_argument("--per_alpha", type=float, default=0.9)
parser.add_argument("--per_beta", type=float, default=0.1)

# ────────────────────────────────────────────────────────────────────────────────
# Dataset utilities
# ────────────────────────────────────────────────────────────────────────────────

def _load_npz(path: str):
    data = np.load(path, allow_pickle=True)
    return data["seqs"], data["unitaries"], data["lengths"]


class OfflineCircuitDataset(Dataset):
    """Pre‑generated circuits and target unitaries."""

    def __init__(self, path: str, pad_token: int):
        self.seqs, self.unitaries, self.lengths = _load_npz(path)
        self.pad_token = pad_token

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return (
            torch.as_tensor(self.seqs[idx], dtype=torch.long),
            torch.as_tensor(self.unitaries[idx]),
            torch.as_tensor(self.lengths[idx], dtype=torch.long),
        )


# ────────────────────────────────────────────────────────────────────────────────
# Training step (Trajectory Balance)
# ────────────────────────────────────────────────────────────────────────────────

def tb_train_step(
    model,
    target_model,
    logZ: torch.nn.Parameter,
    optim_f,
    optim_b,
    optim_z,
    seq_batch: torch.Tensor,  # [B, L]
    U_batch: torch.Tensor,
    length_batch: torch.Tensor,
    action_list,
    args,
):
    """One Trajectory‑Balance update on a batch."""

    device = args.device
    model.train()
    B, L = seq_batch.shape

    # Special tokens derived from action‑list length (set in main).
    pad_idx = args.num_actions  # last action id
    sos_idx = args.num_actions + 1  # BOS id

    # Assemble input with BOS.
    inp = torch.full((B, L + 1), pad_idx, dtype=torch.long, device=device)
    inp[:, 0] = sos_idx
    inp[:, 1:] = seq_batch.to(device)

    # Forward probabilities (all_logits: [L+1, B, V])
    all_logits, _ = model(inp.T)

    # ── log P_f ──
    idx = torch.arange(B, device=device)
    sum_log_Pf = 0.0
    for t in range(1, L + 1):
        logits_t = all_logits[t]  # [B, V]
        a_t = seq_batch[:, t - 1].to(device)
        logp_t = logits_t[idx, a_t] - torch.logsumexp(logits_t, dim=-1)
        mask = (length_batch.to(device) >= t)
        sum_log_Pf += (logp_t * mask).sum()

    # ── rewards ──
    log_r = args.reward_exponent * batch_log_rewards(
        seq_batch, U_batch.cpu().numpy(), action_list, args.n_qubits
    ).to(device)

    # ── log P_b ──
    sum_log_Pb = 0.0
    if args.backward_approach == "uniform":
        sum_log_Pb = sum(torch.log(torch.full((), 1.0 / (t + 1), device=device)) for t in range(L))
    else:
        for t in range(1, L + 1):
            prefix = inp[:, : t + 1]  # [B, t+1]
            net = target_model if args.backward_approach == "tlm" else model
            # net returns (fwd_logits, back_logits)
            _, back_logits_seq = net(prefix.T)  # [t+1, B, V]
            logits_t = back_logits_seq[-1]  # last step [B, V]

            # Invalidate PAD/SOS
            logits_t[:, pad_idx] = -torch.inf
            logits_t[:, sos_idx] = -torch.inf

            a_t = seq_batch[:, t - 1].to(device)
            logpb_t = logits_t[idx, a_t] - torch.logsumexp(logits_t, dim=-1)
            seq_mask = (length_batch.to(device) >= t)
            sum_log_Pb += (logpb_t * seq_mask).sum()

    # ── TB loss ──
    target = log_r - logZ.sum()
    loss = ((sum_log_Pf - sum_log_Pb - target) ** 2).mean() / L

    # ── optimise ──
    optim_f.zero_grad()
    optim_b.zero_grad()
    optim_z.zero_grad()
    loss.backward()
    optim_f.step()
    optim_b.step()
    optim_z.step()

    return loss.item()


# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    # Actions & special tokens
    action_list = construct_action_list(args.n_qubits)
    args.num_actions = len(action_list)  # critical for PAD/SOS indexing
    vocab_size = args.num_actions + 2  # +PAD +SOS

    # Dataset / loader
    dataset = OfflineCircuitDataset(args.data_path, pad_token=args.num_actions)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Model definition
    model = TransformerModel(
        ntoken=vocab_size,
        d_model=64,
        d_hid=64,
        nhead=8,
        nlayers=3,
        seq_len=args.max_length,
        dropout=args.dropout,
        uniform_init=True,
    ).to(device)
    target_model = deepcopy(model)

    # Optimisers
    logZ = torch.nn.Parameter(torch.zeros(64, device=device))
    f_params = [p for n, p in model.named_parameters() if "pb_linear" not in n]
    b_params = [p for n, p in model.named_parameters() if "pb_linear" in n]

    optim_f = torch.optim.Adam(f_params, lr=args.lr, weight_decay=1e-5)
    optim_b = torch.optim.Adam(b_params, lr=args.lr, weight_decay=1e-5)
    optim_z = torch.optim.Adam([logZ], lr=args.z_lr, weight_decay=1e-5)

    # Replay buffer (placeholder for future on‑policy)
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(args.rb_size),
        sampler=PrioritizedSampler(
            max_capacity=args.rb_size, alpha=args.per_alpha, beta=args.per_beta
        ),
        batch_size=args.rb_batch_size,
        priority_key="td_error",
    )

    # Training loop
    running_reward = 0.0
    for it in range(1, args.num_iterations + 1):
        seqs, U, lens = next(iter(loader))  # simple loader‑cycle
        loss = tb_train_step(
            model,
            target_model,
            logZ,
            optim_f,
            optim_b,
            optim_z,
            seqs,
            U,
            lens,
            action_list,
            args,
        )

        running_reward += batch_length_rewards(seqs, action_list, args.n_qubits).mean().item()

        if it % args.print_every == 0:
            avg_r = running_reward / args.print_every
            print(f"iter {it:>6d}\tloss {loss:.4f}\tavg‑len‑reward {avg_r:.3f}\tlogZ {logZ.sum().item():.4f}")
            running_reward = 0.0

        # target‑network soft update (TLM) every validate_every steps
        if args.backward_approach == "tlm" and it % args.validate_every == 0:
            with torch.no_grad():
                for p, tp in zip(model.parameters(), target_model.parameters()):
                    tp.mul_(1 - args.tau).add_(args.tau * p)


if __name__ == "__main__":
    main(parser.parse_args())
