import argparse
import numpy as np
import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple
from scipy.stats import spearmanr

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torch.distributions.categorical import Categorical

from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from tensordict import TensorDict

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator

parser = argparse.ArgumentParser()

parser.add_argument("--n", default=120, type=int)
parser.add_argument("--k", default=4, type=int)
parser.add_argument("--M_size", default=60, type=int)
parser.add_argument("--mode_threshold", default=30, type=int)
parser.add_argument("--reward_exponent", default=2.0, type=float)
parser.add_argument("--seed", default=0, type=int)

parser.add_argument("--device", default='cuda', type=str)
parser.add_argument("--num_iterations", default=50000, type=int)
parser.add_argument("--rand_action_prob", default=0.001, type=float)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--print_every", default=50, type=int)
parser.add_argument("--print_modes", default=False, action='store_true')

parser.add_argument("--leaf_coeff", default=5.0, type=float)
parser.add_argument("--update_target_every", default=5, type=int)
parser.add_argument("--corr_num_rounds", default=10, type=int)
parser.add_argument("--n_qubits", default=3, type=int)
parser.add_argument("--max_length", default=12, type=int)
parser.add_argument("--device", default='cuda', type=str)

# SoftDQN params
parser.add_argument("--start_learning", default=50, type=int)
parser.add_argument("--softdqn_loss", default='Huber', type=str)

# Replay buffer parameters
parser.add_argument("--rb_size", default=100_000, type=int)
parser.add_argument("--rb_batch_size", default=256, type=int)
parser.add_argument("--per_alpha", default=0.9, type=float)
parser.add_argument("--per_beta", default=0.1, type=float)
parser.add_argument("--anneal_per_beta", default=False, action='store_true')

# Munchausen DQN parameters
parser.add_argument("--m_alpha", default=0.0, type=float)
parser.add_argument("--entropy_coeff", default=1.0, type=float)
parser.add_argument("--m_l0", default=-25.0, type=float)

basis_gates = ['h', 'x', 'z', 'cx', 'ccx']

ALPHABET = {
    'h': lambda qc, q: qc.h(q),  # Hadamard on qubit q
    'x': lambda qc, q: qc.x(q),  # Pauli-X on qubit q
    'z': lambda qc, q: qc.z(q),  # Pauli-Z on qubit q
    'cx': lambda qc, q, t: qc.cx(q, t),  # CNOT: control q -> target t
    'ccx': lambda qc, q, t, u: qc.ccx(q, t, u),  # Toffoli: control q,t -> target u
}

def construct_gates(n_qubits: int):
    """
    Build a list of available actions as (gate_label, qubit_index) pairs.

    Supported gates:
      - 'h', 'x', 'z': single-qubit gates on each qubit
      - 'cx': controlled-NOT (CX) from qubit to qubit
      - 'ccx': Toffoli (CCX) using control qubits to target qubit

    Returns:
        List of tuples [(gate_label, qubit_index), ...]
    """
    gates = []
    for q in range(n_qubits):
        gates.extend([
            ('h', q),
            ('x', q),
            ('z', q),
        ])
    for q in range(n_qubits):
        for t in range(n_qubits):
            if q != t:
                gates.extend([
                    ('cx', q, t),
                ])
    for q in range(n_qubits):
        gates.extend([
            ('ccx', q, (q + 1) % n_qubits, (q + 2) % n_qubits),
        ])
    return gates

def sequence_to_unitary(action_list, gates, n_qubits: int):
    qc = QuantumCircuit(n_qubits)
    for idx in action_list:
        if idx == len(gates):
            break
        #print("current idx: ", idx)
        gate = gates[idx][0]
        q = gates[idx][1]
        if gate == 'ccx':
            ALPHABET[gate](qc, q, (q + 1) % n_qubits, (q + 2) % n_qubits)
        elif gate == 'cx':
            ALPHABET[gate](qc, q, gates[idx][2])
        else:
            ALPHABET[gate](qc, q)
    U = Operator(qc).data
    return U.astype(np.complex64)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.2, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, seq_len: int, dropout: float = 0.2):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len + 2)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken + seq_len + 1)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        src = self.embedding(src) 
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output


def construct_action_list(n, gates, action_list_size, seed=0):
    np.random.seed(seed) 
    action_list = []
    for i in range(action_list_size):
        action_list.append("".join([np.random.choice(gates) for _ in range(n)]))
        assert len(action_list[-1]) == n
    return action_list

def construct_U(n, gates, action_list_size, seed=0):
    np.random.seed(seed) 
    U = []
    for i in range(action_list_size):
        U.append(np.random.choice(len(gates)) for _ in range(n))
    return U

def distance(s1, s2, gates, n_qubits: int):
    U = sequence_to_unitary(s1, gates, n_qubits)
    V = sequence_to_unitary(s2, gates, n_qubits)
    return abs(np.trace(U.conj().T @ V)) / (2**n_qubits)

def construct_test_set(action_list, gates, n_qubits: int, seed=0):
    np.random.seed(seed) 
    test_set = []
    for s in action_list:
        test_set.append(s)
        for cnt in range(1, len(s)):
            new_s = list(s)
            subset = np.random.choice(list(range(len(s))), size=cnt, replace=False)
            for i in subset:
                original_gate = new_s[i]
                # 2. Create a list of other possible gates
                possible_replacements = [g for g in gates if g != original_gate]
                # 3. Choose a random gate from the other options
                if possible_replacements: # Should not be empty if len(gates) > 1
                    new_s[i] = np.random.choice(possible_replacements)
            test_set.append(new_s)
            assert len(test_set[-1]) == len(s)
            assert distance(test_set[-1], s, gates, n_qubits) == cnt
    return test_set


def reward(s1, s2, gates, n_qubits: int):
    return np.exp(distance(s1, s2, gates, n_qubits))


def batch_rewards(batch, M, k, gates, n_qubits: int):
    batch_np = batch.cpu().numpy()
    rewards = [reward(batch_np[i], M, gates, n_qubits) for i in range(batch_np.shape[0])]
    return torch.tensor(rewards)

def batch_log_rewards(batch, M, k, gates, n_qubits: int):
    batch_np = batch.cpu().numpy()
    log_rewards = [distance(batch_np[i], M, gates, n_qubits) for i in range(batch_np.shape[0])]
    return torch.tensor(log_rewards)

def process_logits(all_logits, pos_mask, args, gates):
    # Model predicts positional logits p_i and word logits for each position w_ij.
    # The logits used to sample pairs of positions and word (i, j) are computed as p_i + w_ij.
    pos_logits = all_logits[0, :, -(args.n + 1):] # [batch_size, n + 1]
    pos_logits[pos_mask] = float("-inf")
    word_logits = all_logits[:, :, :len(gates)] # [n + 1, batch_size, 2^k]
    sum_logits = torch.moveaxis(word_logits, 1, 0) + pos_logits[:, :, None] #[batch_size, n + 1, 2^k]
    sum_logits = sum_logits.reshape(pos_logits.shape[0], (args.n+ 1) * (len(gates))) #[batch_size, (n/k + 1) * 2^k]
    return pos_logits, word_logits, sum_logits

def sample_forward(sum_logits, sum_uniform, batch, args, gates):
    # There is a bug in pytorch that allows to sample objects that has 0 probability (happens very rarely but still happens).
    # This loop basically resamples until everything is correct.
    while True:
        actions = Categorical(logits=sum_logits.clone()).sample()
        uniform_actions = Categorical(logits=sum_uniform).sample().to(args.device)
        uniform_mask = torch.rand(args.batch_size) < args.rand_action_prob
        actions[uniform_mask] = uniform_actions[uniform_mask]
        positions = actions // (len(gates))
        if (batch[range(args.batch_size), positions] == len(gates)).sum() == args.batch_size:
            break
    assert positions.min() >= 1
    assert positions.max() <= args.n
    words = actions % (len(gates))
    return actions, positions, words


def SoftDQN_collect_experience(rb, model, target_model, M, args, gates):
    # This code is pretty simple because all trajectories in our graph have the same length.

    # The seqence has length n/k + 1 and at the beginning looks like [2^k + 1, 2^k, 2^k, ..., 2^k].
    # 2^k + 1: [BOS] token, 2^k: token for "empty" word.
    n_qubits = args.n_qubits
    batch = torch.tensor([[len(gates) + 1] + ([len(gates)] * (args.n)) for i in range(args.batch_size)]).to(args.device)
    with torch.no_grad():
        for i in range(args.n):
            pos_mask = batch != len(gates)
        
            all_logits = model(batch.T)
            _, _, sum_logits = process_logits(all_logits, pos_mask, args)
            _, _, sum_uniform = process_logits(0.0 * all_logits.clone(), pos_mask, args)

            actions, positions, words = sample_forward(sum_logits, sum_uniform, batch, args, gates)

            next_batch = batch.clone()
            next_batch[range(args.batch_size), positions] = words
            rewards = torch.log(torch.tensor([1 / (i+1)] * args.batch_size).to(args.device)) 

            # The last added word
            if i + 1 == args.n:
                rewards += args.reward_exponent * batch_log_rewards(next_batch[:, 1:], M, args.k, gates, n_qubits).to(args.device)
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
                batch_size=args.batch_size
            )
            rb.extend(rb_record) # add record to replay buffer
            batch = next_batch

    assert batch[:, 1:].max() < len(gates)
    return batch[:, 1:].cpu()
    

def SoftDQN_learn_rb(progress, rb, model, target_model, optimizer, M, args, gates):
    # Select loss function
    if args.softdqn_loss == 'Huber':
        loss_fn = torch.nn.HuberLoss(reduction='none')
    else:
        loss_fn = torch.nn.MSELoss(reduction='none')
    if args.anneal_per_beta:
        # Update beta parameter of experience replay
        add_beta = (1. - args.per_beta) * progress
        rb._sampler._beta = args.per_beta + add_beta

    model.train()
    optimizer.zero_grad()

    # Sample from replay buffer
    rb_batch = rb.sample().to(args.device)
    # Compute td-loss
    pos_mask = rb_batch["state"] != len(gates)
    all_logits = model(rb_batch["state"].T)
    _, _, sum_logits = process_logits(all_logits, pos_mask, args)
    if args.m_alpha > 0:
        all_target_logits = target_model(rb_batch["state"].T)
        _, _, sum_target_logits = process_logits(all_target_logits, pos_mask, args)
        norm_target_logits = sum_target_logits / args.entropy_coeff  

    q_values = sum_logits[range(args.rb_batch_size), rb_batch["action"]]
    
    with torch.no_grad():
        pos_mask = rb_batch["next_state"] != len(gates)
        all_target_logits = target_model(rb_batch["next_state"].T)
        _, _, sum_target_logits = process_logits(all_target_logits, pos_mask, args)
        target_v_next_values = args.entropy_coeff * torch.logsumexp(sum_target_logits / args.entropy_coeff, dim=-1)
        target_v_next_values[rb_batch["is_done"].bool()] = 0.0
        td_target = rb_batch["rewards"] + target_v_next_values
        
        if args.m_alpha > 0:
            target_log_policy = norm_target_logits[range(args.rb_batch_size), rb_batch["action"]] - torch.logsumexp(norm_target_logits, dim=-1)
            munchausen_penalty = torch.clamp(
                args.entropy_coeff * target_log_policy,
                min=args.m_l0, max=1
            )
            td_target += args.m_alpha * munchausen_penalty
    
    td_errors = loss_fn(q_values, td_target)
    td_errors[rb_batch["is_done"].bool()] *= args.leaf_coeff

    # Update PER
    rb_batch["td_error"] = td_errors
    rb.update_tensordict_priority(rb_batch)

    # Compute loss with IS correction
    loss = (td_errors * rb_batch["_weight"]).mean()
    #loss = td_errors.mean()
    loss.backward()
    optimizer.step()

    return loss.cpu().item()


def compute_correlation(model, M, test_set, args, gates, rounds=10, batch_size=180):
    # Sampling a trajectory from PB(tau | x) when PB is uniform over parents 
    # in this case is equvalent to starting at s0 and randomly choosing the order 
    # in which we replace empty words with words at corresponding positions from x.
    # Thus we can sample trajectories and compute PF(tau) in parallel.
    model.eval()
    n_qubits = args.n_qubits
    assert len(test_set) % batch_size == 0
    p_forward_sums = torch.zeros(len(test_set), rounds).to(args.device)

    for round in range(rounds):
        for batch_idx in range(len(test_set) // batch_size):
            batch = torch.tensor([[len(gates) + 1] + ([len(gates)] * (args.n)) for i in range(batch_size)]).to(args.device)
            for i in range(args.n // args.k):
                with torch.no_grad():
                    pos_mask = batch != len(gates)
                    all_logits = model(batch.T)
                    pos_logits, word_logits, sum_logits = process_logits(all_logits, pos_mask, args)

                    # There is a bug in pytorch that allows to sample objects that has 0 probability (happens very rarely but still happens).
                    # This loop basically resamples until everything is correct.
                    while True:
                        uniform_probs = torch.zeros(batch_size, args.n + 1) + 1 / (args.n - i)
                        uniform_probs[pos_mask] = 0.0
                        positions = Categorical(probs=uniform_probs).sample().to(args.device)
                        if (batch[range(batch_size), positions] == len(gates)).sum() == batch_size:
                            break

                    assert positions.min() >= 1
                    assert positions.max() <= args.n 

                    words = []
                    for j in range(batch_size):
                        s = test_set[batch_idx * batch_size + j]
                        word = int(s[(positions[j] - 1) * args.k:positions[j] * args.k], base=2)
                        words.append(word)
                    words = torch.tensor(words).to(args.device)
                    
                    batch_cl = batch.clone()
                    batch_cl[range(batch_size), positions] = words
                    batch = batch_cl

                    actions = positions * (len(gates)) + words
                    log_pf = sum_logits[range(batch_size), actions] / args.entropy_coeff - torch.logsumexp(sum_logits / args.entropy_coeff, dim=-1)
                    p_forward_sums[batch_idx * batch_size:(batch_idx + 1) * batch_size, round] += log_pf

    p_forward_sum = torch.logsumexp(p_forward_sums, dim=-1)
    log_rewards = np.array([distance(s, M, gates, n_qubits) for s in test_set])
    return spearmanr((args.reward_exponent * log_rewards), (p_forward_sum.detach().cpu().numpy()))


def main(args):
    torch.manual_seed(args.seed)
    device = args.device
    n_qubits = args.n_qubits
    gates = construct_gates(args.n_qubits)
    U = construct_U(args.n, gates, args.max_length, seed=args.seed)
    print(U)
    test_set = construct_test_set(U, seed=args.seed)
    print(f"test set size: {len(test_set)}")

    model = TransformerModel(ntoken=2**args.k+2, d_model=64, d_hid=64, nhead=8, nlayers=3, 
                             seq_len=args.n, dropout=args.dropout).to(device)
    target_model = TransformerModel(ntoken=2**args.k+2, d_model=64, d_hid=64, nhead=8, nlayers=3, 
                                    seq_len=args.n, dropout=args.dropout).to(device)
    target_model.load_state_dict(model.state_dict())
    
    log_Z = nn.Parameter(torch.tensor(np.ones(64) * 0.0 / 64, requires_grad=True, device=device))
    
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=1e-5)
    Z_optimizer = torch.optim.Adam([log_Z], args.z_learning_rate, weight_decay=1e-5)

    rb =  TensorDictReplayBuffer(
        storage=LazyTensorStorage(args.rb_size),
        sampler=PrioritizedSampler(
            max_capacity=args.rb_size,
            alpha=args.per_alpha,
            beta=args.per_beta,
        ),
        batch_size=args.rb_batch_size,
        priority_key="td_error"
    )
    
    modes = [False] * len(U)
    avg_reward = 0.0

    corr_nums = []
    mode_nums = []

    args.entropy_coeff *= 1/(1 - args.m_alpha)
    
    for it in range(args.num_iterations + 1):
        progress = float(it) / args.num_iterations
        # First, collect experiences for experience replay
        batch = SoftDQN_collect_experience(rb, model, target_model, U, args, gates)
        # Next, sample transitions from the buffer and calculate the loss
        if it > args.start_learning:
            loss = SoftDQN_learn_rb(progress, rb, model, target_model, optimizer, U, args, gates)
        else:
            loss = 0.0

        if it % args.update_target_every == 0:
            target_model.load_state_dict(model.state_dict())
        
        avg_reward += (batch_rewards(batch, U, args.k, gates, n_qubits) ** args.reward_exponent).sum().item() / args.batch_size

        batch_strings = [seq for seq in batch]
        for m in range(len(U)):
            if modes[m]:
                continue
            for i in range(args.batch_size):
                if distance(U[m], batch_strings[i], gates, n_qubits) <= args.mode_threshold:
                    modes[m] = True
                    break
        
        if it > 0 and it % args.print_every == 0:
            print(f"{it}, loss: {loss}, modes: {sum(modes)}, avg reward: {avg_reward / args.print_every}, log_Z: {log_Z.sum().cpu().item()}")
            avg_reward = 0.0

        if it > 0 and it % 2000 == 0:
            if args.print_modes:
                print("found modes:")
                for m in range(len(U)):
                    if modes[m]:
                        print(U[m])
            mode_nums.append(sum(modes))

            sp_corr = compute_correlation(model, U, test_set, args, gates, rounds=args.corr_num_rounds, batch_size=args.batch_size)
            corr_nums.append(sp_corr.statistic)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)