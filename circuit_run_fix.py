import argparse
import numpy as np
import math
import random
from scipy.stats import spearmanr

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.distributions.categorical import Categorical

from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from tensordict import TensorDict

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

parser = argparse.ArgumentParser()

# --- 변경된 파라미터 ---
# n: 생성할 게이트 시퀀스의 길이
# k: 불필요하므로 삭제
# M_size: 생성할 목표(Mode) 시퀀스의 개수
parser.add_argument("--n", default=12, type=int, help="Length of the gate sequence to generate.")
parser.add_argument("--M_size", default=16, type=int, help="Number of target mode sequences to generate.")
parser.add_argument("--mode_threshold", default=0.95, type=float, help="Similarity threshold to consider a mode 'found'.")
parser.add_argument("--reward_exponent", default=2.0, type=float)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--n_qubits", default=3, type=int)

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


ALPHABET = {
    'h': lambda qc, q: qc.h(q),
    'x': lambda qc, q: qc.x(q),
    'z': lambda qc, q: qc.z(q),
    'cx': lambda qc, q, t: qc.cx(q, t),
    'ccx': lambda qc, q, t, u: qc.ccx(q, t, u),
}

def construct_gates(n_qubits: int):
    gates = []
    qubit_indices = list(range(n_qubits))
    for q in qubit_indices:
        gates.extend([('h', q), ('x', q), ('z', q)])
    for q1 in qubit_indices:
        for q2 in qubit_indices:
            if q1 != q2:
                gates.append(('cx', q1, q2))
    # CCX 게이트는 n_qubits가 3 이상일 때만 의미가 있음
    if n_qubits >= 3:
        for q1 in qubit_indices:
            for q2 in qubit_indices:
                for q3 in qubit_indices:
                    if len(set([q1, q2, q3])) == 3:
                        gates.append(('ccx', q1, q2, q3))
    return gates

def sequence_to_unitary(action_list, gates, n_qubits: int):
    qc = QuantumCircuit(n_qubits)
    # action_list는 이제 정수 토큰의 리스트
    for idx in action_list:
        if idx >= len(gates): # 'empty' 토큰은 무시
            continue
        
        gate_info = gates[idx]
        gate_name = gate_info[0]
        qubits = gate_info[1:]
        
        ALPHABET[gate_name](qc, *qubits)
        
    U = Operator(qc).data
    return U.astype(np.complex64)

# --- 새로운 함수: 유사도(Similarity) 측정 ---
def similarity(U, V):
    # 두 Unitary 행렬 간의 유사도를 측정 (0에서 1 사이의 값)
    trace_val = np.trace(U.conj().T @ V)
    return (np.abs(trace_val) / U.shape[0])**2

def M_similarity(seq, M, gates, n_qubits: int):
    # 주어진 시퀀스가 목표(M) 시퀀스들 중 가장 유사한 것과의 유사도를 반환
    U_seq = sequence_to_unitary(seq, gates, n_qubits)
    max_sim = 0.0
    for target_seq in M:
        U_target = sequence_to_unitary(target_seq, gates, n_qubits)
        sim = similarity(U_seq, U_target)
        if sim > max_sim:
            max_sim = sim
    return max_sim

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
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        # --- 수정: 출력 레이어 ---
        # 위치 로짓 + 단어 로짓을 예측
        self.linear = nn.Linear(d_model, ntoken + seq_len)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        src = self.embedding(src) 
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output

# --- 수정: 목표 모드(M) 생성 함수 ---
def construct_M_modes(n, gates, M_size, seed=0):
    np.random.seed(seed) 
    M = []
    for _ in range(M_size):
        # 게이트 인덱스로 이루어진 리스트를 생성
        action_indices = np.random.choice(len(gates), size=n, replace=True).tolist()
        M.append(action_indices)
    return M

# --- 수정: 테스트셋 생성 함수 ---
def construct_test_set(M, gates, seed=0):
    random.seed(seed)
    test_set = []
    for s in M:
        test_set.append(s)
        for cnt in range(1, len(s)):
            new_s = s[:]
            subset = random.sample(range(len(s)), cnt)
            for i in subset:
                original_gate_idx = new_s[i]
                possible_replacements = [idx for idx in range(len(gates)) if idx != original_gate_idx]
                if possible_replacements:
                    new_s[i] = random.choice(possible_replacements)
            test_set.append(new_s)
    return test_set

def batch_final_rewards(batch, M, gates, n_qubits: int):
    batch_np = batch.cpu().numpy()
    rewards = [M_similarity(batch_np[i], M, gates, n_qubits) for i in range(batch_np.shape[0])]
    return torch.tensor(rewards, dtype=torch.float32)

# --- 수정: 로짓 처리 함수 ---
def process_logits(all_logits, pos_mask, args, gates):
    seq_len = args.n + 1
    vocab_size = len(gates)
    
    pos_logits = all_logits[0, :, -seq_len:]
    pos_logits[pos_mask] = float("-inf")
    word_logits = all_logits[:, :, :vocab_size]
    
    sum_logits = torch.moveaxis(word_logits, 1, 0) + pos_logits[:, :, None]
    sum_logits = sum_logits.reshape(pos_logits.shape[0], seq_len * vocab_size)
    return pos_logits, word_logits, sum_logits

# --- 수정: 샘플링 함수 ---
def sample_forward(sum_logits, sum_uniform, batch, args, gates):
    vocab_size = len(gates)
    while True:
        actions = Categorical(logits=sum_logits.clone()).sample()
        uniform_actions = Categorical(logits=sum_uniform).sample().to(args.device)
        uniform_mask = (torch.rand(args.batch_size, device=args.device) < args.rand_action_prob)
        actions[uniform_mask] = uniform_actions[uniform_mask]
        
        positions = actions // vocab_size
        # 배치에서 해당 위치가 비어있는지 확인
        if (batch.gather(1, positions.unsqueeze(1)).squeeze(1) == vocab_size).all():
            break
            
    words = actions % vocab_size
    return actions, positions, words

def SoftDQN_collect_experience(rb, model, target_model, M, args, gates):
    n_qubits = args.n_qubits
    vocab_size = len(gates)
    bos_token = vocab_size + 1
    empty_token = vocab_size

    batch = torch.full((args.batch_size, args.n + 1), empty_token, dtype=torch.long, device=args.device)
    batch[:, 0] = bos_token
    
    with torch.no_grad():
        for i in range(args.n):
            pos_mask = (batch != empty_token)
        
            all_logits = model(batch.T)
            _, _, sum_logits = process_logits(all_logits, pos_mask, args, gates)
            # Uniform logits for random sampling
            uniform_logits = torch.zeros_like(all_logits)
            _, _, sum_uniform = process_logits(uniform_logits, pos_mask, args, gates)

            actions, positions, words = sample_forward(sum_logits, sum_uniform, batch, args, gates)

            next_batch = batch.clone()
            # positions 텐서를 인덱싱에 적합한 형태로 변경
            next_batch.scatter_(1, positions.unsqueeze(1), words.unsqueeze(1))
            
            rewards = torch.log(torch.tensor([1 / (i+1)] * args.batch_size, device=args.device)) 

            if i + 1 == args.n:
                final_rewards = batch_final_rewards(next_batch[:, 1:], M, gates, n_qubits).to(args.device)
                rewards += args.reward_exponent * final_rewards
                is_done = torch.ones(args.batch_size, device=args.device)
            else:
                is_done = torch.zeros(args.batch_size, device=args.device)

            rb_record = TensorDict({
                "state": batch, "action": actions, "next_state": next_batch,
                "rewards": rewards, "is_done": is_done,
            }, batch_size=args.batch_size)
            rb.extend(rb_record)
            batch = next_batch

    assert batch[:, 1:].max() < vocab_size
    return batch[:, 1:].cpu()
    
def SoftDQN_learn_rb(progress, rb, model, target_model, optimizer, M, args, gates):
    loss_fn = torch.nn.HuberLoss(reduction='none') if args.softdqn_loss == 'Huber' else torch.nn.MSELoss(reduction='none')
    if args.anneal_per_beta:
        rb._sampler._beta = min(1.0, args.per_beta + (1. - args.per_beta) * progress)

    model.train()
    optimizer.zero_grad()

    rb_batch = rb.sample().to(args.device)
    
    pos_mask = rb_batch["state"] != len(gates)
    all_logits = model(rb_batch["state"].T)
    _, _, sum_logits = process_logits(all_logits, pos_mask, args, gates)
    
    q_values = sum_logits.gather(1, rb_batch["action"].unsqueeze(1)).squeeze(1)
    
    with torch.no_grad():
        pos_mask_next = rb_batch["next_state"] != len(gates)
        all_target_logits_next = target_model(rb_batch["next_state"].T)
        _, _, sum_target_logits_next = process_logits(all_target_logits_next, pos_mask_next, args, gates)
        
        target_v_next_values = args.entropy_coeff * torch.logsumexp(sum_target_logits_next / args.entropy_coeff, dim=-1)
        target_v_next_values[rb_batch["is_done"].bool()] = 0.0
        td_target = rb_batch["rewards"] + target_v_next_values
        
        if args.m_alpha > 0:
            pos_mask_state = rb_batch["state"] != len(gates)
            all_target_logits_state = target_model(rb_batch["state"].T)
            _, _, sum_target_logits_state = process_logits(all_target_logits_state, pos_mask_state, args, gates)
            norm_target_logits = sum_target_logits_state / args.entropy_coeff
            
            target_log_policy = norm_target_logits.gather(1, rb_batch["action"].unsqueeze(1)).squeeze(1) - torch.logsumexp(norm_target_logits, dim=-1)
            munchausen_penalty = torch.clamp(args.entropy_coeff * target_log_policy, min=args.m_l0, max=0)
            td_target += args.m_alpha * munchausen_penalty
    
    td_errors = loss_fn(q_values, td_target)
    td_errors[rb_batch["is_done"].bool()] *= args.leaf_coeff

    rb_batch["td_error"] = td_errors.detach()
    rb.update_tensordict_priority(rb_batch)

    loss = (td_errors * rb_batch["_weight"]).mean()
    loss.backward()
    optimizer.step()

    return loss.item()

# --- 대대적으로 수정된 `compute_correlation` 함수 ---
def compute_correlation(model, M, test_set, args, gates, rounds=10, batch_size=180):
    model.eval()
    n_qubits = args.n_qubits
    assert len(test_set) % batch_size == 0
    p_forward_sums = torch.zeros(len(test_set), rounds, device=args.device)
    
    vocab_size = len(gates)
    bos_token = vocab_size + 1
    empty_token = vocab_size

    for round_idx in range(rounds):
        for batch_idx in range(len(test_set) // batch_size):
            # 각 라운드, 각 배치마다 초기화
            batch = torch.full((batch_size, args.n + 1), empty_token, dtype=torch.long, device=args.device)
            batch[:, 0] = bos_token
            
            current_test_batch = test_set[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            
            # 순서를 무작위로 섞어 게이트를 하나씩 채워넣음
            positions_to_fill = list(range(1, args.n + 1))
            random.shuffle(positions_to_fill)

            for i, pos in enumerate(positions_to_fill):
                with torch.no_grad():
                    pos_mask = (batch != empty_token)
                    all_logits = model(batch.T)
                    _, _, sum_logits = process_logits(all_logits, pos_mask, args, gates)

                    words = [s[pos-1] for s in current_test_batch] # Get the correct gate token from test set
                    words = torch.tensor(words, dtype=torch.long, device=args.device)
                    
                    actions = (pos * vocab_size) + words
                    
                    log_pf = sum_logits.gather(1, actions.unsqueeze(1)).squeeze(1) / args.entropy_coeff - torch.logsumexp(sum_logits / args.entropy_coeff, dim=-1)
                    p_forward_sums[batch_idx * batch_size:(batch_idx + 1) * batch_size, round_idx] += log_pf
                    
                    # 배치 업데이트
                    batch.scatter_(1, torch.full((batch_size, 1), pos, device=args.device), words.unsqueeze(1))

    p_forward_sum = torch.logsumexp(p_forward_sums, dim=-1)
    log_rewards = np.array([M_similarity(s, M, gates, n_qubits) for s in test_set])
    
    # spearmanr requires at least 2 data points with variance
    if len(np.unique(log_rewards)) < 2 or len(np.unique(p_forward_sum.cpu().numpy())) < 2:
        return (0.0, 1.0) # Return a neutral correlation if no variance
        
    return spearmanr(args.reward_exponent * log_rewards, p_forward_sum.detach().cpu().numpy())

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    n_qubits = args.n_qubits
    gates = construct_gates(n_qubits)
    print(f"Constructed {len(gates)} possible gates for {n_qubits} qubits.")
    
    # U를 M(Modes)으로 이름 변경, 목표가 되는 '액션 시퀀스' 리스트
    M = construct_M_modes(args.n, gates, args.M_size, seed=args.seed)
    test_set = construct_test_set(M, gates, seed=args.seed)
    print(f"Test set size: {len(test_set)}")

    # --- 수정: 모델 초기화 ---
    ntoken = len(gates) + 2  # gates + empty_token + bos_token
    seq_len = args.n + 1   # sequence + bos_token
    
    model = TransformerModel(ntoken=ntoken, d_model=64, d_hid=64, nhead=8, nlayers=3, 
                             seq_len=seq_len, dropout=args.dropout).to(device)
    target_model = TransformerModel(ntoken=ntoken, d_model=64, d_hid=64, nhead=8, nlayers=3, 
                                    seq_len=seq_len, dropout=args.dropout).to(device)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=1e-5)

    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(args.rb_size, device=device),
        sampler=PrioritizedSampler(max_capacity=args.rb_size, alpha=args.per_alpha, beta=args.per_beta),
        batch_size=args.rb_batch_size,
        priority_key="td_error"
    )
    
    modes = [False] * len(M)
    total_final_reward = 0.0

    args.entropy_coeff *= 1/(1 - args.m_alpha) if args.m_alpha < 1 else 1.0
    
    for it in range(args.num_iterations + 1):
        progress = float(it) / args.num_iterations
        
        batch = SoftDQN_collect_experience(rb, model, target_model, M, args, gates)
        
        if it > args.start_learning:
            loss = SoftDQN_learn_rb(progress, rb, model, target_model, optimizer, M, args, gates)
        else:
            loss = 0.0

        if it % args.update_target_every == 0:
            target_model.load_state_dict(model.state_dict())
        
        # --- 수정: 보상 계산 로직 ---
        # 최종 상태의 보상만 집계
        final_rewards = batch_final_rewards(batch, M, gates, n_qubits)
        total_final_reward += final_rewards.sum().item()

        for m in range(len(M)):
            if modes[m]:
                continue
            for i in range(args.batch_size):
                # 생성된 시퀀스와 목표 M 간의 유사도 확인
                if M_similarity(batch[i].tolist(), [M[m]], gates, n_qubits) >= args.mode_threshold:
                    modes[m] = True
                    break
        
        if it > 0 and it % args.print_every == 0:
            avg_reward = total_final_reward / (args.print_every * args.batch_size)
            print(f"Iter: {it}, Loss: {loss:.4f}, Modes found: {sum(modes)}/{len(M)}, Avg final reward: {avg_reward:.4f}")
            total_final_reward = 0.0

        if it > 0 and it % 2000 == 0 and it > args.start_learning:
            if args.print_modes:
                print("Found modes (token sequences):")
                for m_idx, found in enumerate(modes):
                    if found:
                        print(f"  - {M[m_idx]}")
            
            sp_corr = compute_correlation(model, M, test_set, args, gates, 
                                          rounds=args.corr_num_rounds, batch_size=args.batch_size)
            print(f"Test set reward correlation: {sp_corr.correlation:.4f} (p-value: {sp_corr.pvalue:.4f})")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)