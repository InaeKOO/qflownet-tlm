import argparse
import numpy as np
import random
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, random_unitary

# --- 1. 파라미터 설정 ---
parser = argparse.ArgumentParser(description="GFlowNet for Unitary Decomposition")
parser.add_argument("--n_qubits", type=int, default=2, help="Number of qubits in the circuit.")
parser.add_argument("--max_seq_len", type=int, default=10, help="Maximum length of the gate sequence.")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
parser.add_argument("--num_training_steps", type=int, default=10000, help="Total number of training steps.")
parser.add_argument("--batch_size", type=int, default=64, help="Number of trajectories per training batch.")
parser.add_argument("--reward_beta", type=float, default=5.0, help="Temperature for reward scaling R = beta * similarity.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (cuda or cpu).")
parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")

# --- 2. 게이트 집합 및 유틸리티 함수 ---
def construct_gates_and_inverses(n_qubits: int):
    """사용 가능한 게이트와 그 역행렬(conjugate transpose)을 생성합니다."""
    gate_unitaries = []
    
    qubit_indices = list(range(n_qubits))
    
    # --- 핵심 수정: 각 게이트를 n_qubits 회로에 적용하여 Operator 생성 ---
    
    # 단일 큐비트 게이트
    for q in qubit_indices:
        # H gate
        qc_h = QuantumCircuit(n_qubits)
        qc_h.h(q)
        gate_unitaries.append(Operator(qc_h).data)
        
        # X gate
        qc_x = QuantumCircuit(n_qubits)
        qc_x.x(q)
        gate_unitaries.append(Operator(qc_x).data)
        
        # S gate
        qc_s = QuantumCircuit(n_qubits)
        qc_s.s(q)
        gate_unitaries.append(Operator(qc_s).data)

    # 2-큐비트 게이트
    if n_qubits >= 2:
        for q1 in qubit_indices:
            for q2 in qubit_indices:
                if q1 != q2:
                    qc_cx = QuantumCircuit(n_qubits)
                    qc_cx.cx(q1, q2)
                    gate_unitaries.append(Operator(qc_cx).data)
    
    # 역행렬 계산
    gate_inverses = [u.conj().T for u in gate_unitaries]
    
    return gate_unitaries, gate_inverses
def unitary_to_tensor(unitary: np.ndarray, device: torch.device) -> torch.Tensor:
    """Unitary 행렬(NumPy)을 CNN 입력 텐서(PyTorch)로 변환합니다."""
    # (2, D, D) 형태로 변환: 채널 0은 실수부, 채널 1은 허수부
    tensor = torch.tensor(np.stack([unitary.real, unitary.imag]), dtype=torch.float32)
    return tensor.to(device)

def calculate_similarity(U, V):
    """두 Unitary 행렬 간의 유사도를 측정합니다."""
    trace_val = np.trace(U.conj().T @ V)
    dim = U.shape[0]
    return (np.abs(trace_val) / dim)**2

# --- 3. 모델: Unitary 행렬을 읽는 CNN ---
class UnitaryCNN(nn.Module):
    def __init__(self, n_qubits: int, num_gates: int):
        super().__init__()
        dim = 2**n_qubits
        self.conv_stack = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * dim * dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_gates) # 최종 출력: 각 게이트에 대한 로짓
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """입력: (Batch, 2, Dim, Dim), 출력: (Batch, Num_gates)"""
        x = self.conv_stack(x)
        logits = self.fc_stack(x)
        return logits

# --- 4. 메인 학습 로직 ---
def main(args):
    torch.manual_seed(args.seed)
    # --- 핵심 수정 1: NumPy의 마스터 난수 생성기(rng)를 초기화합니다. ---
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    gates, gate_inverses = construct_gates_and_inverses(args.n_qubits)
    num_gates = len(gates)
    identity_matrix = np.eye(2**args.n_qubits)
    print(f"Constructed {num_gates} gates for {args.n_qubits} qubits.")

    model = UnitaryCNN(n_qubits=args.n_qubits, num_gates=num_gates).to(device)
    log_Z = nn.Parameter(torch.zeros(1, device=device))

    optimizer = torch.optim.Adam(list(model.parameters()) + [log_Z], lr=args.learning_rate)
    
    print("Starting training...")
    for step in range(args.num_training_steps):
        batch_loss = 0.0
        
        # --- 궤적(Trajectory) 생성 ---
        for _ in range(args.batch_size):
            # --- 핵심 수정 2: 마스터 생성기를 사용해 새로운 정수 시드를 만듭니다. ---
            random_seed = rng.integers(np.iinfo(np.int32).max)
            target_unitary = random_unitary(2**args.n_qubits, seed=random_seed).data
            
            current_unitary = target_unitary
            
            log_forward_prob = 0.0
            
            # (이하 루프 내용은 동일합니다)
            for _ in range(args.max_seq_len):
                state_tensor = unitary_to_tensor(current_unitary, device).unsqueeze(0)
                action_logits = model(state_tensor).squeeze(0)
                action_dist = Categorical(logits=action_logits)
                action_idx = action_dist.sample()
                log_forward_prob += action_dist.log_prob(action_idx)
                gate_inv = gate_inverses[action_idx.item()]
                current_unitary = gate_inv @ current_unitary
                
                similarity_to_I = calculate_similarity(current_unitary, identity_matrix)
                if similarity_to_I > 0.999:
                    break
            
            reward = args.reward_beta * (similarity_to_I - 1.0)
            log_backward_prob = 0.0 
            traj_loss = (log_Z + log_forward_prob - log_backward_prob - reward)**2
            batch_loss += traj_loss

        # (이하 업데이트 로직은 동일합니다)
        loss = batch_loss / args.batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Step {step:5d} | Loss: {loss.item():.4f} | log Z: {log_Z.item():.4f}")
            
    print("Training finished.")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)