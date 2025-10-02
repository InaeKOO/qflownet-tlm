import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, random_unitary
import math

# --- [Heuristic Model] from unitary_num.py (이전과 동일, 생략) ---
# ... (Heuristic 모델 클래스들은 여기에 그대로 존재합니다) ...
class DownBlock2D(nn.Module):
    """A 2d down scale block."""
    def __init__(self, in_ch, out_ch, kernel_size=2, stride=2, padding=0, use_conv=True):
        super().__init__()
        self.use_conv = use_conv
        if self.use_conv:
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
            self.convId = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding="same") if in_ch!=out_ch else nn.Identity()

    def forward(self, x):
        if self.use_conv:
            x = self.conv1(x)
        else:
            x = self.avg_pool(x)
            x = self.convId(x)
        return x

class PositionalEncoding(nn.Module):
    """An absolute pos encoding layer."""
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: Tensor, shape [batch_size, seq_len , embedding_dim] """
        x = x + self.pe[None, :x.size(1)]
        return self.dropout(x)

class PositionalEncodingTransposed(PositionalEncoding):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__(d_model, dropout, max_len)
        self.pe = torch.permute(self.pe, (1, 0)) # [max_len, d_model] to [d_model, max_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: Tensor, shape [batch_size, embedding_dim, seq_len] """
        x = x + self.pe[None, :, :x.size(2)]
        return self.dropout(x)

class PositionalEncoding2D(PositionalEncodingTransposed):
    """A 2D absolute pos encoding layer."""
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__(d_model=d_model//2, dropout=dropout, max_len=max_len)
        self.d_model_half = d_model//2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: Tensor, shape [batch_size, gate_color, space , time] """
        p1 = self.pe[None, :, :x.size(2), None] #space encoding
        p2 = self.pe[None, :, None, :x.size(3)] #time encoding
        x[:, :self.d_model_half] = x[:, :self.d_model_half] + p1
        x[:, self.d_model_half:] = x[:, self.d_model_half:] + p2
        return self.dropout(x)

class FeedForward(nn.Module):
    """A small dense feed-forward network as used in `transformers`."""
    def __init__(self, in_ch, out_ch, inner_mult=1):
        super().__init__()
        self.proj1 = nn.Linear(in_ch, in_ch*inner_mult)
        self.proj2 = nn.Linear(in_ch*inner_mult, out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj1(x)
        x = self.act(x)
        x = self.proj2(x)
        return x

class BasisSelfAttnBlock(nn.Module):
    """A self attention block, i.e. a `transformer` encoder."""
    def __init__(self, ch, num_heads, dropout=0):
        super().__init__()
        self.self_att = nn.MultiheadAttention(ch, num_heads=num_heads, batch_first=False) #[t, b, c]
        self.ff = FeedForward(ch, ch)
        self.norm1 = nn.LayerNorm(ch)
        self.norm2 = nn.LayerNorm(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None, need_weights=False):
        self_out = self.norm1(x)
        self_out, _ = self.self_att(self_out, key=self_out, value=self_out, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=need_weights)
        self_out = self.drop(self_out) + x
        feed_out = self.norm2(self_out)
        feed_out = self.ff(feed_out)
        feed_out = self.drop(feed_out) + self_out
        return feed_out

class SpatialTransformerSelfAttn(nn.Module):
    """A spatial residual `transformer`, only uses self-attention."""
    def __init__(self, ch, num_heads, depth, dropout=0.0):
        super().__init__()
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=ch, eps=1e-6, affine=True)
        self.transformer_blocks = nn.ModuleList([BasisSelfAttnBlock(ch, num_heads, dropout) for d in range(depth)])

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        b, ch, space, time = x.shape
        x_in = x
        x = self.norm(x)
        x = torch.reshape(x, (b, ch, space*time))
        x = torch.permute(x, (2, 0, 1)).contiguous()
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, attn_mask, key_padding_mask)
        x = torch.permute(x, (1, 2, 0))
        x = torch.reshape(x, (b, ch, space, time)).contiguous()
        return x + x_in

class Unitary_encoder(nn.Module):
    """Encoder for unitary conditions."""
    def __init__(self, cond_emb_size, model_features=None, num_heads=8, transformer_depths=(4, 4), dropout=0.1):
        super().__init__()
        self.cond_emb_size = cond_emb_size
        if model_features is None:
            in_ch, mid_ch1, mid_ch2, out_ch = 2, cond_emb_size // 4, cond_emb_size // 2, cond_emb_size
        else:
            assert len(model_features) == 4
            in_ch, mid_ch1, mid_ch2, out_ch = model_features

        self.conv_in = nn.Conv2d(in_ch, mid_ch1, kernel_size=1, stride=1, padding=0)
        self.pos_enc = PositionalEncoding2D(d_model=mid_ch1)
        self.down1 = DownBlock2D(mid_ch1, mid_ch2, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        assert len(transformer_depths) == 2
        self.spatialTransformer1 = SpatialTransformerSelfAttn(mid_ch1, num_heads=num_heads, depth=transformer_depths[0], dropout=dropout)
        self.spatialTransformer2 = SpatialTransformerSelfAttn(mid_ch2, num_heads=num_heads, depth=transformer_depths[1], dropout=dropout)
        self.head = nn.Conv2d(mid_ch2, out_ch, kernel_size=1, stride=1, padding=0)
        self._init_weights()

    def _init_weights(self):
        self.head.weight.data.zero_()

    def forward(self, x):
        b, *_ = x.shape
        x = self.conv_in(x)
        x = self.pos_enc(x)
        x = self.spatialTransformer1(x)
        x = self.down1(x)
        x = self.spatialTransformer2(x)
        x = self.head(x)
        x = torch.reshape(x, (b, self.cond_emb_size, -1))
        x = torch.permute(x, (0, 2, 1))
        return x

class GateLenHead(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=None, dropout=0.1):
        super().__init__()
        if hidden is None:
            hidden = in_dim // 2
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes)
        )
    def forward(self, z_seq):
        z = z_seq.mean(dim=1)
        return self.fc(z)

class GateLenPredictor(nn.Module):
    def __init__(self, encoder: nn.Module, cond_emb_size: int, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = GateLenHead(cond_emb_size, n_classes)

    def forward(self, U_tensor): # 입력이 텐서여야 함
        z_seq = self.encoder(U_tensor)
        logits = self.head(z_seq)
        return logits


# --- 1. 파라미터 설정 ---
parser = argparse.ArgumentParser(description="GFlowNet for Unitary Decomposition with Heuristic Reward")
# ... (인자 파싱 부분은 이전과 동일, 생략) ...
parser.add_argument("--n_qubits", type=int, default=3, help="Number of qubits in the circuit.")
parser.add_argument("--max_seq_len", type=int, default=10, help="Maximum length of the gate sequence.")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
parser.add_argument("--num_training_steps", type=int, default=10000, help="Total number of training steps.")
parser.add_argument("--batch_size", type=int, default=64, help="Number of trajectories per training batch.")
parser.add_argument("--reward_beta", type=float, default=5.0, help="Temperature for terminal reward scaling R = beta * similarity.")
parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (cuda or cpu).")
parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
parser.add_argument("--use_heuristic_reward", action='store_true', help="Enable reward shaping using the heuristic model.")
parser.add_argument("--heuristic_model_path", type=str, default="gatelen_predictor.pt", help="Path to the pre-trained gate length predictor model.")
parser.add_argument("--heuristic_reward_scale", type=float, default=1.0, help="Scaling factor for the heuristic step rewards.")


# ==============================================================================
# ✨✨✨ 새로운 게이트 정의 및 유틸리티 함수 ✨✨✨
# ==============================================================================
ALPHABET = {
    'h': lambda qc, q: qc.h(q),
    'x': lambda qc, q: qc.x(q),
    'z': lambda qc, q: qc.z(q),
    'cx': lambda qc, q, t: qc.cx(q, t),
    'ccx': lambda qc, q, t, u: qc.ccx(q, t, u),
}

def construct_action_list(n_qubits: int):
    """오류가 수정된 액션 리스트 생성 함수"""
    actions = []
    # 단일 큐빗 게이트
    for q in range(n_qubits):
        actions.extend([('h', q), ('x', q), ('z', q)])
    
    # 2-큐빗 게이트
    for q in range(n_qubits):
        for t in range(n_qubits):
            if q != t:
                actions.append(('cx', q, t))
    
    # 3-큐빗 게이트 (n_qubits가 3 이상일 때만)
    if n_qubits >= 3:
        for q_start in range(n_qubits):
            q1, q2, q3 = q_start, (q_start + 1) % n_qubits, (q_start + 2) % n_qubits
            # 세 큐빗이 모두 다른지 확인
            if len(set([q1, q2, q3])) == 3:
                actions.append(('ccx', q1, q2, q3)) # (control1, control2, target)
                
    return actions

def action_to_unitary(action: tuple, n_qubits: int) -> np.ndarray:
    """액션 튜플을 유니터리 행렬로 변환하는 헬퍼 함수"""
    gate_label = action[0]
    qubit_indices = action[1:]
    
    qc = QuantumCircuit(n_qubits)
    gate_func = ALPHABET[gate_label]
    gate_func(qc, *qubit_indices)
    
    return Operator(qc).data
# ==============================================================================

def unitary_to_tensor(unitary: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.tensor(np.stack([unitary.real, unitary.imag]), dtype=torch.float32)
    return tensor.to(device)

def calculate_similarity(U, V):
    trace_val = np.trace(U.conj().T @ V)
    dim = U.shape[0]
    return (np.abs(trace_val) / dim)**2

# --- 3. 휴리스틱 보상 모델 래퍼 (이전과 동일, 생략) ---
# ... (HeuristicRewardModel 클래스는 여기에 그대로 존재합니다) ...
class HeuristicRewardModel:
    def __init__(self, model_path: str, device: torch.device):
        self.device = device
        cond_emb_size = 256
        max_len = 12 
        encoder = Unitary_encoder(cond_emb_size=cond_emb_size)
        self.model = GateLenPredictor(
            encoder, 
            cond_emb_size=cond_emb_size, 
            n_classes=max_len + 1
        ).to(device)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
            print(f"Successfully loaded heuristic model from {model_path}")
        except FileNotFoundError:
            print(f"Error: Heuristic model file not found at {model_path}")
            print("Disabling heuristic reward.")
            self.model = None
        except Exception as e:
            print(f"Error loading heuristic model: {e}")
            print("Disabling heuristic reward.")
            self.model = None

    @torch.no_grad()
    def predict_gate_length(self, unitary: np.ndarray) -> int:
        if self.model is None:
            return 0
        u_tensor = unitary_to_tensor(unitary, self.device).unsqueeze(0)
        logits = self.model(u_tensor)
        pred_len = logits.argmax(dim=1).item()
        return pred_len

# --- 4. GFlowNet 모델 (이전과 동일, 생략) ---
# ... (UnitaryCNN 클래스는 여기에 그대로 존재합니다) ...
class UnitaryCNN(nn.Module):
    def __init__(self, n_qubits: int, num_gates: int):
        super().__init__()
        dim = 2**n_qubits
        self.conv_stack = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.fc_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * dim * dim, 256), nn.ReLU(),
            nn.Linear(256, num_gates)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_stack(x)
        logits = self.fc_stack(x)
        return logits

# --- 5. 메인 학습 로직 ---
def main(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- ✨ 새로운 방식으로 액션 및 유니터리 행렬 생성 ---
    actions = construct_action_list(args.n_qubits)
    action_unitaries = [action_to_unitary(act, args.n_qubits) for act in actions]
    action_inverses = [u.conj().T for u in action_unitaries]
    num_actions = len(actions)
    
    identity_matrix = np.eye(2**args.n_qubits)
    print(f"Constructed {num_actions} actions for {args.n_qubits} qubits.")

    # --- ✨ GFlowNet 모델의 출력 차원을 `num_actions`에 맞게 수정 ---
    model = UnitaryCNN(n_qubits=args.n_qubits, num_gates=num_actions).to(device)
    
    # --- (이하 로직은 이전과 거의 동일) ---
    heuristic_model = None
    if args.use_heuristic_reward:
        heuristic_model = HeuristicRewardModel(args.heuristic_model_path, device)
        if heuristic_model.model is None:
            args.use_heuristic_reward = False

    log_Z = nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.Adam(list(model.parameters()) + [log_Z], lr=args.learning_rate)
    
    print("Starting training...")
    for step in range(args.num_training_steps):
        batch_loss = 0.0
        
        for _ in range(args.batch_size):
            random_seed = rng.integers(np.iinfo(np.int32).max)
            target_unitary = random_unitary(2**args.n_qubits, seed=random_seed).data
            current_unitary = target_unitary
            
            log_forward_prob = 0.0
            trajectory_heuristic_reward = 0.0

            for _ in range(args.max_seq_len):
                h_old = 0
                if args.use_heuristic_reward:
                    h_old = heuristic_model.predict_gate_length(current_unitary)

                state_tensor = unitary_to_tensor(current_unitary, device).unsqueeze(0)
                action_logits = model(state_tensor).squeeze(0)
                action_dist = Categorical(logits=action_logits)
                action_idx = action_dist.sample()
                log_forward_prob += action_dist.log_prob(action_idx)
                
                # ✨ action_inverses 리스트에서 선택
                gate_inv = action_inverses[action_idx.item()]
                current_unitary = gate_inv @ current_unitary
                
                h_new = 0
                if args.use_heuristic_reward:
                    h_new = heuristic_model.predict_gate_length(current_unitary)
                    step_reward = h_old - h_new
                    trajectory_heuristic_reward += step_reward * args.heuristic_reward_scale
                
                similarity_to_I = calculate_similarity(current_unitary, identity_matrix)
                if similarity_to_I > 0.999:
                    break
            
            terminal_reward = args.reward_beta * (similarity_to_I - 1.0)
            total_reward = terminal_reward + trajectory_heuristic_reward
            
            log_backward_prob = 0.0
            traj_loss = (log_Z + log_forward_prob - log_backward_prob - total_reward)**2
            batch_loss += traj_loss

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