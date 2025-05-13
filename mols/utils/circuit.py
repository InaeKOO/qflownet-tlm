import torch
import math
import cmath
import string

# 1-큐빗 기본 게이트 정의
H = (1/math.sqrt(2)) * torch.tensor([[1, 1],
                                     [1, -1]], dtype=torch.complex128)
X = torch.tensor([[0, 1],
                  [1, 0]], dtype=torch.complex128)
Y = torch.tensor([[0, -1j],
                  [1j, 0]], dtype=torch.complex128)
Z = torch.tensor([[1, 0],
                  [0, -1]], dtype=torch.complex128)
T = torch.tensor([[1, 0],
                  [0, cmath.exp(1j * math.pi / 4)]], dtype=torch.complex128)
I = torch.eye(2, dtype=torch.complex128)

# 3-큐빗 행렬로 확장하기 (tensor product 순서: qubit0 ⊗ qubit1 ⊗ qubit2)
def embed_single(gate: torch.Tensor, qubit: int) -> torch.Tensor:
    mats = []
    for q in range(3):
        mats.append(gate if q == qubit else I)
    return torch.kron(torch.kron(mats[0], mats[1]), mats[2])

# CNOT 을 3-큐빗 전체 공간으로 확장
def cnot(control: int, target: int) -> torch.Tensor:
    n = 3
    dim = 2 ** n
    M = torch.zeros((dim, dim), dtype=torch.complex128)
    for i in range(dim):
        # bit 위치: qubit k 에 대응되는 비트는 (n-1-k) 번째
        if ((i >> (n - 1 - control)) & 1) == 1:
            j = i ^ (1 << (n - 1 - target))
        else:
            j = i
        M[j, i] = 1
    return M

# 알파벳 A…U 에 매핑
letters = list(string.ascii_uppercase)[:18]  # A through U
mapping: dict[str, torch.Tensor] = {}
idx = 0

# single-qubit (5 gates × 3 위치 = 15)
for G in (H, X, Y, Z, T):
    for q in range(3):
        mapping[letters[idx]] = embed_single(G, q)
        idx += 1

# two-qubit (CNOT, CZ) × 3 인접 쌍 = 6
for pair in ((0, 1), (1, 2), (2, 0)):
    mapping[letters[idx]] = cnot(*pair)
    idx += 1

# 문자열 → 행렬 리스트 변환 함수
def sequence_to_matrices(seq: list[any]) -> list[torch.Tensor]:
    """
    seq: e.g. "ABJUC"
    returns: [mapping['A'], mapping['B'], ...] 
    """
    return [mapping[s] for s in seq]

def remove_small_imag(x: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
    """
    복소텐서 x의 허수부 중 |im| < tol 인 요소는 0으로 만드는 함수.
    real 부분은 그대로 유지합니다.
    """
    re = x.real
    im = x.imag
    # 허수부의 작은 값들만 0으로
    re_clean = torch.where(re.abs() < tol, torch.zeros_like(re), re)
    im_clean = torch.where(im.abs() < tol, torch.zeros_like(im), im)
    return torch.complex(re_clean, im_clean)

def total_matrix(matrices: list[torch.Tensor]) -> torch.Tensor:
    """
    seq: e.g. "ABJUC"
    returns: [mapping['A'], mapping['B'], ...] 
    """
    mat = torch.eye(2**3, dtype=torch.complex128)
    for m in matrices:
        mat = mat @ m
    return remove_small_imag(mat)

# 예시
if __name__ == "__main__":
    seq = "ABCDEFGHIJKLMNOPQR"
    mats = sequence_to_matrices(seq)
    for letter, M in zip(seq, mats):
        #print(f"{letter}: matrix shape = {M.mH@M}")
        pass
    total = total_matrix(mats)
    print(total.mH @ total)
