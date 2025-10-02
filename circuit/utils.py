# utils.py
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
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

basis_gates = ['h', 'x', 'z', 'cx', 'ccx']
# Mapping from gate labels to functions that apply them on a QuantumCircuit
ALPHABET = {
    'h': lambda qc, q: qc.h(q),  # Hadamard on qubit q
    'x': lambda qc, q: qc.x(q),  # Pauli-X on qubit q
    'z': lambda qc, q: qc.z(q),  # Pauli-Z on qubit q
    'cx': lambda qc, q, t: qc.cx(q, t),  # CNOT: control q -> target t
    'ccx': lambda qc, q, t, u: qc.ccx(q, t, u),  # Toffoli: control q,t -> target u
}

def construct_action_list(n_qubits: int):
    """
    Build a list of available actions as (gate_label, qubit_index) pairs.

    Supported gates:
      - 'h', 'x', 'z': single-qubit gates on each qubit
      - 'cx': controlled-NOT (CX) from qubit to qubit
      - 'ccx': Toffoli (CCX) using control qubits to target qubit

    Returns:
        List of tuples [(gate_label, qubit_index), ...]
    """
    actions = []
    for q in range(n_qubits):
        actions.extend([
            ('h', q),
            ('x', q),
            ('z', q),
        ])
    for q in range(n_qubits):
        for t in range(n_qubits):
            if q != t:
                actions.extend([
                    ('cx', q, t),
                ])
    for q in range(n_qubits):
        actions.extend([
            ('ccx', q, (q + 1) % n_qubits, (q + 2) % n_qubits),
        ])
    return actions


def transpile_action_sequence(
    action_seq: list[int],
    action_list: list[tuple],
    num_qubits: int,
    optimization_level: int = 1,
) -> list[int]:
    # 1) action_seq → QuantumCircuit
    qc = QuantumCircuit(num_qubits)
    for act in action_seq:
        gate = action_list[act][0]
        q = action_list[act][1]
        if gate == 'ccx': 
            ALPHABET[gate](qc, q, (q + 1) % num_qubits, (q + 2) % num_qubits)
        elif gate == 'cx':
            ALPHABET[gate](qc, q, action_list[act][2])
        else:
            ALPHABET[gate](qc, q)
    
    # 2) transpile
    transpiled_qc = transpile(
        qc,
        basis_gates=basis_gates,
        optimization_level=optimization_level,
    )
    
    # 3) transpiled QuantumCircuit → new_seq
    #    (gate_name, qubits, params) → index mapping
    mapping: dict[tuple, int] = {}
    for idx, action in enumerate(action_list):
        if action[0] == 'ccx':
            mapping[(action[0], action[1], action[2], action[3])] = idx
        elif action[0] == 'cx':
            mapping[(action[0], action[1], action[2])] = idx
        else:
            mapping[(action[0], action[1])] = idx
    
    new_seq: list[int] = []
    for instr, qargs, _cargs in transpiled_qc.data:
        if instr.name == 'ccx':
            key = (instr.name, qargs[0]._index, qargs[1]._index, qargs[2]._index)
        elif instr.name == 'cx':
            key = (instr.name, qargs[0]._index, qargs[1]._index)
        else:
            key = (instr.name, qargs[0]._index)
        if key not in mapping:
            raise ValueError(
                f"No matching action index for instruction {instr.name} on qubit {key[1]}"
            )
        new_seq.append(mapping[key])
    
    return new_seq
   

def sequence_to_unitary(seq_indices, action_list, n_qubits: int):
    """
    Given a sequence of action indices, construct the corresponding
    QuantumCircuit and return its unitary as a numpy.complex64 array.

    Args:
        seq_indices: 1D list or array of integer indices into action_list
        action_list: list of (gate_label, qubit_index) tuples
        n_qubits: number of qubits in the circuit

    Returns:
        Unitary matrix U of shape (2**n_qubits, 2**n_qubits) as complex64
    """
    qc = QuantumCircuit(n_qubits)
    for idx in seq_indices:
        if idx == len(action_list):
            break
        #print("current idx: ", idx)
        gate = action_list[idx][0]
        q = action_list[idx][1]
        if gate == 'ccx':
            ALPHABET[gate](qc, q, (q + 1) % n_qubits, (q + 2) % n_qubits)
        elif gate == 'cx':
            ALPHABET[gate](qc, q, action_list[idx][2])
        else:
            ALPHABET[gate](qc, q)
    U = Operator(qc).data
    return U.astype(np.complex64)


def unitary_distance(U: np.ndarray, V: np.ndarray) -> float:
    """
    Compute a normalized distance between two unitaries using the Frobenius norm.

    Args:
        U: first unitary matrix as numpy array
        V: second unitary matrix as numpy array

    Returns:
        Frobenius norm of (U - V) divided by Frobenius norm of U
    """
    num = np.linalg.norm(U - V, ord='fro')
    den = np.linalg.norm(U, ord='fro')
    return num / (den + 1e-12)


def log_reward(seq_indices, action_list, target_U: np.ndarray, n_qubits: int = 3, eps: float = 1e-9) -> float:
    """
    Compute the log-fidelity reward for a given action sequence relative to a target unitary.

    Fidelity = |Tr(U_seq^dagger @ target_U)| / 2**n_qubits

    Args:
        seq_indices: iterable of integer indices into action_list
        action_list: list of action tuples matching keys in ALPHABET
        target_U: target unitary matrix as a numpy array
        n_qubits: number of qubits in the circuit
        eps: small constant to avoid log(0)

    Returns:
        Fidelity reward as a float
    """
    U_seq = sequence_to_unitary(seq_indices, action_list, n_qubits)
    fid = abs(np.trace(U_seq.conj().T @ target_U)) / (2**n_qubits)
    return fid

def log_length_reward(seq_indices, action_list, n_qubits: int = 3, eps: float = 1e-9) -> float:
    """
    Compute the log-length reward for a given action sequence.
    """
    return -np.log(len(seq_indices) + eps)

def process_logits(all_logits, pos_mask, args):
    # Model predicts positional logits p_i and word logits for each position w_ij.
    # The logits used to sample pairs of positions and word (i, j) are computed as p_i + w_ij.
    pos_logits = all_logits[0, :, -(args.max_length + 1) :]  # [batch_size, n/k + 1]
    pos_logits[pos_mask] = -torch.inf
    word_logits = all_logits[:, :, : args.num_actions]  # [n/k + 1, batch_size, 2^k]
    sum_logits = torch.moveaxis(word_logits, 1, 0) + pos_logits[:, :, None]  # [batch_size, n/k + 1, 2^k]
    sum_logits = sum_logits.reshape(
        pos_logits.shape[0], (args.max_length + 1) * (args.num_actions)
    )  # [batch_size, (n/k + 1) * 2^k]
    return pos_logits, word_logits, sum_logits

def batch_log_rewards(batch, unitaries, action_list, n_qubits):
    batch_np = batch.cpu().numpy()
    log_rewards = [log_reward(batch_np[i], action_list, unitaries[i], n_qubits) for i in range(batch_np.shape[0])]
    return torch.tensor(log_rewards)

def batch_log_length_rewards(batch, action_list, n_qubits):
    batch_np = batch.cpu().numpy()
    log_rewards = [log_length_reward(batch_np[i], action_list, n_qubits) for i in range(batch_np.shape[0])]
    return torch.tensor(log_rewards)

def reward(s, action_list, U, n_qubits):
    return np.exp(log_reward(s, action_list, U, n_qubits))

def batch_rewards(batch, unitaries, action_list, n_qubits):
    batch_np = batch.cpu().numpy()
    rewards = [reward(batch_np[i], action_list, unitaries[i], n_qubits) for i in range(batch_np.shape[0])]
    return torch.tensor(rewards)

def batch_length_rewards(batch, action_list, n_qubits):
    batch_np = batch.cpu().numpy()
    rewards = [log_length_reward(batch_np[i], action_list, n_qubits) for i in range(batch_np.shape[0])]
    return torch.tensor(rewards)

def compute_correlation(model, U, action_list, num_qubits, test_set, args, rounds=10, batch_size=180):
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
                [[args.num_actions + 1] + ([args.num_actions] * (args.max_length)) for i in range(batch_size)]
            ).to(args.device)
            for i in range(args.max_length):
                with torch.no_grad():
                    pos_mask = batch != args.num_actions
                    all_logits, _ = model(batch.T)
                    pos_logits, word_logits, sum_logits = process_logits(all_logits, pos_mask, args)

                    # There is a bug in pytorch that allows to sample objects that has 0 probability (happens very rarely but still happens).
                    # This loop basically resamples until everything is correct.
                    while True:
                        uniform_probs = torch.zeros(batch_size, args.max_length + 1) + 1 / (args.max_length - i)
                        uniform_probs[pos_mask] = 0.0
                        positions = Categorical(probs=uniform_probs).sample().to(args.device)
                        if (batch[range(batch_size), positions] == args.num_actions).sum() == batch_size:
                            break

                    assert positions.min() >= 1
                    assert positions.max() <= args.max_length

                    start = batch_idx * batch_size
                    end   = start + batch_size
                    seq_batch = test_set[start:end] 

                    steps = positions - 1    

                    words = seq_batch[torch.arange(batch_size), steps]

                    batch[torch.arange(batch_size), positions] = words

                    actions = positions * args.num_actions + words

                    log_pf = sum_logits[range(batch_size), actions] / args.entropy_coeff - torch.logsumexp(
                        sum_logits / args.entropy_coeff, dim=-1
                    )
                    p_forward_sums[batch_idx * batch_size : (batch_idx + 1) * batch_size, round] += log_pf

    p_forward_sum = torch.logsumexp(p_forward_sums, dim=-1)
    log_rewards = np.array([log_reward(s, action_list, U, num_qubits) for s in test_set])
    return spearmanr((args.reward_exponent * log_rewards), (p_forward_sum.detach().cpu().numpy())).statistic

def compute_correlation_wpb(model, U, action_list, num_qubits, test_set, args, rounds=10, batch_size=180):
    model.eval()
    assert len(test_set) % batch_size == 0
    logP_sums = torch.zeros(len(test_set), rounds).to(args.device)

    for round in range(rounds):
        for batch_idx in range(len(test_set) // batch_size):
            batch = []
            for j in range(batch_size):
                # test_set[j] is [a0, a1, ..., a_{T-1}]
                seq_idxs = test_set[batch_idx * batch_size + j]
                current = [args.num_actions + 1] + list(seq_idxs) #be careful
                batch.append(current)
            batch = torch.tensor(batch, device=args.device)

            for i in range(args.max_length):
                with torch.no_grad():
                    _, pb_logits = model(batch.T)
                    pb_logits[batch >= (args.num_actions)] = -torch.inf

                    while True:
                        positions = Categorical(logits=pb_logits).sample().to(args.device)
                        if (batch[range(batch_size), positions] != args.num_actions).sum() == batch_size:
                            break
                    logPb = Categorical(logits=pb_logits).log_prob(positions)

                    assert positions.min() >= 1
                    assert positions.max() <= args.max_length

                    actions = positions * (args.num_actions) + batch[range(batch_size), positions]
                    batch[range(batch_size), positions] = args.num_actions

                    pos_mask = batch != args.num_actions
                    forward_logits, _ = model(batch.T)
                    _, _, forward_logits_sum = process_logits(forward_logits, pos_mask, args)
                    logPf = Categorical(logits=forward_logits_sum).log_prob(actions)

                    logP_sums[batch_idx * batch_size : (batch_idx + 1) * batch_size, round] += logPf - logPb

    logP_sum = torch.logsumexp(logP_sums, dim=-1)
    log_rewards = np.array([log_reward(s, action_list, U, num_qubits) for s in test_set])
    return spearmanr((args.reward_exponent * log_rewards), (logP_sum.detach().cpu().numpy())).statistic
