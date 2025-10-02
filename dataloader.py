# dataloader.py
import argparse
import os
import random
import numpy as np
from collections import Counter
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator

basis_gates = ['h', 's', 'sdg', 't', 'tdg', 'z', 'cx']
# Mapping from gate labels to functions that apply them on a QuantumCircuit
ALPHABET = {
    'h': lambda qc, q: qc.h(q),  # Hadamard on qubit q
    's': lambda qc, q: qc.s(q),  # s on qubit q
    'sdg': lambda qc, q: qc.sdg(q),  # sdg on qubit q
    't': lambda qc, q: qc.t(q),  # t on qubit q
    'tdg': lambda qc, q: qc.tdg(q),  # tdg on qubit q
    'z': lambda qc, q: qc.z(q),  # z on qubit q
    'cx': lambda qc, q, t: qc.cx(q, t),  # CNOT: control q -> target t
}

def construct_action_list(n_qubits: int):
    """
    Build a list of available actions as (gate_label, qubit_index) pairs.

    Supported gates:
      - 'h', 's', 'z', 't', 'cx': single-qubit gates on each qubit
      - 'cx': controlled-NOT (CX) from qubit to qubit

    Returns:
        List of tuples [(gate_label, qubit_index), ...]
    """
    actions = []
    for q in range(n_qubits):
        actions.extend([
            ('h', q),
            ('s', q),
            ('sdg', q),
            ('t', q),
            ('tdg', q),
            ('z', q),
        ])
    for q in range(n_qubits):
        for t in range(n_qubits):
            if q != t:
                actions.extend([
                    ('cx', q, t),
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
        if gate == 'cx':
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
        if action[0] == 'cx':
            mapping[(action[0], action[1], action[2])] = idx
        else:
            mapping[(action[0], action[1])] = idx
    
    new_seq: list[int] = []
    for instr, qargs, _cargs in transpiled_qc.data:
        if instr.name == 'cx':
            key = (instr.name, qargs[0]._index, qargs[1]._index)
        else:
            key = (instr.name, qargs[0]._index)
        if key not in mapping:
            raise ValueError(
                f"No matching action index for instruction {instr.name} on qubit {key[1]}"
            )
        new_seq.append(mapping[key])
    
    return new_seq
   

def sequence_to_unitary(action_seq, action_list, n_qubits: int):
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
    for idx in action_seq:
        if idx >= len(action_list):
            break
        #print("current idx: ", idx)
        gate = action_list[idx][0]
        q = action_list[idx][1]
        if gate == 'cx':
            ALPHABET[gate](qc, q, action_list[idx][2])
        else:
            ALPHABET[gate](qc, q)
    U = Operator(qc).data
    return U.astype(np.complex64)

def generate_dataset(
    file_path: str,
    num_samples: int,
    max_len: int,
    n_qubits: int
):
    """
    Generate a dataset of random gate sequences and their corresponding unitaries.
    Saves data to a compressed NPZ file with keys:
      - 'seqs': integer sequences [num_samples, max_len], padded with pad_idx
      - 'lengths': actual sequence lengths [num_samples]
      - 'unitaries': unitary matrices [num_samples, 2^n_qubits, 2^n_qubits]
    """
    actions = construct_action_list(n_qubits)
    action2idx = {act: i for i, act in enumerate(actions)}
    pad_idx = len(actions)
    
    # Track length distribution for monitoring
    length_distribution = {}

    # Preallocate arrays
    seqs = np.full((num_samples, max_len), pad_idx, dtype=np.int32)
    lengths = np.zeros((num_samples,), dtype=np.int32)
    unitaries = np.zeros((num_samples, 2**n_qubits, 2**n_qubits), dtype=np.complex64)

    samples_generated = 0
    max_attempts = num_samples * 10  # Prevent infinite loops
    attempts = 0

    while samples_generated < num_samples and attempts < max_attempts:
        attempts += 1
        
        # Sample original sequence length
        L = random.randint(1, max_len)
        
        # Sample random action sequence
        seq = [random.choice(actions) for _ in range(L)]
        
        # Convert actions to indices and transpile
        try:
            transpiled_seq = transpile_action_sequence([action2idx[a] for a in seq], actions, n_qubits)
            transpiled_len = len(transpiled_seq)
            
            # Skip if transpiled sequence is too long (should be rare)
            if transpiled_len > max_len:
                continue
                
            # Skip if we have too many samples of this length (optional balancing)
            if transpiled_len in length_distribution and length_distribution[transpiled_len] > num_samples // max_len:
                continue
                
        except Exception as e:
            # Skip sequences that fail to transpile
            print(f"Warning: Failed to transpile sequence: {e}")
            continue

        # Store the sequence
        seqs[samples_generated, :transpiled_len] = transpiled_seq
        lengths[samples_generated] = transpiled_len
        
        # Update length distribution
        length_distribution[transpiled_len] = length_distribution.get(transpiled_len, 0) + 1
        
        # Always compute the unitary (removed the problematic counter logic)
        try:
            U = sequence_to_unitary(transpiled_seq, actions, n_qubits)
            unitaries[samples_generated] = U
        except Exception as e:
            print(f"Warning: Failed to compute unitary for sequence: {e}")
            # Fill with identity matrix as fallback
            unitaries[samples_generated] = np.eye(2**n_qubits, dtype=np.complex64)

        samples_generated += 1

        if samples_generated % 100_000 == 0:
            print(f"Generated {samples_generated}/{num_samples} samples")
            print(f"Length distribution: {dict(sorted(length_distribution.items()))}")

    if samples_generated < num_samples:
        print(f"Warning: Only generated {samples_generated}/{num_samples} samples after {attempts} attempts")
        # Truncate arrays to actual number of samples
        seqs = seqs[:samples_generated]
        lengths = lengths[:samples_generated]
        unitaries = unitaries[:samples_generated]

    # Save the dataset to a compressed NPZ file
    np.savez_compressed(file_path, seqs=seqs, lengths=lengths, unitaries=unitaries)
    print(f"Dataset saved to {file_path}")
    print(f"Final length distribution: {dict(sorted(length_distribution.items()))}")


def generate_balanced_dataset(
    file_path: str,
    num_samples: int,
    max_len: int,
    n_qubits: int
):
    """
    Generate a balanced dataset with equal representation of each sequence length.
    This function ensures that each length from 1 to max_len gets approximately
    the same number of samples.
    """
    actions = construct_action_list(n_qubits)
    action2idx = {act: i for i, act in enumerate(actions)}
    pad_idx = len(actions)
    
    # Calculate samples per length
    samples_per_length = num_samples // max_len
    remaining_samples = num_samples % max_len
    
    # Preallocate arrays
    seqs = np.full((num_samples, max_len), pad_idx, dtype=np.int32)
    lengths = np.zeros((num_samples,), dtype=np.int32)
    unitaries = np.zeros((num_samples, 2**n_qubits, 2**n_qubits), dtype=np.complex64)
    
    sample_idx = 0
    
    for target_length in range(1, max_len + 1):
        # Calculate how many samples we need for this length
        current_samples_needed = samples_per_length
        if target_length <= remaining_samples:
            current_samples_needed += 1
            
        samples_for_this_length = 0
        max_attempts = current_samples_needed * 20  # Allow more attempts for rare lengths
        attempts = 0
        
        while samples_for_this_length < current_samples_needed and attempts < max_attempts:
            attempts += 1
            
            # Generate a sequence of the target length
            seq = [random.choice(actions) for _ in range(target_length)]
            
            try:
                transpiled_seq = transpile_action_sequence([action2idx[a] for a in seq], actions, n_qubits)
                transpiled_len = len(transpiled_seq)
                
                # Only accept sequences that transpile to the target length
                if transpiled_len != target_length:
                    continue
                    
            except Exception as e:
                continue
            
            # Store the sequence
            seqs[sample_idx, :transpiled_len] = transpiled_seq
            lengths[sample_idx] = transpiled_len
            
            # Compute unitary
            try:
                U = sequence_to_unitary(transpiled_seq, actions, n_qubits)
                unitaries[sample_idx] = U
            except Exception as e:
                print(f"Warning: Failed to compute unitary: {e}")
                unitaries[sample_idx] = np.eye(2**n_qubits, dtype=np.complex64)
            
            sample_idx += 1
            samples_for_this_length += 1
            
        print(f"Length {target_length}: Generated {samples_for_this_length}/{current_samples_needed} samples")
    
    # Truncate to actual number of samples generated
    if sample_idx < num_samples:
        seqs = seqs[:sample_idx]
        lengths = lengths[:sample_idx]
        unitaries = unitaries[:sample_idx]
        print(f"Warning: Only generated {sample_idx}/{num_samples} samples")
    
    # Save the dataset
    np.savez_compressed(file_path, seqs=seqs, lengths=lengths, unitaries=unitaries)
    print(f"Balanced dataset saved to {file_path}")
    
    # Print final distribution
    length_counts = Counter(lengths)
    print("Final length distribution:")
    for length in sorted(length_counts.keys()):
        print(f"  Length {length}: {length_counts[length]} samples")


class CircuitDataset(Dataset):
    """
    PyTorch Dataset for loading quantum circuit data.
    Each item is a tuple:
      - seq: LongTensor of action indices [max_len]
      - length: int, actual length of the sequence
      - unitary: ComplexTensor [2^n_qubits, 2^n_qubits]
    """
    def __init__(self, data_path: str):
        data = np.load(data_path)
        self.seqs = data['seqs']
        self.lengths = data['lengths']
        self.unitaries = data['unitaries']

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx: int):
        seq = torch.tensor(self.seqs[idx], dtype=torch.long)
        length = int(self.lengths[idx])
        U = torch.tensor(self.unitaries[idx], dtype=torch.complex64)
        return seq, length, U


def get_dataloader(
    data_path: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4
):
    """
    Create and return a DataLoader for the CircuitDataset.
    """
    dataset = CircuitDataset(data_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate or load circuit dataset.")
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the .npz dataset file')
    parser.add_argument('--num_samples', type=int, default=12_000_000,
                        help='Number of circuits to generate')
    parser.add_argument('--max_len', type=int, default=12,
                        help='Maximum gate sequence length')
    parser.add_argument('--n_qubits', type=int, default=3,
                        help='Number of qubits')
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Generating dataset at {args.data_path}")
        generate_balanced_dataset(
            args.data_path,
            args.num_samples,
            args.max_len,
            args.n_qubits
        )
    else:
        print(f"Loading existing dataset at {args.data_path}")
        data = np.load(args.data_path, allow_pickle=True)
        lengths = data['lengths']
        counter = Counter(lengths)
        df = pd.DataFrame(counter.items(), columns=["length", "count"]) 
        df = df.sort_values("length").reset_index(drop=True)
        print(df)

