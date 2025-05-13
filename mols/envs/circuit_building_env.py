from copy import deepcopy
from typing import Any, List, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data

from gflownet.envs.graph_building_env import (
    ActionIndex,
    Graph,
    GraphAction,
    GraphActionType,
    GraphBuildingEnv,
    GraphBuildingEnvContext,
)

gate_1 = ["A", "B", "C", "D", "E", "R", "P"]
gate_2 = ["F", "G", "H", "I", "J", "P", "Q"]
gate_3 = ["K", "L", "M", "N", "O", "Q", "R"]

class QuantumCircuit(Graph):
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        # Each qubit is represented as a sequence of gates
        self.circuit: List[Any] = []

    def __repr__(self):
        return "".join(map(str, self.seq))
    @property
    def nodes(self):
        return self.circuit

class CircuitBuildingEnv(GraphBuildingEnv):
    """This class extends GraphBuildingEnv to generate quantum circuits as sequences of gates on multiple qubits."""

    def __init__(self, num_qubits: int):
        super().__init__()
        self.num_qubits = num_qubits

    def new(self):
        return QuantumCircuit(self.num_qubits)

    def step(self, g: Graph, a: GraphAction):
        c: QuantumCircuit = deepcopy(g)  # type: ignore
        if a.action == GraphActionType.AddNode:
            # The value contains both the gate type and the qubit index
            c.circuit.append(a.value)
        return c

    def count_backward_transitions(self, g: Graph, check_idempotent: bool = False):
        return 1

    def parents(self, g: Graph):
        c: QuantumCircuit = deepcopy(g)  # type: ignore
        n = len(c.circuit)
        if n == 0:
            return []
        parents = []
        i = 1
        first_gate = c.circuit[-1]
        parents.append((GraphAction(GraphActionType.AddNode, value=first_gate), c.circuit[:-1]))
        while i<n:
            second_gate = c.circuit[-i-1]
            if first_gate in gate_1 and second_gate in gate_1:
                i += 1
                continue
            elif first_gate in gate_2 and second_gate in gate_2:
                i += 1
                continue
            elif first_gate in gate_3 and second_gate in gate_3:
                i += 1
                continue
            else:
                parents.append((GraphAction(GraphActionType.AddNode, value=second_gate), c.circuit[:-i-1] + c.circuit[-i:]))
                break
        j = i + 1
        while j<n:
            third_gate = c.circuit[-j]
            if third_gate in gate_1 and (first_gate in gate_1 or second_gate in gate_1):
                j += 1
                continue
            elif third_gate in gate_2 and (first_gate in gate_2 or second_gate in gate_2):
                j += 1
                continue
            elif third_gate in gate_3 and (first_gate in gate_3 or second_gate in gate_3):
                j += 1
                continue
            else:
                parents.append((GraphAction(GraphActionType.AddNode, value=third_gate), c.circuit[:-j] + c.circuit[-j+1:]))
                break
        return parents


    def reverse(self, g: Graph, ga: GraphAction):
        return GraphAction(GraphActionType.Stop)

class CircuitBatch:
    def __init__(self, circuits: List[torch.Tensor], pad: int):
        self.circuits = circuits
        self.x = pad_sequence(circuits, batch_first=False, padding_value=pad)
        self.mask = self.x.eq(pad).mT
        self.lens = torch.tensor([len(i) for i in circuits], dtype=torch.long)
        self.logit_idx = self.x.ne(pad).T.flatten().nonzero().flatten()
        self.num_graphs = self.lens.sum().item()

    def to(self, device):
        for name in dir(self):
            x = getattr(self, name)
            if isinstance(x, torch.Tensor):
                setattr(self, name, x.to(device))
        return self

class AutoregressiveCircuitBuildingContext(GraphBuildingEnvContext):
    """This context generates quantum circuits by adding gates to qubits in an autoregressive fashion."""

    def __init__(self, gates: Sequence[str], num_qubits: int, num_cond_dim=0):
        self.gates = gates #gates as alphabet
        self.num_qubits = num_qubits
        self.action_type_order = [GraphActionType.Stop, GraphActionType.AddNode]

        # Each gate can be applied to any qubit
        self.num_tokens = len(gates) + 2  # Gates + BOS + PAD
        self.bos_token = len(gates)
        self.pad_token = len(gates) + 1
        self.num_actions = len(gates) +1 # (Gates × Qubits) + Stop
        self.num_cond_dim = num_cond_dim

    def ActionIndex_to_GraphAction(self, g: Data, aidx: ActionIndex, fwd: bool = True) -> GraphAction:
        t = self.action_type_order[aidx.action_type]
        if t is GraphActionType.Stop:
            return GraphAction(t)
        elif t is GraphActionType.AddNode:
           return GraphAction(t, value=aidx.col_idx)
        raise ValueError(aidx)

    def GraphAction_to_ActionIndex(self, g: Data, action: GraphAction) -> ActionIndex:
        if action.action is GraphActionType.Stop:
            col = 0
            type_idx = self.action_type_order.index(action.action)
        elif action.action is GraphActionType.AddNode:
            col = action.value
            type_idx = self.action_type_order.index(action.action)
        else:
            raise ValueError(action)
        return ActionIndex(action_type=type_idx, row_idx=0, col_idx=int(col))

    def graph_to_Data(self, g: Graph):
        c: QuantumCircuit = g  # type: ignore
        return torch.tensor([self.bos_token] + c.circuit, dtype=torch.long)

    def collate(self, graphs: List[Data]):
        return CircuitBatch(graphs, pad=self.pad_token)

    def is_sane(self, g: Graph) -> bool:
        return True

    def graph_to_obj(self, g: Graph):
        c: QuantumCircuit = g  # type: ignore
        return "".join(self.gates[int(i)] for i in c.circuit)

    def object_to_log_repr(self, g: Graph):
        return self.graph_to_obj(g)