import torch
import torch.multiprocessing as mp

import socket
from typing import Dict, List, Tuple
import math
import inspect
from datetime import datetime

import numpy as np
from torch import Tensor
from gflownet import GFNTask, LogScalar, ObjectProperties
from gflownet.config import Config, init_empty
from gflownet.envs.circuit_building_env import AutoregressiveCircuitBuildingContext, CircuitBuildingEnv
from gflownet.models.circuit_transformer import CircuitTransformerGFN
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.transforms import to_logreward
from gflownet.utils.circuit import sequence_to_matrices, total_matrix

toffoli = torch.tensor([
    [1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,1,0],
], dtype=torch.complex128)

def random_unitary(
        num_qubits: int,
        dtype: torch.dtype = torch.complex128,
        device: torch.device | str = "cpu",
) -> torch.Tensor:
    d = 2 ** num_qubits
    real = torch.randn(d, d, dtype=dtype, device=device).real
    imag = torch.randn(d, d, dtype=dtype, device=device).real
    Z = torch.complex(real, imag)
    Q, R = torch.linalg.qr(Z)
    diag_R = torch.diagonal(R)
    phase = diag_R / torch.abs(diag_R)
    Q = Q * phase.unsqueeze(0)
    I = torch.eye(d, dtype=dtype, device=device)
    #return total_matrix(sequence_to_matrices("A"))
    return toffoli

def string_to_gate():
    pass

def calculate_fidelity(circuit_str: str, target: torch.Tensor) -> float:
    """
    circuit_str: e.g. "Qubit 0: H X Z\nQubit 1: Y Z"
    target: [d,d] tensor (d=2**num_qubits) already on correct device

    returns fidelity = |Tr(U^† target)| / d
    """
    #print("Input circuit string:", circuit_str)
    if circuit_str.count('P') > 2 or circuit_str.count('Q') > 2 or circuit_str.count('R') > 2:
        return 0
    for i in range(1, len(circuit_str)):
        if circuit_str[i] == circuit_str[i-1]:
            return 0
    circuit_mat = total_matrix(sequence_to_matrices(circuit_str))
    #print("circuit_mat shape:", circuit_mat.shape)
    #print("target shape:", target.shape)
    #print("circuit_mat device:", circuit_mat.device)
    #print("target device:", target.device)
    # Ensure circuit_mat is on the same device as target
    circuit_mat = circuit_mat.to(target.device)
    d = target.shape[0]  # dimension of the matrix
    trace_val = (circuit_mat.mH @ target).trace()
    fidelity = abs(trace_val) / d
    return fidelity if fidelity > 1e-6 else 0

class ToyCircuitTask(GFNTask):
    def __init__(
        self,
        matrix: torch.Tensor,
        num_qubits: int,
        cfg: Config,
    ) -> None:
        super().__init__()
        # Move target unitary to the correct device
        self.device = torch.device(cfg.device)
        self.matrix = matrix
        self.num_qubits = num_qubits

        self.temperature_conditional = TemperatureConditional(cfg)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], obj_props: ObjectProperties) -> LogScalar:
        return LogScalar(self.temperature_conditional.transform(cond_info, to_logreward(obj_props)))

    def compute_obj_properties(self, objs: List[str]) -> Tuple[ObjectProperties, Tensor]:
        rs = torch.tensor([calculate_fidelity(c, self.matrix) for c in objs]).float()
        return ObjectProperties(rs[:, None]), torch.ones(len(objs), dtype=torch.bool)



class ToyCircuitTrainer(StandardOnlineTrainer):
    task: ToyCircuitTask

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = False
        cfg.num_workers = 8
        cfg.num_validation_gen_steps = 1
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20_000
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10
        cfg.algo.num_from_policy = 64
        cfg.model.num_emb = 64
        cfg.model.num_layers = 4

        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 50
        cfg.algo.max_len = 50
        cfg.algo.sampling_tau = 0.9
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-2
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = False

    def setup_model(self):
        self.model = CircuitTransformerGFN(self.ctx, self.cfg)

    def setup_task(self):
        matrix = random_unitary(
            num_qubits=self.cfg.task.toy_circuit.num_qubits,
            device=self.device
        )
        print("Target: ", matrix)
        self.task = ToyCircuitTask(
            matrix=matrix,
            num_qubits=self.cfg.task.toy_circuit.num_qubits,
            cfg=self.cfg,
        )

    def setup_env_context(self):
        self.env = CircuitBuildingEnv(num_qubits=self.cfg.task.toy_circuit.num_qubits)
        self.ctx = AutoregressiveCircuitBuildingContext(
            gates=self.cfg.task.toy_circuit.gates,
            num_qubits=self.cfg.task.toy_circuit.num_qubits,
            num_cond_dim=self.task.num_cond_dim,
        )

    def setup_algo(self):
        super().setup_algo()
        self.algo.model_is_autoregressive = True


def main():
    config = init_empty(Config())
    config.log_dir = f"./logs/debug_run_toy_circuit_{datetime.now().strftime('%Y%m%d_%H%M')}"
    config.device = "cuda"
    config.overwrite_existing_exp = True
    config.num_training_steps = 100000
    config.checkpoint_every = 200
    config.num_workers = 0
    config.task.toy_circuit.num_qubits = 3
    # Reduce number of gates to avoid token index issues
    #config.task.toy_circuit.gates = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R"]  # Basic quantum gates
    config.task.toy_circuit.gates = "ABCDEFGHIJKLMNOPQR"
    config.print_every = 1
    config.cond.temperature.sample_dist = "constant"
    config.cond.temperature.dist_params = [2.0]
    config.cond.temperature.num_thermometer_dim = 1
    config.algo.train_random_action_prob = 0.05
    
    trial = ToyCircuitTrainer(config)
    trial.run()

if __name__ == "__main__":
    main()