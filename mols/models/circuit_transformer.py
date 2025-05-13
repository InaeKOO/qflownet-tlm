import math
from typing import Optional

import torch
import torch.nn as nn

from gflownet.config import Config
from gflownet.envs.graph_building_env import GraphActionCategorical, GraphBuildingEnvContext
from gflownet.envs.circuit_building_env import CircuitBatch
from gflownet.models.config import SeqPosEnc


class MLPWithDropout(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers, dropout_prob, init_drop=False):
        super(MLPWithDropout, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        layers = [nn.Linear(in_dim, hidden_layers[0]), nn.ReLU()]
        layers += [nn.Dropout(dropout_prob)] if init_drop else []
        for i in range(1, len(hidden_layers)):
            layers.extend([nn.Linear(hidden_layers[i - 1], hidden_layers[i]), nn.ReLU(), nn.Dropout(dropout_prob)])
        layers.append(nn.Linear(hidden_layers[-1], out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class CircuitTransformerGFN(nn.Module):
    """A transformer-encoder based GFN model for quantum circuits with multiple qubit sequences."""

    ctx: GraphBuildingEnvContext

    def __init__(
        self,
        env_ctx,
        cfg: Config,
        num_state_out=1,
    ):
        super().__init__()
        self.ctx = env_ctx
        self.num_state_out = num_state_out
        num_hid = cfg.model.num_emb
        num_outs = env_ctx.num_actions + num_state_out
        mc = cfg.model
        
        # Positional encoding for sequence positions
        if mc.seq_transformer.posenc == SeqPosEnc.Pos:
            self.pos = PositionalEncoding(num_hid, dropout=cfg.model.dropout, max_len=cfg.algo.max_len + 2)
        elif mc.seq_transformer.posenc == SeqPosEnc.Rotary:
            self.pos = RotaryEmbedding(num_hid)
            
        self.use_cond = env_ctx.num_cond_dim > 0
        self.embedding = nn.Embedding(env_ctx.num_tokens, num_hid)
        
        # Transformer encoder for processing sequences
        encoder_layers = nn.TransformerEncoderLayer(num_hid, mc.seq_transformer.num_heads, num_hid, dropout=mc.dropout)
        self.encoder = nn.TransformerEncoder(encoder_layers, mc.num_layers)
        
        self._logZ = nn.Linear(env_ctx.num_cond_dim, 1)
        
        if self.use_cond:
            self.output = MLPWithDropout(num_hid + num_hid, num_outs, [4 * num_hid, 4 * num_hid], mc.dropout)
            self.cond_embed = nn.Linear(env_ctx.num_cond_dim, num_hid)
        else:
            self.output = MLPWithDropout(num_hid, num_outs, [2 * num_hid, 2 * num_hid], mc.dropout)
            
        self.num_hid = num_hid

    def logZ(self, cond_info: Optional[torch.Tensor]):
        if cond_info is None:
            return self._logZ(torch.ones((1, 1), device=self._logZ.weight.device))
        return self._logZ(cond_info)

    def forward(self, xs: CircuitBatch, cond, batched=False):
        """Process a batch of quantum circuits, where each circuit contains multiple qubit sequences.
        
        Args:
            xs: CircuitBatch containing multiple qubit sequences
            cond: Conditional information
            batched: Whether to process in batched mode
            
        Returns:
            GraphActionCategorical for actions and state predictions
        """
        #print("self.pos class:", self.pos.__class__)  
        #print("module path:", self.pos.__class__.__module__)
        #print("xs.x: ", xs.x.size())
        #print("xs type: ", type(xs))
        x = self.embedding(xs.x)
        x = self.pos(x)  # (time, batch, nemb)
        #print("x.size: ",x.size())
        #print("xs.mask: ", xs.mask.size())
        #x = x.squeeze(2)
        #mask = xs.mask.squeeze(1)  # (batch_size, seq_len)
        x = self.encoder(x, src_key_padding_mask=xs.mask, mask=generate_square_subsequent_mask(x.shape[0]).to(x.device))
        pooled_x = x[xs.lens - 1, torch.arange(x.shape[1])]  # (batch, nemb)

        if self.use_cond:
            cond_var = self.cond_embed(cond)  # (batch, nemb)
            cond_var = torch.tile(cond_var, (x.shape[0], 1, 1)) if batched else cond_var
            final_rep = torch.cat((x, cond_var), axis=-1) if batched else torch.cat((pooled_x, cond_var), axis=-1)
        else:
            final_rep = x if batched else pooled_x
        
        out: torch.Tensor = self.output(final_rep)
        ns = self.num_state_out
        
        if batched:
            out = out.transpose(1, 0).contiguous().reshape((-1, out.shape[2]))
            state_preds = out[xs.logit_idx, 0:ns]
            stop_logits = out[xs.logit_idx, ns : ns + 1]
            add_node_logits = out[xs.logit_idx, ns + 1 :]
        else:
            xs.num_graphs = out.shape[0]
            state_preds = out[:, 0:ns]
            stop_logits = out[:, ns : ns + 1]
            add_node_logits = out[:, ns + 1 :]

        return (
            GraphActionCategorical(
                xs,
                raw_logits=[stop_logits, add_node_logits],
                keys=[None, None],
                types=self.ctx.action_type_order,
                slice_dict={},
            ),
            state_preds,
        )


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, interpolation_factor=1.0, base=10000, base_rescale_factor=1.0):
        super().__init__()
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        assert interpolation_factor >= 1.0
        self.interpolation_factor = interpolation_factor

    def get_emb(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        t = t / self.interpolation_factor

        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)

        return freqs

    def forward(self, x, scale=1):
        x1, x2 = x.reshape(x.shape[:-1] + (2, -1)).unbind(dim=-2)
        xrot = torch.cat((-x2, x1), dim=-1)
        freqs = self.get_emb(x.shape[0], x.device)[:, None, :]
        return (x * freqs.cos() * scale) + (xrot * freqs.sin() * scale) 