import math
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from dataclasses import dataclass
from datetime import datetime

# ----------------- config -----------------
def exists(val): return val is not None

def class_to_str(cls):
    return str(cls)[8:-2]

def save_dataclass_yaml(data_obj, file_path):
    conf = OmegaConf.structured(data_obj)
    with open(file_path, 'w') as f:
        OmegaConf.save(config=conf, f=f)

def save_dict_yaml(dict_obj, file_path):
    conf = OmegaConf.create(dict_obj)
    with open(file_path, 'w') as f:
        OmegaConf.save(config=conf, f=f)

def load_config(file_path):
    return OmegaConf.load(f"{file_path}")

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config: raise KeyError("Expected key `target` to instantiate.")
    if not "params" in config: print(f"[WARNING] Expected key `params` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

class Config_Model(nn.Module):
    """A basic `nn.Module` with IO functionality."""
    def __init__(self): super().__init__()
    
    #---------------------
    
    def get_config(self, save_path=None, without_metadata=False):
        if not without_metadata:       
            config = {}
            config["target"]         = class_to_str(type(self)) 
            config["save_path"]      = save_path
            config["save_datetime"]  = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
            config["params"]         = self.params_config  
        else:
            config = self.params_config  
        
        self.config = config        
        return config
    
    def store_model(self, config_path: str=None, save_path: str=None, without_metadata=False):        
    
        config = self.get_config(save_path, without_metadata)
    
        if exists(config_path):
            if without_metadata: save_dataclass_yaml(config, config_path)
            else               : save_dict_yaml(config, config_path)            
                       
        if exists(save_path):
            torch.save(self.state_dict(), save_path)     
    
    #---------------------
    
    @staticmethod
    def from_config(config, device: torch.device, save_path: str=None):  
        """Use this if we have a loaded config. Maybe within other classes (e.g. pipeline and nested models)"""
        
        model = instantiate_from_config(config)
        model = model.to(device) 
        print(f"[INFO]: `{class_to_str(type(model))}` instantiated from given config on {device}.")
        
        #--------------------------------        
        if not exists(save_path):            
            if "save_path" in config:
                save_path = config["save_path"]
            else:
                print("[INFO]: Found no key `save_path` path in config.")
                                  
        if exists(save_path):
            model.load_state_dict(torch.load(save_path, map_location=torch.device(device).type, weights_only=True), strict=True)
        else:
            print(f"[INFO]: `{class_to_str(type(model))}`. No save_path` provided. No state dict loaded.")

        return model
    
    @staticmethod
    def from_config_file(config_path, device: torch.device, save_path: str=None):    
        config = load_config(config_path)
        return Config_Model.from_config(config, device, save_path)       

# ----------------- layers -----------------

class DownBlock2D(nn.Module):  
    """A 2d down scale block."""
    def __init__(self, in_ch, out_ch, kernel_size=2, stride=2, padding=0, use_conv=True):
        super().__init__()  
        self.use_conv = use_conv                
        if self.use_conv: 
            self.conv1  = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)           
        else:       
            self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)    
            self.convId   = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding="same") if in_ch!=out_ch else nn.Identity()   
        
    def forward(self, x):               
        if self.use_conv:
            #x = F.pad(x, pad=(0,0,0,1), mode="constant", value=0) #for 2d: pad=(0,1,0,1)
            x = self.conv1(x)          
        else:
            x = self.avg_pool(x)
            x = self.convId(x)           
        return x

class UpBlock2D(nn.Module):  
    """A 2d up scale block."""
    def __init__(self, in_ch, out_ch, kernel_size=2, stride=2, padding=0, use_conv=True):
        super().__init__()  
        self.use_conv  = use_conv                 
        self.up_sample = nn.Upsample(scale_factor=kernel_size)           
        if self.use_conv: 
            self.conv1  = nn.Conv2d(in_ch, out_ch, kernel_size=(1,3), stride=1, padding="same")  
        else:             
            self.convId = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding="same") if in_ch!=out_ch else nn.Identity()   
            
    def forward(self, x):        
        x = self.up_sample(x)       
        if self.use_conv: 
            x = self.conv1(x)          
        else:             
            x = self.convId(x)                
        return x

class ResDownBlock2D(nn.Module):  
    """A 2d residual down scale block."""
    def __init__(self, in_ch, out_ch, kernel_size=2, stride=2, padding=0):
        super().__init__()     
        self.act    = nn.SiLU() 
        self.conv1  = nn.Conv2d( in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding) 
        self.norm   = torch.nn.GroupNorm(num_groups=16, num_channels=out_ch, eps=1e-5, affine=True)        
        self.convId = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding="same") if in_ch!=out_ch else lambda x:x
        self.down   = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
               
    def forward(self, x):         
        r1 = self.conv1(x)
        r1 = self.norm(r1)
        r1 = self.act(r1)
        
        r2 = self.convId(x)
        r2 = self.down(r2)
        return self.act(r1 + r2)

class ResUpBlock2D(nn.Module):  
    """A 2d residual up scale block."""
    def __init__(self, in_ch, out_ch, kernel_size=2, stride=2, padding=0):
        super().__init__()     
        self.act    = nn.SiLU()
        self.conv1  = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding) 
        self.norm   = torch.nn.GroupNorm(num_groups=16, num_channels=out_ch, eps=1e-6, affine=True)        
        self.convId = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding="same") if in_ch!=out_ch else nn.Identity()
        self.up     = nn.Upsample(scale_factor=kernel_size)
               
    def forward(self, x):         
        r1 = self.conv1(x)
        r1 = self.norm(r1)
        r1 = self.act(r1)
        
        r2 = self.convId(x)
        r2 = self.up(r2)
        return self.act(r1 + r2)

class ResBlock2D(nn.Module):
    """A 2d residual block."""
    def __init__(self, in_ch, out_ch, kernel_size, skip=True):
        super().__init__()             
        self.act   = nn.SiLU()             
        self.conv1 = nn.Conv2d( in_ch, out_ch, kernel_size=kernel_size, stride=1, padding ="same")        
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding ="same")
        
        self.norm1  = torch.nn.GroupNorm(num_groups=32, num_channels=in_ch) #, eps=1e-6, affine=True)         
        self.norm2  = torch.nn.GroupNorm(num_groups=32, num_channels=out_ch) #, eps=1e-6, affine=True)  
        
        self.skip  = skip 
        if self.skip:        
            self.skip_connection= nn.Conv2d(in_ch,  out_ch, kernel_size=1, stride=1, padding ="same") if in_ch!=out_ch else nn.Identity()
         
    def forward(self, x):      
        
        #in layers
        h = self.norm1(x) 
        h = self.act(h)
        h = self.conv1(h)  
        
        #out layers
        h = self.norm2(h) 
        h = self.act(h)
        h = self.conv2(h)
    
        #----
    
        if not self.skip: return h  
                   
        return self.skip_connection(x) + h

class ResBlock2D_Conditional(nn.Module):
    """A 2d residual block with input of a time-step $t$ embedding."""
    def __init__(self, in_ch, out_ch, t_emb_size, kernel_size, skip=True):
        super().__init__()             
        self.act   = nn.SiLU()             
        self.conv1 = nn.Conv2d( in_ch, out_ch, kernel_size=kernel_size, stride=1, padding ="same")        
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding ="same")
        
        self.norm1  = torch.nn.GroupNorm(num_groups=32, num_channels=in_ch) #, eps=1e-6, affine=True)         
        self.norm2  = torch.nn.GroupNorm(num_groups=32, num_channels=out_ch) #, eps=1e-6, affine=True)  
        
        self.skip  = skip 
        if self.skip:        
            self.skip_connection= nn.Conv2d(in_ch,  out_ch, kernel_size=1, stride=1, padding ="same") if in_ch!=out_ch else nn.Identity()
        
        self.t_proj = nn.Linear(t_emb_size, out_ch)    
               
    def forward(self, x, t_emb):      
        
        #in layers
        h = self.norm1(x) 
        h = self.act(h)
        h = self.conv1(h)  
        
        #embed
        h = h + self.t_proj(t_emb)[:, :, None, None]
        
        #out layers
        h = self.norm2(h) 
        h = self.act(h)
        h = self.conv2(h)
    
        #----
    
        if not self.skip: return h  
                   
        return self.skip_connection(x) + h

class FeedForward(nn.Module):
    """A small dense feed-forward network as used in `transformers`."""
    def __init__(self, in_ch, out_ch, inner_mult=1):
        super().__init__()
        self.proj1 = nn.Linear(in_ch, in_ch*inner_mult)   
        self.proj2 = nn.Linear(in_ch*inner_mult, out_ch) 
        self.act   = nn.SiLU()
        
    def forward(self, x):
        x = self.proj1(x)
        x = self.act(x)
        x = self.proj2(x) 
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
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len , embedding_dim]``
        """
        x = x + self.pe[None, :x.size(1)]
        return self.dropout(x)

class TimeEmbedding(PositionalEncoding):
    """A time embedding layer"""
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__(d_model, dropout, max_len)             
        self.ff = FeedForward(d_model, d_model)  
       
    def forward(self, t: torch.Tensor):       
        x = self.pe[t]       
        x = self.ff(x)               
        return self.dropout(x)

class PositionalEncodingTransposed(PositionalEncoding):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__(d_model, dropout, max_len)        
        self.pe = torch.permute(self.pe, (1, 0)) # [max_len, d_model] to [d_model, max_len]
               
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, embedding_dim, seq_len]``
        """
        x = x + self.pe[None, :, :x.size(2)]
        return self.dropout(x)

class PositionalEncoding2D(PositionalEncodingTransposed):
    """A 2D absolute pos encoding layer."""
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__(d_model=d_model//2, dropout=dropout, max_len=max_len)    
        self.d_model_half = d_model//2    
        # self.proj         = nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, padding ="same")      
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, gate_color, space , time]``
        """
                                          
        p1 = self.pe[None, :, :x.size(2),       None] #space encoding
        p2 = self.pe[None, :,       None, :x.size(3)] #time encoding 
                                
        x[:, :self.d_model_half] = x[:, :self.d_model_half] + p1
        x[:, self.d_model_half:] = x[:, self.d_model_half:] + p2
        
        # x = self.proj(x)
        
        return self.dropout(x)

class PositionalEncoding2DSpaceOnly(PositionalEncodingTransposed):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__(d_model=d_model, dropout=dropout, max_len=max_len)      
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, gate_color, space , time]``
        """
                                          
        p1 = self.pe[None, :, :x.size(2), None] #space encoding        
        return self.dropout(x + p1)

# ----------------- transformers -----------------

class BasisSelfAttnBlock(nn.Module):
    """A self attention block, i.e. a `transformer` encoder."""
    def __init__(self, ch, num_heads, dropout=0):
        super().__init__()
        self.self_att  = nn.MultiheadAttention(ch, num_heads=num_heads, batch_first=False) #[t, b, c]
        self.ff    = FeedForward(ch, ch)   
        self.norm1 = nn.LayerNorm(ch)
        self.norm2 = nn.LayerNorm(ch)
        self.drop  = nn.Dropout(dropout)
               
    def forward(self, x, attn_mask=None, key_padding_mask=None, need_weights=False):
        #x     ... [  t, batch, ch]       
        #c_emb ... [seq, batch, ch]
        
        self_out    = self.norm1(x)  
        self_out, _ = self.self_att(self_out, key=self_out, value=self_out, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=need_weights)
        self_out    = self.drop(self_out) + x      
        
        feed_out = self.norm2(self_out)              
        feed_out = self.ff(feed_out)
        feed_out = self.drop(feed_out) + self_out            
                   
        return feed_out     

class BasisCrossAttnBlock(nn.Module):
    """A cross attention block, i.e. a `transformer` decoder."""
    def __init__(self, ch, cond_emb_size, num_heads, dropout=0.0):
        super().__init__()
        self.self_att  = nn.MultiheadAttention(ch, num_heads=num_heads, batch_first=False) #[t, b, c]
        self.cross_att = nn.MultiheadAttention(ch, num_heads=num_heads, batch_first=False) 
        self.ff    = FeedForward(ch, ch)   
        self.norm1 = nn.LayerNorm(ch)
        self.norm2 = nn.LayerNorm(ch)
        self.norm3 = nn.LayerNorm(ch)
        self.drop  = nn.Dropout(dropout)
        
    def forward(self, x, c_emb, attn_mask=None, key_padding_mask=None, need_weights=False):
        #x     ... [  t, batch, ch]       
        #c_emb ... [seq, batch, ch]
        
        self_out    = self.norm1(x)  
        self_out, _ = self.self_att(self_out, key=self_out, value=self_out, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=need_weights)
        self_out    = self.drop(self_out) + x      
        
        cross_out    = self.norm2(self_out)   
        cross_out, _ = self.cross_att(cross_out, key=c_emb, value=c_emb, need_weights=need_weights)
        cross_out    = self.drop(cross_out) + self_out         
        
        feed_out = self.norm3(cross_out)              
        feed_out = self.ff(feed_out)
        feed_out = self.drop(feed_out) + cross_out            
                   
        return feed_out     

class SpatialTransformerSelfAttn(nn.Module):
    """A spatial residual `transformer`, only uses self-attention."""
    def __init__(self, ch, num_heads, depth, dropout=0.0):
        super().__init__()       
        self.norm               = torch.nn.GroupNorm(num_groups=32, num_channels=ch, eps=1e-6, affine=True)
        self.transformer_blocks = nn.ModuleList([BasisSelfAttnBlock(ch, num_heads, dropout) for d in range(depth)])
        
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        #x      ... [batch, ch, space, time]  
        #c_emb  ... [batch, seq, ch]
        b, ch, space, time = x.shape
            
        x_in = x
        
        #-------------------------
        x = self.norm(x) 
        
        x = torch.reshape(x, (b, ch, space*time))
        x = torch.permute(x, (2, 0, 1)).contiguous()           # to [t, batch, ch]    
        
        #-------------------------   
        # x = self.proj_in(x) #NEW   only used so that ch is a multiple of heads
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, attn_mask, key_padding_mask)
                
        # feed_out = self.proj_out(feed_out) #NEW
        #-------------------------
            
        x = torch.permute(x, (1, 2, 0))              # back to [batch, ch, t] 
        x = torch.reshape(x, (b, ch, space, time)).contiguous()
                
        return x + x_in

class SpatialTransformer(nn.Module):
    """A spatial residual `transformer`, uses self- and cross-attention on conditional input."""
    
    def __init__(self, ch, cond_emb_size, num_heads, depth, dropout=0.0):
        super().__init__()       
        self.cat_proj           = nn.Linear(cond_emb_size, ch)  
        self.norm               = torch.nn.GroupNorm(num_groups=32, num_channels=ch, eps=1e-6, affine=True)
        self.transformer_blocks = nn.ModuleList([BasisCrossAttnBlock(ch, cond_emb_size, num_heads, dropout) for d in range(depth)])
        
    def forward(self, x, c_emb, attn_mask=None, key_padding_mask=None):
        #x      ... [batch, ch, space, time]  
        #c_emb  ... [batch, seq, ch]
        b, ch, space, time = x.shape
            
        x_in = x
        
        #-------------------------
        x = self.norm(x) 
        
        x = torch.reshape(x, (b, ch, space*time))
        x = torch.permute(x, (2, 0, 1)).contiguous()           # to [t, batch, ch]    
       
        c_emb = self.cat_proj(c_emb)        
        c_emb = torch.permute(c_emb, (1, 0, 2)).contiguous()  # to [seq, batch, ch]
        
        #-------------------------   
        # x = self.proj_in(x) #NEW   only used so that ch is a multiple of heads
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, c_emb, attn_mask, key_padding_mask)
                
        # feed_out = self.proj_out(feed_out) #NEW
        #-------------------------
            
        x = torch.permute(x, (1, 2, 0))              # back to [batch, ch, t] 
        x = torch.reshape(x, (b, ch, space, time)).contiguous()
                
        return x + x_in

# ----------------- utils -----------------

def load_data(path):
    npz = np.load(path, allow_pickle=True)
    # expected keys: 'seqs', 'unitaries', 'lengths'
    return npz['seqs'], npz['unitaries'], npz['lengths']

def unitary_to_tensor(U: torch.Tensor) -> torch.Tensor:
    """
    U: [B, N, N] complex tensor or numpy array
    return: [B, 2, N, N] float tensor (real/imag)
    """
    if not torch.is_tensor(U):
        U = torch.from_numpy(U)
    if not torch.is_complex(U):
        raise ValueError("U must be complex dtype")
    x = torch.stack([U.real, U.imag], dim=1).float()
    return x

# ----------------- dataset -----------------
class UnitaryGateLenDataset(Dataset):
    def __init__(self, U_arr, lengths):
        """
        U_arr: [B, N, N] complex np.array
        lengths: [B] int (gate length labels)
        """
        self.U = torch.from_numpy(U_arr)  # complex tensor
        self.y = torch.from_numpy(lengths).long()
    def __len__(self):
        return self.U.shape[0]
    def __getitem__(self, idx):
        return self.U[idx], self.y[idx]

# ----------------- config holder -----------------
@dataclass
class Unitary_encoder_config:
    cond_emb_size: int
    model_features: list
    num_heads: int
    transformer_depths: list
    dropout: float

# ----------------- model -----------------
class Unitary_encoder(Config_Model):
    """Encoder for unitary conditions."""
    def __init__(self, cond_emb_size, model_features=None, num_heads=8, transformer_depths=(4, 4), dropout=0.1):
        super().__init__()
        self.cond_emb_size = cond_emb_size

        if model_features is None:
            in_ch   = 2
            mid_ch1 = cond_emb_size // 4
            mid_ch2 = cond_emb_size // 2
            out_ch  = cond_emb_size
            model_features = [in_ch, mid_ch1, mid_ch2, out_ch]
        else:
            assert len(model_features) == 4
            in_ch, mid_ch1, mid_ch2, out_ch = model_features

        self.params_config = Unitary_encoder_config(cond_emb_size, model_features, num_heads, list(transformer_depths), dropout)

        # conv padding="same"은 1x1에서 무의미. downblock은 내부 정의에 따름.
        self.conv_in = nn.Conv2d(in_ch, mid_ch1, kernel_size=1, stride=1, padding=0)
        self.pos_enc = PositionalEncoding2D(d_model=mid_ch1)

        self.down1 = DownBlock2D(mid_ch1, mid_ch2, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

        assert len(transformer_depths) == 2
        self.spatialTransformer1 = SpatialTransformerSelfAttn(
            mid_ch1, num_heads=num_heads, depth=transformer_depths[0], dropout=dropout
        )
        self.spatialTransformer2 = SpatialTransformerSelfAttn(
            mid_ch2, num_heads=num_heads, depth=transformer_depths[1], dropout=dropout
        )

        self.head = nn.Conv2d(mid_ch2, out_ch, kernel_size=1, stride=1, padding=0)

        self._init_weights()

    def _init_weights(self):
        self.head.weight.data.zero_()

    def forward(self, x):
        # x: [B, 2, 2^n, 2^n]
        b, *_ = x.shape

        x = self.conv_in(x)
        x = self.pos_enc(x)

        x = self.spatialTransformer1(x)
        x = self.down1(x)

        x = self.spatialTransformer2(x)

        x = self.head(x)
        x = torch.reshape(x, (b, self.cond_emb_size, -1))
        x = torch.permute(x, (0, 2, 1))  # [B, seq, ch]
        return x

class GateLenHead(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=None, dropout=0.1):
        super().__init__()
        if hidden is None:
            hidden = in_dim // 2
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes)
        )
    def forward(self, z_seq):
        # z_seq: [B, seq, in_dim]
        z = z_seq.mean(dim=1)  # GAP
        return self.fc(z)      # [B, n_classes]

class GateLenPredictor(nn.Module):
    def __init__(self, encoder: nn.Module, cond_emb_size: int, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head    = GateLenHead(cond_emb_size, n_classes)
    def forward(self, U):
        x = unitary_to_tensor(U)     # [B, 2, N, N]
        z_seq = self.encoder(x)      # [B, seq, cond_emb_size]
        logits = self.head(z_seq)    # [B, n_classes]
        return logits

# ----------------- train / eval -----------------
def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    for U_batch, y_batch in loader:
        U_batch = U_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(U_batch)
        loss = F.cross_entropy(logits, y_batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * y_batch.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for U_batch, y_batch in loader:
        U_batch = U_batch.to(device)
        y_batch = y_batch.to(device)
        pred = model(U_batch).argmax(dim=1)
        correct += (pred == y_batch).sum().item()
        total += y_batch.size(0)
    return correct / total

@torch.no_grad()
def test_accuracy(model_path, npz_path, batch_size=1024, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 1) load data (assumes separate test arrays in npz)
    npz = np.load(npz_path, allow_pickle=True)
    U_test      = npz["unitaries"]      # shape [T, N, N], complex
    lengths_test= npz["lengths"]        # shape [T]

    test_ds = UnitaryGateLenDataset(U_test, lengths_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 2) rebuild model (sizes must match training)
    cond_emb_size = 256
    max_len       = 12
    model = GateLenPredictor(
        Unitary_encoder(cond_emb_size),
        cond_emb_size,
        n_classes=max_len + 1
    ).to(device)

    # 3) load weights
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 4) measure accuracy
    batch_accs = []
    for U_batch, y_batch in test_loader:
        U_batch = U_batch.to(device)
        y_batch = y_batch.to(device)
        preds   = model(U_batch).argmax(dim=1)
        batch_accs.append((preds - y_batch).float().mean().item())

    mean_acc = float(np.mean(batch_accs))
    std_acc  = float(np.std(batch_accs))
    return mean_acc, std_acc

@torch.no_grad()
def test_accuracy_per_length(model_path, npz_path, batch_size=1024, device=None, tolerance=1):
    """Test accuracy broken down by each gate length.
    
    Args:
        tolerance: Allow predictions within ±tolerance to be considered correct.
                  Default is 1, meaning ±1 is acceptable.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 1) load data
    npz = np.load(npz_path, allow_pickle=True)
    U_test = npz["unitaries"]      # shape [T, N, N], complex
    lengths_test = npz["lengths"]  # shape [T]

    test_ds = UnitaryGateLenDataset(U_test, lengths_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 2) rebuild model
    cond_emb_size = 256
    max_len = 12
    model = GateLenPredictor(
        Unitary_encoder(cond_emb_size),
        cond_emb_size,
        n_classes=max_len + 1
    ).to(device)

    # 3) load weights
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 4) collect predictions and true labels
    all_preds = []
    all_labels = []
    
    for U_batch, y_batch in test_loader:
        U_batch = U_batch.to(device)
        y_batch = y_batch.to(device)
        preds = model(U_batch).argmax(dim=1)
        
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # 5) calculate accuracy per length
    unique_lengths = np.unique(all_labels)
    length_stats = {}
    
    print(f"{'Length':<8} {'Count':<8} {'Accuracy':<12} {'Std':<12}")
    print("-" * 44)
    
    for length in sorted(unique_lengths):
        mask = all_labels == length
        if np.sum(mask) == 0:
            continue
            
        length_preds = all_preds[mask]
        length_labels = all_labels[mask]
        
        # Calculate accuracy for this length with tolerance
        # Allow predictions within ±tolerance to be considered correct
        diff = np.abs(length_preds - length_labels)
        correct = (diff <= tolerance).astype(float)
        acc_mean = np.mean(correct)
        acc_std = np.std(correct)
        count = len(length_labels)
        
        length_stats[int(length)] = {
            'count': count,
            'accuracy': acc_mean,
            'std': acc_std
        }
        
        print(f"{length:<8} {count:<8} {acc_mean:.4f}      {acc_std:.4f}")
    
    # Overall stats with tolerance
    overall_diff = np.abs(all_preds - all_labels)
    overall_correct = (overall_diff <= tolerance).astype(float)
    overall_acc = np.mean(overall_correct)
    overall_std = np.std(overall_correct)
    
    print("-" * 44)
    print(f"{'Overall':<8} {len(all_labels):<8} {overall_acc:.4f}      {overall_std:.4f}")
    
    return length_stats

# ----------------- main -----------------
def main(
    npz_path: str,
    cond_emb_size: int = 256,
    max_len: int = 12,
    batch_size: int = 1024,
    epochs: int = 20,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    val_ratio: float = 0.1,
    num_workers: int = 8,
    seed: int = 42,
    device: str = None
):
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    seqs, unitaries, lengths = load_data(npz_path)

    # dataset / split
    ds = UnitaryGateLenDataset(unitaries, lengths)
    n_total = len(ds)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    encoder = Unitary_encoder(cond_emb_size=cond_emb_size)
    model   = GateLenPredictor(encoder, cond_emb_size=cond_emb_size, n_classes=max_len + 1).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, device)
        acc     = evaluate(model, val_loader, device)
        print(f"{epoch:03d} | loss {tr_loss:.4f} | val_acc {acc:.4f}")

    # save
    torch.save(model.state_dict(), "gatelen_predictor.pt")

if __name__ == "__main__":
    # 예시 실행: main("data.npz", max_len=20)
    # main("train_data.npz")
    
    # Overall accuracy
    acc, std = test_accuracy("gatelen_predictor.pt", "test_data.npz")
    print(f"Overall test_acc = {acc:.4f} ± {std:.4f}")
    print()
    
    # Per-length accuracy breakdown
    print("Per-length accuracy breakdown (±1 tolerance):")
    length_stats = test_accuracy_per_length("gatelen_predictor.pt", "test_data.npz", tolerance=0)
    
    # Return the statistics for further analysis if needed
    print("\nDetailed statistics available in 'length_stats' dictionary")
