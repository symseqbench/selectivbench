# SPDX-License-Identifier: GPL-3.0
# Copyright (c) 2025-present, SelectivBench Contributors

"""
State Space Model implementations for sequence modeling.

This module contains implementations of various state space models (SSMs) including:
- S4 (Structured State Space Sequence Model)
- Mamba and variants
- Transformer with various attention mechanisms
- Other recurrent architectures
"""

import math
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from einops import rearrange, repeat
from tqdm import tqdm


class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            if not self.transposed:
                X = rearrange(X, 'b ... d -> b d ...')
            # Use rand instead of binomial distribution for better performance
            mask_shape = X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
            mask = torch.rand(*mask_shape, device=X.device) < 1. - self.p
            X = X * mask * (1.0 / (1 - self.p))
            if not self.transposed:
                X = rearrange(X, 'b d ... -> b ... d')
            return X
        return X


class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt)  # (H)
        C = torch.view_as_complex(self.C)  # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H N L)
        C = C * (torch.exp(dtA) - 1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    def __init__(self, 
                 d_model, 
                 d_state=64, 
                 dropout=0.0, 
                 transposed=True, 
                 act_function='none',
                 **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        self.act_function = act_function
        self.act_output = 'glu'

        # Pointwise activation function
        assert self.act_function in ['relu', 'none']
        if self.act_function == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.GELU()

        # Use DropoutNd instead of Dropout2d (bugged in PyTorch 1.11)
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        if self.act_output == 'relu':
            self.output_linear = nn.Sequential(
                nn.Conv1d(self.h, self.h, kernel_size=1),
                nn.ReLU()
            )
        else:
            self.output_linear = nn.Sequential(
                nn.Conv1d(self.h, 2*self.h, kernel_size=1),
                nn.GLU(dim=-2),
            )


    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L) # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L) # (H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.act(y))
        y = self.output_linear(y)
        if not self.transposed:
            y = y.transpose(-1, -2)
        # Return a dummy state to satisfy this repo's interface
        return y, None


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        self.d = d_model

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm * self.d ** (-1./2)
        x_normed = x / (rms + self.eps)
        return x_normed * self.weight


class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.register_buffer('inv_freq', (1 / base) ** (torch.arange(0, dim, 2) / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        feat_size = x.shape[-1]
        assert feat_size % 2 == 0, "head dim needs to be divisible by 2"
        if seq_len != self.seq_len_cached:
            t = torch.arange(seq_len, device=x.device)
            freqs = torch.outer(t, self.inv_freq)
            self.seq_len_cached = seq_len
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        x1, x2 = x.chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)
    

class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 n_head=4,
                 dim_feedforward=None,
                 dropout=0.0,
                 rope_base=10000,
                 att_dropout=False):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        # Linear projections for queries, keys, and values
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Rotary embeddings
        self.rotary = Rotary(self.head_dim, base=rope_base)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # RMSNorm for normalization
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out_norm = RMSNorm(d_model)
        
        # SwiGLU feedforward layer (d_ff is typically set to 4 times the embedding size)
        self.linear1 = nn.Linear(d_model, 2 * (4 * d_model))
        self.linear2 = nn.Linear((4 * d_model), d_model)
        
        # Dropout
        if att_dropout:
            self.att_drop = dropout
        else:
            self.att_drop = 0.0

        self.dropout = nn.Dropout(dropout)

        self.beta = nn.Parameter(torch.ones(1))

    def swish(self, x):
        """Swish activation function with trainable beta."""
        return x * torch.sigmoid(self.beta * x)

    def swiglu(self, x):
        """SwiGLU with Swish activation."""
        x1, x2 = x.chunk(2, dim=-1)  # Split into two halves
        return self.swish(x1) * x2
    

    def forward(self, x, att_mask):

        # Input dimensions: (B, T, d_model)
        B, T, _ = x.shape
        
        # Linear projections for Q, K, V
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)  # (B, T, n_head, head_dim)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim)
        
        # Apply RMSNorm to Q and K
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Apply rotary embeddings to Q and K
        q = self.rotary(q)
        k = self.rotary(k)
       
        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
            dropout_p=self.att_drop
        )
        
        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, d_model)
        attn_output = self.out_proj(attn_output)
        
        # Residual connection with RMSNorm
        x = x + self.out_norm(attn_output)
        
        # Feedforward layer with SwiGLU
        ff_output = self.linear2(self.swiglu(self.linear1(x)))
        
        # Final residual connection
        x = x + self.dropout(ff_output)
        
        return x

def sliding_window_mask(seq_len, window_size, device):
    """
    Create a sliding window attention mask.
    
    Args:
        seq_len (int): Length of the sequence.
        window_size (int): Half the window size (number of tokens to attend on each side).
        device (torch.device): Device for the mask tensor.
    
    Returns:
        torch.Tensor: Sliding window mask of shape (seq_len, seq_len).
    """
    # Start with a matrix full of zeros (default: no masking)
    mask = torch.full((seq_len, seq_len), 0.0, device=device)

    # Set positions outside the window to -inf
    for i in range(seq_len):
        # Mask positions outside the window range
        mask[i, :max(0, i - window_size)] = float('-inf')
        mask[i, min(seq_len, i + 1):] = float('-inf')

    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
 
 
class CustomModel(nn.Module):
    def __init__(self, base_model, feat_size, hidden_dim=2048):
        super().__init__()
        self.base_model = base_model
        self.proj = nn.Linear(feat_size, hidden_dim)

    def forward(self, inputs):
        # inputs: [batch_size, seq_len, feat_size]
        hidden = self.proj(inputs)  # shape: [batch_size, seq_len, hidden_dim]
        return self.base_model(inputs_embeds=hidden)


class RecNet(nn.Module):
    def __init__(self,
                 model,
                 inp_size,
                 d_output, 
                 lr=0.1,
                 dropout=0.2,
                 layers=4,
                 d_model=256,
                 d_state=64,
                 dt_rank=0,
                 dt_relu=None,
                 taylor_order=None,
                 window_size=0,
                 n_head=4,
                 norm="layernorm",
                 reg=True,
                 att_dropout=False,
                 rope_base=10000,
                 pos_encoding=False,
                 visualize_layers=False,
                 act_function='none'
                 ):
        super(RecNet, self).__init__()

        self.n_layers = layers

        self.encoder = nn.Linear(inp_size, d_model)

        self.reg = reg

        self.model = model
        self.d_model = d_model
        self.d_state = d_state
        self.taylor_order = taylor_order
        self.n_head = n_head
        self.dropout = dropout
        self.rope_base = rope_base
        self.att_dropout = att_dropout

        self.norms = nn.ModuleList()
        self.drops = nn.ModuleList()
        self.layers = nn.ModuleList()

        self.act_function = act_function

        self.visualize_layers = visualize_layers

        if self.visualize_layers: 
            self.cell_state_history = [[] for _ in range(self.n_layers)]

        if dt_rank==0:
            self.dt_rank = "auto"
        else:
            self.dt_rank = dt_rank
        if dt_relu=="soft":
            self.dt_relu = None
        else:
            self.dt_relu = dt_relu

        self.normalize = (norm!="none")
            
        self.window_size = window_size

        self.model = model
        for layer in range(layers):
            if norm == "batchnorm":
                self.norms.append(nn.BatchNorm1d(self.d_model, momentum=0.05))

            elif norm == "layernorm":
                self.norms.append(nn.LayerNorm(self.d_model))
            self.drops.append(nn.Dropout(dropout))
            
            if model == "transformer":
                self.layers.append(TransformerDecoderLayer(d_model=self.d_model, 
                                                           n_head=self.n_head, 
                                                           dropout=self.dropout,
                                                           rope_base=self.rope_base,
                                                           att_dropout=self.att_dropout))

            elif model == "mamba":
                from mamba_ssm import Mamba
                self.layers.append(Mamba(
                    d_model=self.d_model,  # Model dimension
                    d_state=self.d_state,  # SSM state expansion factor
                    d_conv=4,  # Local convolution width
                    expand=2,  # Block expansion factor
                    layer_idx=layer,
                ))
            elif model == "mamba2":
                from mamba_ssm import Mamba2
                assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"
                head_dim = self.d_model // self.n_head
                self.layers.append(Mamba2(
                    d_model=self.d_model,  # Model dimension
                    d_state=self.d_state,  # SSM state expansion factor
                    d_conv=4,  # Local convolution width
                    expand=2,  # Block expansion factor
                    headdim=head_dim,
                    layer_idx=layer
                ))
            elif model == "gated_delta_net":
                from fla.layers import GatedDeltaNet
                assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"
                head_dim = self.d_model // self.n_head
               
                self.layers.append(GatedDeltaNet(
                    hidden_size=self.d_model,
                    head_dim=head_dim,
                    num_heads=self.n_head,
                    use_short_conv=True,
                    expand_v=2,  # Expansion ratio for the value dimension
                    conv_size=4
                ))
            elif model == "mesa_net":
                from fla.layers import MesaNet
                assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"
                head_dim = self.d_model // self.n_head
              
                self.layers.append(MesaNet(
                    hidden_size=self.d_model,
                    head_dim=head_dim,
                    num_heads=self.n_head,
                    use_short_conv=True,
                    expand_v=2,  # Expansion ratio for the value dimension
                    conv_size=4
                ))

            elif model == "gated_delta_product":
                from fla.layers import GatedDeltaProduct

                assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"
                head_dim = self.d_model // self.n_head
                
                self.layers.append(GatedDeltaProduct(
                    hidden_size=self.d_model,
                    head_dim=head_dim,
                    num_heads=self.n_head,
                    use_short_conv=True,
                    num_householder=4,
                    expand_v=2,  # Expansion ratio for the value dimension
                    conv_size=4
                ))
            elif model == "delta_net":
                from fla.layers import DeltaNet
                self.layers.append(DeltaNet(
                    hidden_size=self.d_model,
                    num_heads=self.n_head,
                    use_short_conv=True,
                    expand_k=1,
                    expand_v=2,
                    conv_size=4
                )) 

            elif model == "gla":
                from fla.layers import GatedLinearAttention
                self.layers.append(GatedLinearAttention(
                    hidden_size=self.d_model,
                    num_heads=self.n_head,
                    use_short_conv=True,
                    expand_k=1,
                    expand_v=2,
                    conv_size=4
                )) 
            elif model == "retnet":
                from fla.layers import MultiScaleRetention
                self.layers.append(MultiScaleRetention(
                    hidden_size=self.d_model,
                    num_heads=self.n_head,
                    use_short_conv=True,
                    expand_k=1,
                    expand_v=2,
                    conv_size=4
                )) 

            elif model == 's4':
                self.layers.append(S4D(d_model=self.d_model,
                                       d_state=self.d_state,
                                       dropout=dropout, 
                                       transposed=True,
                                       act_function=self.act_function,
                                       lr=min(0.001, lr)))
            elif model == 'gru':
                self.layers.append(nn.GRU(
                    input_size=self.d_model,
                    hidden_size=self.d_model,
                    num_layers=1,
                    bidirectional=False,
                    batch_first=True  # Assuming input tensor is (batch, seq, feature)
                ))

            else:
                raise ValueError(f"{model} is not defined")

        # Linear decoder
        self.decoder = nn.Linear(self.d_model, d_output)

    def inference(self, x, inference_params):

        # TODO: Correct
        if inference_params is not None:

            spar = []

            x = self.encoder(x)

            for layer, drop, norm in zip(self.layers, self.drops, self.norms):

                z = x
                seqlength = x.shape[1]
                lay_spar = [0,0]
                hidden = torch.zeros(x.shape).to(x.device)
                for t in range(seqlength):

                    step, step_spar = layer(z[:,t,:].unsqueeze(1), inference_params)
                    hidden[:,t,:] = step.squeeze()
                    lay_spar[0]+=step_spar[0]/seqlength
                    lay_spar[1]+=step_spar[0]/seqlength
                
                # Dropout on the output of the S4 block
                z = drop(hidden)

                # Residual connection
                x = z + x

                # Postnorm
                if self.normalize:
                    x = norm(x)

                act3_spar = torch.sum(x==0)/x.numel()
                lay_spar.append(act3_spar)
                spar.append(lay_spar)

            if not self.reg:
                # Pooling: average pooling over the sequence length
                x = x.mean(dim=1)

            x = self.decoder(x)  # (B, d_model) -> (B, d_output)

            return x, spar      

    def forward(self, x):

        if self.model == 'transformer':
            seq_len = x.shape[1]
            if self.window_size == 0:
                tgt_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)

                # Replace 1s with -inf and 0s with 0.0
                tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float('-inf'))
                tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, float(0.0)).to(x.device)
            else:
                tgt_mask = sliding_window_mask(seq_len, self.window_size, x.device)

        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        for l, (layer, drop, norm) in enumerate(zip(self.layers, self.drops, self.norms)):
            z = x

            # Apply layer: handle different model types
            if self.model in ('s4'):
                z, _ = layer(z.transpose(-1, -2))
                z = z.transpose(-1, -2)
            elif self.model == 'transformer':
                z = layer(z, att_mask=tgt_mask)
            elif self.model in ('delta_net', 'gated_delta_net', 'gla', 'retnet', 'gated_delta_product', 'mesa_net'):
                z = layer(z)[0]
            else:
                try:
                    z, _ = layer(z)
                except:
                    z = layer(z)

            # Dropout on the output of the S4 block
            z = drop(z)

            # Residual connection
            x = z + x

            # Postnorm
            if self.normalize:
                x = norm(x)

            # This is for plotting the output of each layer
            if self.visualize_layers:
                self.cell_state_history[l] = x.detach().cpu().numpy()

        if not self.reg:
            # Pooling: average pooling over the sequence length
            x = x.mean(dim=1)

        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x

    
if __name__ == "__main__":
    model = RecNet(model='mamba',
                   inp_size=20,
                   d_output=20).to('cuda')

    x = torch.randn(64, 60, 20).to('cuda')
    y = model(x)

    print(y.shape)