# models/encoder.py

import os
import torch
import torch.nn as nn

from models.grkan import GRKANLinear

class ConvLayer(nn.Module):
    """Optional convolutional downsampling layer used in Informer."""
    def __init__(self, c_in):
        super().__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=padding,
            padding_mode='circular'
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # x: [B, L, D]
        x = self.downConv(x.permute(0, 2, 1))  # → [B, D, L]
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        return x.transpose(1, 2)               # → [B, L', D]


class EncoderLayer(nn.Module):
    """
    Single layer of the Informer encoder, with feed‑forward replaced
    by KAT‑style GR‑KAN blocks (GR1 → Linear1 → GR2 → Linear2).
    """
    def __init__(self,
                 attention,
                 d_model,
                 d_ff=None,
                 dropout=0.1,
                 activation="relu",
                 G=8,
                 m=5,
                 n=4,
                 coeff_dir="/user1/res/cvpr/soumyo.b_r/Understanding Informer"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # GR1: identity‑initialized rational block
        self.gr1 = GRKANLinear(
            in_dim=d_model,
            out_dim=d_model,
            groups=G,
            m=m,
            n=n,
            bias=False,
            init_activation=None,   # identity
            coeffs_dir=coeff_dir
        )
        # Linear1: replaces the original Conv1d(1×1)
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)

        # GR2: rational block approximating activation
        self.gr2 = GRKANLinear(
            in_dim=d_ff,
            out_dim=d_ff,
            groups=G,
            m=m,
            n=n,
            bias=False,
            init_activation=activation,  # 'relu' or 'gelu'
            coeffs_dir=coeff_dir
        )
        # Linear2: replaces the original Conv1d(1×1)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x, attn_mask=None):
        # x: [B, L, D]
        # 1) Self‑attention + residual
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)

        # 2) KAT‑style feed‑forward
        y = self.norm1(x)      # pre‑norm
        y = self.gr1(y)        # GR1 (identity at init)
        y = self.fc1(y)        # Linear1
        y = self.gr2(y)        # GR2 (≈activation)
        y = self.fc2(y)        # Linear2
        y = self.dropout(y)

        # 3) final residual + norm
        out = self.norm2(x + y)
        return out, attn


class Encoder(nn.Module):
    """Stacks multiple EncoderLayer blocks, optionally interleaved with ConvLayers."""
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x: [B, L, D]
        attns = []
        if self.conv_layers is not None:
            # interleave attention layers with convolutional downsampling
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            # final attention layer without a following conv
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            # pure transformer encoder
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class EncoderStack(nn.Module):
    """
    Multi‐resolution encoder: runs several Encoders on downsampled segments
    and concatenates their outputs along the time dimension.
    """
    def __init__(self, encoders, inp_lens):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x: [B, L, D]
        x_stack, all_attns = [], []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1] // (2 ** i_len)
            x_seg, attn = encoder(x[:, -inp_len:, :], attn_mask=attn_mask)
            x_stack.append(x_seg)
            all_attns.append(attn)
        # concatenate along the time axis
        x_cat = torch.cat(x_stack, dim=-2)
        return x_cat, all_attns
