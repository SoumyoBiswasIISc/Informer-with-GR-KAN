# models/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.grkan import GRKANLinear

class DecoderLayer(nn.Module):
    """
    Single layer of the Informer decoder, with feed‑forward replaced
    by KAT‑style GR‑KAN blocks (GR1 → Linear1 → GR2 → Linear2).
    """
    def __init__(self,
                 self_attention,
                 cross_attention,
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

        self.self_attention = self_attention
        self.cross_attention = cross_attention

        # three LayerNorms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        # —— Replace Conv1d→activation→Conv1d with GRKAN feed‑forward ——
        # GR1: identity
        self.gr1 = GRKANLinear(
            in_dim=d_model,
            out_dim=d_model,
            groups=G,
            m=m,
            n=n,
            bias=False,
            init_activation=None,
            coeffs_dir=coeff_dir
        )
        # Linear1 (replaces conv1)
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)

        # GR2: approximates activation
        self.gr2 = GRKANLinear(
            in_dim=d_ff,
            out_dim=d_ff,
            groups=G,
            m=m,
            n=n,
            bias=False,
            init_activation=activation,
            coeffs_dir=coeff_dir
        )
        # Linear2 (replaces conv2)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # 1) masked self‑attention + residual + norm
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)

        # 2) cross‑attention + residual + norm
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        x = self.norm2(x)

        # 3) KAT‑style feed‑forward
        y = self.gr1(x)    # GR1 identity
        y = self.fc1(y)    # Linear1
        y = self.gr2(y)    # GR2 ≈ activation
        y = self.fc2(y)    # Linear2
        y = self.dropout(y)

        # 4) final residual + norm
        out = self.norm3(x + y)
        return out


class Decoder(nn.Module):
    """Stacks multiple DecoderLayer blocks, followed by optional final LayerNorm."""
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x
