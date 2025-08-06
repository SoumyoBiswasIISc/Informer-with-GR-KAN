import torch
import torch.nn as nn
import numpy as np

class Rational(nn.Module):
    def __init__(self, m, n, groups, init_identity=True):
        super().__init__()
        self.groups = groups
        self.m = m
        self.n = n
        # numerator coeffs: [groups, m+1]
        self.a = nn.Parameter(torch.zeros(groups, m+1))
        # denominator coeffs: [groups, n+1]
        self.b = nn.Parameter(torch.zeros(groups, n+1))
        if init_identity:
            with torch.no_grad():
                # R(x)=x
                self.a.zero_()
                self.a[:, 1] = 1.0
                self.b.zero_()
                self.b[:, 0] = 1.0

    def forward(self, x):
        B, L, D = x.shape
        G = self.groups
        assert D % G == 0, "D must be divisible by groups"
        xg = x.view(B, L, G, D // G)  # (B, L, G, D//G)

        # Horner's method for numerator
        Px = torch.zeros_like(xg)
        for i in range(self.m, -1, -1):
            ai = self.a[:, i].view(1, 1, G, 1)  # (1, 1, G, 1)
            Px = Px * xg + ai

        # Horner's method for denominator
        Qx = torch.zeros_like(xg)
        for j in range(self.n, -1, -1):
            bj = self.b[:, j].view(1, 1, G, 1)
            Qx = Qx * xg + bj

        # Your intended denominator: 1 + |Q(x)|
        denom = 1 + Qx.abs()

        out = Px / denom
        return out.view(B, L, D)

class GRKANLinear(nn.Module):
    def __init__(self, in_dim, out_dim, groups, m=5, n=4,
                 bias=True, init_activation=None, coeffs_dir=None):
        super().__init__()
        # Rational block
        self.rational = Rational(m, n, groups,
                                 init_identity=(init_activation is None))

        # If we have pre-fitted activation approximator, load it
        if init_activation and coeffs_dir:
            # Load coefficients
            a = np.load(f"{coeffs_dir}/rational_num_coeffs.npy")  # (m+1,)
            b = np.load(f"{coeffs_dir}/rational_den_coeffs.npy")  # (n,)

            G = groups
            a_t = torch.from_numpy(a).float().unsqueeze(0).repeat(G, 1)   # (G, m+1)
            # make b_t shape (G, n+1) with first col = 1
            b0 = torch.ones(G, 1)
            b_rest = torch.from_numpy(b).float().unsqueeze(0).repeat(G, 1)  # (G, n)
            b_t = torch.cat([b0, b_rest], dim=1)                            # (G, n+1)

            with torch.no_grad():
                self.rational.a.copy_(a_t)
                self.rational.b.copy_(b_t)

        # Linear projection at the end
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        # x: (B, L, in_dim)
        z = self.rational(x)
        return self.linear(z)