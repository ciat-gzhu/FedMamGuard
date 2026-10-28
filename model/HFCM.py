import torch
import torch.nn as nn
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm


class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, l = x.size()
        n = l - 1
        x_minus_mu_square = (x - x.mean(dim=-1, keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=-1, keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.act(y)


class PermuteRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        return x


class HFCM(nn.Module):
    def __init__(self, hidden, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hfcm = nn.Sequential(
            nn.Conv1d(1, hidden//2, kernel_size=3, stride=2),
            SimAM(),
            nn.SiLU(),
            PermuteRMSNorm(hidden//2),
            nn.Conv1d(hidden//2, hidden, kernel_size=3, stride=2),
            SimAM(),
            nn.SiLU(),
            PermuteRMSNorm(hidden),
        )

    def forward(self, x):
        return self.hfcm(x)


class HFCM_NoAttn(nn.Module):
    def __init__(self, hidden, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hfcm = nn.Sequential(
            nn.Conv1d(1, hidden//2, kernel_size=3, stride=2),
            nn.SiLU(),
            PermuteRMSNorm(hidden//2),
            nn.Conv1d(hidden//2, hidden, kernel_size=3, stride=2),
            nn.SiLU(),
            PermuteRMSNorm(hidden),
        )

    def forward(self, x):
        return self.hfcm(x)
