"""Selective state-space block (Mamba, Gu & Dao 2023).

Uses the official CUDA implementation from `mamba_ssm` when it is installed.
Otherwise falls back to `MambaRef`, a pure-PyTorch reference with the same
parameterisation (in_proj, depthwise conv1d, x_proj, dt_proj, A_log, D,
out_proj), so parameter counts are identical. The reference scan is a Python
loop over time: correct but slow, meant for CPU tests and debugging only.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # pragma: no cover - depends on the CUDA build
    from mamba_ssm import Mamba as _MambaCUDA
    HAS_MAMBA_SSM = True
except Exception:  # ImportError or a CUDA/compile error
    _MambaCUDA = None
    HAS_MAMBA_SSM = False


class MambaRef(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, dt_rank="auto",
                 dt_min=1e-3, dt_max=1e-1, dt_init_floor=1e-4):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # dt bias initialised so that softplus(bias) ~ LogUniform(dt_min, dt_max)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x):  # x: (B, L, d_model)
        B_, L, _ = x.shape
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u = self.conv1d(u.transpose(1, 2))[..., :L].transpose(1, 2)  # causal
        u = F.silu(u)
        dbc = self.x_proj(u)
        dt, Bm, Cm = torch.split(dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                       # (B, L, d_inner), Eq. (14)
        A = -torch.exp(self.A_log.float())                      # (d_inner, N)
        h = u.new_zeros(B_, self.d_inner, self.d_state)
        ys = []
        for t in range(L):                                      # Eq. (15)
            dA = torch.exp(dt[:, t, :, None] * A)
            dBu = dt[:, t, :, None] * Bm[:, t, None, :] * u[:, t, :, None]
            h = dA * h + dBu
            ys.append((h * Cm[:, t, None, :]).sum(-1))
        y = torch.stack(ys, dim=1) + u * self.D
        y = y * F.silu(z)
        return self.out_proj(y)


def make_mamba(d_model, d_state=64, d_conv=4, expand=2, force_reference=False):
    if HAS_MAMBA_SSM and not force_reference:
        return _MambaCUDA(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    return MambaRef(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
