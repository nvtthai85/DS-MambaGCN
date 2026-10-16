"""DS-MambaGCN: dual-stream Mamba-GCN for skeleton-based action recognition.

Switches (all set from the YAML config, see configs/):
  topology : "phys" | "phys+data" | "phys+adapt" | "full"          (Table 14)
  coupling : "parallel" | "serial_gcn_mamba" | "serial_mamba_gcn"
             | "spatial_only" | "temporal_only"                      (Tables 13/15)
  fusion   : "csaf" | "xattn_only" | "gate_only" | "add" | "concat"  (Tables 13/16)
  scan     : "bi" | "fwd"                                            (Table 13)
  use_motion: bool                                                   (Table 13)

Equation and table numbers refer to the paper.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import normalized_adjacency, physical_adjacency
from .mamba import make_mamba

TOPOLOGIES = ("phys", "phys+data", "phys+adapt", "full")
COUPLINGS = ("parallel", "serial_gcn_mamba", "serial_mamba_gcn", "spatial_only", "temporal_only")
FUSIONS = ("csaf", "xattn_only", "gate_only", "add", "concat")


# --------------------------------------------------------------------------- #
# Phase 1: skeleton embedding (Eqs. 6-8)
# --------------------------------------------------------------------------- #
class SkeletonEmbedding(nn.Module):
    def __init__(self, in_channels, dim, T, V, use_motion=True):
        super().__init__()
        self.use_motion = use_motion
        self.proj = nn.Linear(in_channels * (2 if use_motion else 1), dim)
        self.pos_s = nn.Parameter(torch.zeros(1, 1, V, dim))   # P_s
        self.pos_t = nn.Parameter(torch.zeros(1, T, 1, dim))   # P_t
        nn.init.trunc_normal_(self.pos_s, std=0.02)
        nn.init.trunc_normal_(self.pos_t, std=0.02)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):  # x: (B, T, V, C)
        if self.use_motion:
            m = torch.zeros_like(x)
            m[:, 1:] = x[:, 1:] - x[:, :-1]                     # Eq. (6), M_1 = 0
            x = torch.cat([x, m], dim=-1)
        f = self.proj(x)                                        # Eq. (7)
        return self.norm(f + self.pos_s + self.pos_t)           # Eq. (8)


# --------------------------------------------------------------------------- #
# Phase 2: Temporal Mamba Stream (Eqs. 16-19)
# --------------------------------------------------------------------------- #
class BiMambaBlock(nn.Module):
    """One block: LayerNorm -> forward SSM (+ backward SSM on the flipped
    sequence) -> Linear merge (Eq. 18) -> residual. Each joint trajectory is an
    independent sequence."""

    def __init__(self, dim, d_state, d_conv, expand, scan="bi", share_bidir=False,
                 force_reference=False):
        super().__init__()
        self.scan = scan
        self.norm = nn.LayerNorm(dim)
        self.fwd = make_mamba(dim, d_state, d_conv, expand, force_reference)
        if scan == "bi":
            self.bwd = self.fwd if share_bidir else make_mamba(dim, d_state, d_conv, expand,
                                                               force_reference)
            self.merge = nn.Linear(2 * dim, dim)
        else:
            self.bwd, self.merge = None, None

    def forward(self, x):  # (B, T, V, D)
        B, T, V, D = x.shape
        h = self.norm(x).permute(0, 2, 1, 3).reshape(B * V, T, D)
        y = self.fwd(h)                                         # Eq. (16)
        if self.scan == "bi":
            yb = self.bwd(h.flip(1)).flip(1)                    # Eq. (17)
            y = self.merge(torch.cat([y, yb], dim=-1))          # Eq. (18)
        y = y.reshape(B, V, T, D).permute(0, 2, 1, 3)
        return x + y


class TemporalMambaStream(nn.Module):
    def __init__(self, dim, depth, d_state, d_conv, expand, scan, share_bidir,
                 force_reference=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            BiMambaBlock(dim, d_state, d_conv, expand, scan, share_bidir, force_reference)
            for _ in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# --------------------------------------------------------------------------- #
# Phase 3: Spatial GCN Stream with AGTL (Eqs. 20-23)
# --------------------------------------------------------------------------- #
class AGTL(nn.Module):
    """A_combined = A_phys + A_data + A_adapt (Eq. 20); terms switched by `topology`.

    A_adapt is computed once per sample from Pool_T(input) and shared by all
    GCN layers and channels, as stated in the revised Sec. 3.5.
    """

    def __init__(self, dim, V, topology="full", dk=None):
        super().__init__()
        assert topology in TOPOLOGIES, topology
        self.use_data = topology in ("phys+data", "full")
        self.use_adapt = topology in ("phys+adapt", "full")
        self.register_buffer("A_phys", torch.from_numpy(
            normalized_adjacency(physical_adjacency(V))))
        if self.use_data:
            # 2s-AGCN initialises its global graph near zero so training starts from A_phys.
            self.A_data = nn.Parameter(torch.full((V, V), 1e-6))
        if self.use_adapt:
            self.dk = dk or dim // 4
            self.phi_q = nn.Linear(dim, self.dk)
            self.phi_k = nn.Linear(dim, self.dk)

    def forward(self, x):  # x: (B, T, V, D) -> (B, V, V)
        B = x.shape[0]
        A = self.A_phys.unsqueeze(0).expand(B, -1, -1)
        if self.use_data:
            A = A + self.A_data
        if self.use_adapt:
            pooled = x.mean(dim=1)                              # Pool_T, (B, V, D)
            q, k = self.phi_q(pooled), self.phi_k(pooled)       # Eq. (21)
            A = A + torch.softmax(q @ k.transpose(1, 2) / math.sqrt(self.dk), dim=-1)  # Eq. (22)
        return A


class GCNLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, A):  # x: (B, T, V, D), A: (B, V, V)
        y = torch.einsum("buv,btvd->btud", A, self.W(x))       # Eq. (23)
        return x + F.gelu(self.norm(y))


class SpatialGCNStream(nn.Module):
    def __init__(self, dim, depth, V, topology="full", dk=None):
        super().__init__()
        self.agtl = AGTL(dim, V, topology, dk)
        self.layers = nn.ModuleList([GCNLayer(dim) for _ in range(depth)])

    def forward(self, x):
        A = self.agtl(x)
        for layer in self.layers:
            x = layer(x, A)
        return x


# --------------------------------------------------------------------------- #
# Phase 4: fusion (Eqs. 24-27) and its ablation variants (Table 16)
# --------------------------------------------------------------------------- #
class Fusion(nn.Module):
    def __init__(self, dim, mode="csaf", heads=8, dropout=0.1, gate_hidden=None):
        super().__init__()
        assert mode in FUSIONS, mode
        self.mode = mode
        if mode in ("csaf", "xattn_only"):
            self.norm_q = nn.LayerNorm(dim)
            self.norm_kv = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        if mode in ("csaf", "gate_only"):
            h = gate_hidden or dim // 4
            self.gate = nn.Sequential(nn.Linear(2 * dim, h), nn.GELU(), nn.Linear(h, 1))
        if mode == "concat":
            self.proj = nn.Linear(2 * dim, dim)

    def cross_attention(self, h_temp, h_spat):
        """Per-frame attention over the V joints (Eqs. 24-25); T and V are never flattened."""
        B, T, V, D = h_temp.shape
        q = self.norm_q(h_temp).reshape(B * T, V, D)
        kv = self.norm_kv(h_spat).reshape(B * T, V, D)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return out.reshape(B, T, V, D)

    def gate_alpha(self, h_temp, h_spat):
        """alpha of Eq. (26). Pool_{T,V} = global average pooling over time and joints."""
        z = torch.cat([h_temp.mean(dim=(1, 2)), h_spat.mean(dim=(1, 2))], dim=-1)
        return torch.sigmoid(self.gate(z)).view(-1, 1, 1, 1)

    def forward(self, h_temp, h_spat):
        alpha = None
        if self.mode == "csaf":
            alpha = self.gate_alpha(h_temp, h_spat)
            out = alpha * h_temp + (1 - alpha) * self.cross_attention(h_temp, h_spat)  # Eq. (27)
        elif self.mode == "xattn_only":
            out = 0.5 * h_temp + 0.5 * self.cross_attention(h_temp, h_spat)
        elif self.mode == "gate_only":
            alpha = self.gate_alpha(h_temp, h_spat)
            out = alpha * h_temp + (1 - alpha) * h_spat
        elif self.mode == "add":
            out = h_temp + h_spat
        else:  # concat
            out = self.proj(torch.cat([h_temp, h_spat], dim=-1))
        return out, alpha


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class DSMambaGCN(nn.Module):
    def __init__(self, num_classes, T, V=25, in_channels=3, dim=256, depth=4,
                 d_state=64, d_conv=4, expand=2, share_bidir=False, heads=8,
                 dropout=0.1, dk=None, gate_hidden=None, topology="full",
                 coupling="parallel", fusion="csaf", scan="bi", use_motion=True,
                 force_reference_mamba=False):
        super().__init__()
        assert coupling in COUPLINGS, coupling
        self.coupling = coupling
        self.embed = SkeletonEmbedding(in_channels, dim, T, V, use_motion)
        need_t = coupling != "spatial_only"
        need_s = coupling != "temporal_only"
        self.tms = TemporalMambaStream(dim, depth, d_state, d_conv, expand, scan, share_bidir,
                                       force_reference_mamba) if need_t else None
        self.sgs = SpatialGCNStream(dim, depth, V, topology, dk) if need_s else None
        self.fusion = Fusion(dim, fusion, heads, dropout, gate_hidden) \
            if coupling == "parallel" else None
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x, return_extra=False):
        """x: (B, C, T, V, M). Bodies are folded into the batch and their pooled
        features averaged before the classifier."""
        B, C, T, V, M = x.shape
        x = x.permute(0, 4, 2, 3, 1).reshape(B * M, T, V, C)
        f = self.embed(x)
        alpha = None
        if self.coupling == "parallel":
            h, alpha = self.fusion(self.tms(f), self.sgs(f))
        elif self.coupling == "serial_gcn_mamba":
            h = self.tms(self.sgs(f))
        elif self.coupling == "serial_mamba_gcn":
            h = self.sgs(self.tms(f))
        elif self.coupling == "spatial_only":
            h = self.sgs(f)
        else:
            h = self.tms(f)
        z = h.mean(dim=(1, 2)).view(B, M, -1).mean(dim=1)      # Eq. (28) + body average
        logits = self.head(z)                                   # Eq. (29), softmax in the loss
        if not return_extra:
            return logits
        if alpha is not None:
            alpha = alpha.view(B, M).mean(dim=1)
        return logits, {"alpha": alpha, "z": z}


def build_model(cfg_model: dict, num_classes: int, T: int) -> DSMambaGCN:
    return DSMambaGCN(num_classes=num_classes, T=T, **cfg_model)


def param_breakdown(model: nn.Module) -> dict:
    """Trainable parameters per top-level module, for the revised Table 6."""
    out = {}
    for name, child in model.named_children():
        if child is None:
            continue
        out[name] = sum(p.numel() for p in child.parameters() if p.requires_grad)
    out["total"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return out
