"""PyTorch checks for every ablation switch (run on a machine with PyTorch).

    python -m pytest -q tests/test_model_torch.py

Uses the pure-PyTorch Mamba reference so it also runs on CPU. It checks that
every configuration in configs/ builds and trains one step, that alpha is
returned only by gated fusions, that parallel streams really see the same
input, and that the parameter count equals tools/audit_complexity.py.
"""
import glob
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dsmamba.model import DSMambaGCN, param_breakdown  # noqa: E402
from train import load_config  # noqa: E402

TINY = dict(dim=32, depth=2, d_state=8, heads=4, force_reference_mamba=True)


def tiny_model(**kw):
    return DSMambaGCN(num_classes=60, T=16, **{**TINY, **kw})


@pytest.mark.parametrize("cfg_path", sorted(p for p in glob.glob(os.path.join(ROOT, "configs/*/*.yaml"))
                                            if not p.endswith("base.yaml")))
def test_every_config_trains_one_step(cfg_path):
    m = dict(load_config(cfg_path)["model"])
    m.update(TINY)
    model = DSMambaGCN(num_classes=10, T=16, **m)
    x = torch.randn(2, 3, 16, 25, 2)
    logits, extra = model(x, return_extra=True)
    assert logits.shape == (2, 10) and extra["z"].shape == (2, 32)
    gated = m["coupling"] == "parallel" and m["fusion"] in ("csaf", "gate_only")
    assert (extra["alpha"] is not None) == gated
    if gated:
        assert extra["alpha"].shape == (2,) and ((extra["alpha"] >= 0) & (extra["alpha"] <= 1)).all()
    logits.logsumexp(-1).mean().backward()
    assert all(p.grad is not None for n, p in model.named_parameters()
               if p.requires_grad and "pos_t" not in n), "unused parameters in " + cfg_path


def test_topology_terms():
    for topo, has_data, has_adapt in (("phys", False, False), ("phys+data", True, False),
                                      ("phys+adapt", False, True), ("full", True, True)):
        agtl = tiny_model(topology=topo).sgs.agtl
        assert hasattr(agtl, "A_data") == has_data and hasattr(agtl, "phi_q") == has_adapt


def test_serial_wiring_order():
    calls = []
    model = tiny_model(coupling="serial_mamba_gcn")
    model.tms.register_forward_hook(lambda *_: calls.append("tms"))
    model.sgs.register_forward_hook(lambda *_: calls.append("sgs"))
    model(torch.randn(1, 3, 16, 25, 1))
    assert calls == ["tms", "sgs"]
    calls.clear()
    model = tiny_model(coupling="serial_gcn_mamba")
    model.tms.register_forward_hook(lambda *_: calls.append("tms"))
    model.sgs.register_forward_hook(lambda *_: calls.append("sgs"))
    model(torch.randn(1, 3, 16, 25, 1))
    assert calls == ["sgs", "tms"]


def test_parallel_streams_share_input():
    model = tiny_model()
    seen = {}
    model.tms.register_forward_pre_hook(lambda m, a: seen.__setitem__("t", a[0]))
    model.sgs.register_forward_pre_hook(lambda m, a: seen.__setitem__("s", a[0]))
    model(torch.randn(1, 3, 16, 25, 1))
    assert seen["t"] is seen["s"], "both streams must receive the same F'"


def test_backward_scan_sees_future():
    """With a bidirectional scan, changing the last frame must change the first output frame."""
    torch.manual_seed(0)
    for scan, should_change in (("bi", True), ("fwd", False)):
        tms = tiny_model(scan=scan, coupling="temporal_only").tms.eval()
        x = torch.randn(1, 16, 25, 32)
        y1 = tms(x)
        x2 = x.clone()
        # random perturbation: a constant shift would be removed by the block's LayerNorm
        x2[:, -1] += torch.randn_like(x2[:, -1])
        y2 = tms(x2)
        changed = not torch.allclose(y1[:, 0], y2[:, 0], atol=1e-6)
        assert changed == should_change, scan


@pytest.mark.parametrize("expand,share", [(1, False), (2, False), (1, True)])
def test_param_count_matches_audit(expand, share):
    model = DSMambaGCN(num_classes=60, T=300, dim=256, depth=4, d_state=64, expand=expand,
                       share_bidir=share, force_reference_mamba=True)
    total = param_breakdown(model)["total"]
    args = [sys.executable, "tools/audit_complexity.py", "--expand", str(expand)]
    if share:
        args.append("--share-bidir")
    out = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True).stdout
    audit = float([ln for ln in out.splitlines() if ln.startswith("TOTAL")][0].split()[1])
    assert abs(total / 1e6 - audit) < 0.0015, (total, audit)
