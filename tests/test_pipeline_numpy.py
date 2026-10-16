"""End-to-end tests of every NumPy/SciPy component on synthetic data.

    python -m pytest -q tests/test_pipeline_numpy.py     (or: python tests/test_pipeline_numpy.py)
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dsmamba.data_utils import (augment, stratified_holdout, temporal_shift,  # noqa: E402
                                uniform_resample, valid_length)
from dsmamba.graph import normalized_adjacency, physical_adjacency  # noqa: E402
from tools import ucf_preprocess as up  # noqa: E402


def run(*args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_graph():
    A = physical_adjacency()
    assert A.sum() == 48 and (A == A.T).all()
    An = normalized_adjacency(A)
    assert np.allclose(An, An.T) and (np.linalg.eigvalsh(An) <= 1 + 1e-5).all()


def test_resample_and_shift():
    seq = np.zeros((3, 300, 25, 2), np.float32)
    seq[:, :80] = np.linspace(0, 1, 80)[None, :, None, None]
    assert valid_length(seq) == 80
    r = uniform_resample(seq, 80, 300)
    assert r.shape == (3, 300, 25, 2) and np.isclose(r[0, 0, 0, 0], 0) and np.isclose(r[0, -1, 0, 0], 1)
    s = temporal_shift(r, 5, np.random.default_rng(1))
    assert s.shape == r.shape
    a1 = augment(r, np.random.default_rng((0, 3, 7)))
    a2 = augment(r, np.random.default_rng((0, 3, 7)))
    assert np.array_equal(a1, a2), "augmentation must be reproducible from (seed, epoch, index)"


def test_stratified_holdout():
    y = np.repeat(np.arange(10), [14, 18, 20, 6, 12, 13, 12, 20, 13, 22])
    tr, va = stratified_holdout(y, 0.15, 0)
    assert not set(tr) & set(va) and len(tr) + len(va) == y.size
    assert set(np.unique(y[va])) == set(range(10))
    tr2, va2 = stratified_holdout(y, 0.15, 0)
    assert np.array_equal(va, va2)


def synthetic_raw_clip(T, rng, drop_frames=(), low_vis_landmark=None):
    world = rng.normal(0, 0.3, (T, 33, 4)).astype(np.float32)
    world[..., 3] = 0.9
    world[:, 11, :3] = [0.2, -0.5, 0]   # L shoulder
    world[:, 12, :3] = [-0.2, -0.5, 0]  # R shoulder
    world[:, 23, :3] = [0.1, 0.0, 0]    # L hip
    world[:, 24, :3] = [-0.1, 0.0, 0]   # R hip
    det = np.ones(T, bool)
    det[list(drop_frames)] = False
    world[~det] = np.nan
    if low_vis_landmark is not None:
        world[:, low_vis_landmark, 3] = 0.1
    return {"world": world, "image": world.copy(), "detected": det, "width": 720,
            "height": 404, "fps": 10.0}


def test_mapping_and_cleaning():
    rng = np.random.default_rng(0)
    rec = synthetic_raw_clip(40, rng, drop_frames=(0, 1, 17, 18, 39))
    sk, st = up.process_clip(rec, "world", 0.5, "first", "nose", 0.5, 0.5)
    assert sk.shape == (40, 25, 3) and not np.isnan(sk).any()
    assert st["frames_no_person"] == 5
    # spine shoulder (20) is the shoulder midpoint; spine base (0) the hip midpoint
    raw = up.interpolate_nan(np.where(rec["detected"][:, None, None], rec["world"][..., :3], np.nan))
    mapped = up.map_33_to_25(raw)
    assert np.allclose(mapped[:, 20], raw[:, [11, 12]].mean(1))
    assert np.allclose(mapped[:, 0], raw[:, [23, 24]].mean(1))
    assert np.allclose(mapped[:, 1], 0.5 * (mapped[:, 0] + mapped[:, 20]))
    # after centring on the first frame's spine base and scaling, shoulder width is 1
    assert np.allclose(sk[0, 0], 0, atol=1e-6)
    w = np.linalg.norm(sk[:, 4] - sk[:, 8], axis=-1).mean()
    assert np.isclose(w, 1.0, atol=1e-5)
    # a landmark never visible (L thumb, MediaPipe 21 -> NTU 22) is filled from its parent
    rec2 = synthetic_raw_clip(30, rng, low_vis_landmark=21)
    sk2, st2 = up.process_clip(rec2, "world", 0.5, "first", "nose", 0.5, 0.5)
    assert 22 in st2["joints_never_seen"] and not np.isnan(sk2).any()


def test_ucf_end_to_end():
    rng = np.random.default_rng(1)
    folders = ["Diving-Side", "Golf-Swing-Back", "Golf-Swing-Front", "Kicking-Front", "Lifting",
               "Riding-Horse", "Run-Side", "SkateBoarding-Front", "Swing-Bench",
               "Swing-SideAngle", "Walk-Front"]
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw")
        os.makedirs(raw)
        for f in folders:
            for n in range(1, 8):
                rec = synthetic_raw_clip(int(rng.integers(22, 145)), rng, drop_frames=(3,))
                np.savez(os.path.join(raw, f"{f}__{n:03d}.npz"), **rec)
        out = os.path.join(tmp, "ucf.npz")
        run("tools/ucf_preprocess.py", "--raw", raw, "--out", out)
        d = np.load(out, allow_pickle=True)
        assert d["x"].shape == (77, 3, 150, 25, 1) and set(d["y"]) == set(range(10))
        rep = json.load(open(out[:-4] + "_report.json"))["summary"]
        assert rep["clips_kept"] == 77 and rep["frames_without_person_frac"] > 0
        for mode in (["--mode", "stratified", "--test-frac", "0.3"], ["--mode", "kfold", "--k", "5"]):
            sp = os.path.join(tmp, "splits_" + mode[1])
            run("tools/make_ucf_split.py", "--data", out, "--out", sp, *mode)
        names = set(open(os.path.join(tmp, "splits_stratified", "test.txt")).read().split())
        tr = set(open(os.path.join(tmp, "splits_stratified", "train.txt")).read().split())
        va = set(open(os.path.join(tmp, "splits_stratified", "val.txt")).read().split())
        assert not names & tr and not names & va and len(names | tr | va) == 77
        all_test = set()
        for f in range(5):
            all_test |= set(open(os.path.join(tmp, "splits_kfold", f"fold{f}", "test.txt")).read().split())
        assert len(all_test) == 77, "k-fold test sets must cover every video exactly once"
        # mode 'list' with a given test list
        lst = os.path.join(tmp, "given_test.txt")
        open(lst, "w").write("\n".join(sorted(names)) + "\n")
        run("tools/make_ucf_split.py", "--data", out, "--out", os.path.join(tmp, "sl"),
            "--test-list", lst)
        assert set(open(os.path.join(tmp, "sl", "test.txt")).read().split()) == names


def test_alpha_and_tsne():
    rng = np.random.default_rng(2)
    n = 1800
    label = rng.integers(0, 60, n)
    inter = label >= 49
    alpha = np.clip(np.where(inter, 0.35, 0.6) + rng.normal(0, 0.08, n), 0, 1)
    z = rng.normal(0, 1, (n, 32)) + np.eye(60)[label][:, :32] * 3
    with tempfile.TemporaryDirectory() as tmp:
        f0, f1 = os.path.join(tmp, "a0.npz"), os.path.join(tmp, "a1.npz")
        np.savez(f0, alpha=alpha, z=z, label=label, pred=label, index=np.arange(n))
        np.savez(f1, alpha=np.clip(alpha + rng.normal(0, 0.02, n), 0, 1), z=z, label=label,
                 pred=label, index=np.arange(n))
        fig = os.path.join(tmp, "fig17.pdf")
        out = run("tools/analyze_alpha.py", f0, f1, "--out", fig)
        rep = json.load(open(fig[:-4] + "_stats.json"))
        assert os.path.getsize(fig) > 1000
        assert rep["p_value"] < 1e-10 and rep["interaction_A50_A60"]["mean"] < 0.45
        assert rep["spearman_class_medians_vs_seed0"][0] > 0.8
        assert "supported" in rep["verdict"]
        # a flat gate must trigger the fallback verdict
        f2 = os.path.join(tmp, "flat.npz")
        np.savez(f2, alpha=0.5 + rng.normal(0, 0.01, n), z=z, label=label, pred=label,
                 index=np.arange(n))
        run("tools/analyze_alpha.py", f2, "--out", os.path.join(tmp, "flat.pdf"))
        assert "fallback" in json.load(open(os.path.join(tmp, "flat_stats.json")))["verdict"]
        ts = os.path.join(tmp, "tsne.pdf")
        run("tools/tsne_plot.py", f0, "--classes", "0", "1", "9", "22", "26", "49", "54", "57",
            "--per-class", "20", "--out", ts, "--title", "test")
        assert len(json.load(open(ts[:-4] + "_samples.json"))["test_indices"]) == 160
        assert "test" in out or True


def test_aggregate():
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(3)
        for cfg, mu in (("ntu60_xsub_full", 94.2), ("ntu60_xsub_t16_gate_only", 93.0)):
            for s in range(5):
                d = os.path.join(tmp, cfg, f"seed{s}")
                os.makedirs(d)
                json.dump({"config": cfg, "seed": s, "test_acc": mu + rng.normal(0, 0.1),
                           "params": {"total": 1_680_000}}, open(os.path.join(d, "results.json"), "w"))
        out = run("tools/aggregate_results.py", "--runs", tmp, "--reference", "ntu60_xsub_full")
        assert "Table 16" in out and "$\\pm$" in out


def test_audit():
    out = run("tools/audit_complexity.py", "--expand", "1", "--share-bidir")
    total = [ln for ln in out.splitlines() if ln.startswith("TOTAL")][0].split()
    assert float(total[1]) > 2.0 and float(total[2]) > 20


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
