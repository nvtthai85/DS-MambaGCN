"""Tests for tools/ntu_preprocess.py and the compact NTU loader (NumPy only)."""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dsmamba.data_utils import CompactSkeletons, ntu_split_mask, uniform_resample  # noqa: E402


def write_skel(path, bodies_spec, n_frames):
    lines = [str(n_frames)]
    for f in range(n_frames):
        present = [b for b in bodies_spec if f in b[1]]
        lines.append(str(len(present)))
        for bid, _, kind in present:
            lines.append(f"{bid} 0 1 2 1 2 0 0.1 0.1 2")
            lines.append("25")
            for j in range(25):
                if kind == "person":
                    x, y, z = 0.1 * np.sin(f / 5 + j) + 0.01 * (j % 3), 0.05 * j - 0.6, 3.0
                elif kind == "still":
                    x, y, z = 0.0, 0.05 * j - 0.6, 3.0
                else:  # ghost: wide and flat
                    x, y, z = -1.0 + 0.08 * j, 0.001 * j, 3.0
                lines.append(f"{x:.5f} {y:.5f} {z:.5f} 1 1 1 1 1 1 1 1 2")
    open(path, "w").write("\n".join(lines) + "\n")


def test_preprocess_and_loader():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        raw = os.path.join(d, "raw")
        os.makedirs(raw)
        for s in (1, 2):
            for c in (1, 2, 3):
                for p in (1, 3, 8):
                    for a in (1, 50, 61):
                        n = f"S{s:03d}C{c:03d}P{p:03d}R001A{a:03d}"
                        nf = int(rng.integers(30, 90))
                        spec = [("b1", set(range(nf)), "person"),
                                ("b2", set(range(5, nf)), "still" if a != 50 else "person"),
                                ("b3", set(range(nf)), "ghost"),
                                ("b4", set(range(3)), "person")]
                        write_skel(os.path.join(raw, n + ".skeleton"), spec, nf)
        write_skel(os.path.join(raw, "S001C001P001R002A002.skeleton"), [], 10)
        # a lone actor who is wide (lying on the floor) must be kept, not dropped as a ghost
        write_skel(os.path.join(raw, "S001C001P001R003A043.skeleton"), [("b9", set(range(40)), "ghost")], 40)
        # a short-lived lone body must be kept too
        write_skel(os.path.join(raw, "S001C001P001R004A008.skeleton"), [("b8", set(range(6)), "person")], 6)
        open(os.path.join(d, "missing.txt"), "w").write("S002C003P008R001A001\n")
        out = os.path.join(d, "compact")
        r = subprocess.run([sys.executable, "tools/ntu_preprocess.py", "--raw", raw, "--out", out,
                            "--missing", os.path.join(d, "missing.txt"), "--max-action", "60"],
                           cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        rep = json.load(open(os.path.join(out, "report.json")))
        assert rep["skipped_counts"] == {"official_missing": 1, "no_valid_body": 1,
                                         "unreadable": 0, "action_out_of_range": 18}
        cs = CompactSkeletons(out)
        assert len(cs) == 37 and set(cs.labels.tolist()) == {0, 7, 42, 49}
        assert rep["samples_with_filtered_bodies"] == 35
        m = cs.meta
        assert set(m["subject"][ntu_split_mask(m, "xsub")]) == {1, 8}
        assert set(m["camera"][ntu_split_mask(m, "xview")]) == {2, 3}
        assert set(m["setup"][ntu_split_mask(m, "xset")]) == {2}
        for k in range(len(cs)):
            if cs.meta["label"][k] in (7, 42):
                continue
            seq = cs[k]
            assert seq.shape[0] == 3 and seq.shape[2:] == (25, 2)
            # the ghost (x spread 1.9 m) is never kept
            assert np.ptp(seq[0, :, :, 0], axis=1).max() < 0.5
            assert np.ptp(seq[0, :, :, 1], axis=1).max() < 0.5
            # the 3-frame body is never kept
            assert seq.shape[1] >= 25
        k = list(m["label"]).index(49)
        assert np.abs(cs[k][..., 1]).sum() > 0, "interaction sample keeps the second person"
        assert uniform_resample(cs[k], cs[k].shape[1], 300).shape == (3, 300, 25, 2)
        sub = cs.subset([0, 2])
        assert len(sub) == 2 and np.array_equal(sub[1], cs[2])


if __name__ == "__main__":
    test_preprocess_and_loader()
    print("ok")
