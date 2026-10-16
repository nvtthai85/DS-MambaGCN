"""Share of the strongest learned A_data edges that are NOT physical bones (Sec. 3.5).

    python tools/adata_edges.py runs/ntu60_xsub_full/seed*/best.pt --k 24

Definition used in the paper: A_data is symmetrised as (|A| + |A^T|) / 2, the
diagonal is excluded, the K joint pairs (of 300 unordered pairs for V = 25) with
the largest weight are taken, and the script reports the percentage of those
pairs that are not bones of the skeleton. K = 24 equals the number of bones, so a
graph that only rediscovered the skeleton would score 0 %. The value is averaged
over the checkpoints given (one per seed) and printed with its spread; a
permutation baseline (expected share for K random pairs) is printed for scale.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsmamba.graph import BONES  # noqa: E402


def non_physical_share(A, k):
    V = A.shape[0]
    S = (np.abs(A) + np.abs(A.T)) / 2
    iu = np.triu_indices(V, 1)
    w = S[iu]
    top = np.argsort(-w)[:k]
    bones = {tuple(sorted(b)) for b in BONES}
    pairs = [(int(iu[0][t]), int(iu[1][t])) for t in top]
    return 100.0 * sum(p not in bones for p in pairs) / k, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="+", help="best.pt / last.pt (state_dict) files")
    ap.add_argument("--k", type=int, default=24)
    a = ap.parse_args()
    vals = []
    for p in a.ckpt:
        sd = torch.load(p, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        key = next((n for n in sd if n.endswith("A_data")), None)
        if key is None:
            print(f"{p}: no A_data (topology without the data-driven term)")
            continue
        share, pairs = non_physical_share(sd[key].float().numpy(), a.k)
        vals.append(share)
        print(f"{p}: {share:.1f}% of the top-{a.k} edges are non-physical; "
              f"strongest 5 pairs (0-based joints): {pairs[:5]}")
    if vals:
        V, nb = 25, len(BONES)
        chance = 100.0 * (1 - nb / (V * (V - 1) / 2))
        print(f"mean {np.mean(vals):.1f}% (sd {np.std(vals, ddof=1) if len(vals) > 1 else 0:.1f}, "
              f"{len(vals)} checkpoints); random pairs would give {chance:.1f}%")


if __name__ == "__main__":
    main()
