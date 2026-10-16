"""Step 3 of the UCF Sports pipeline: write fixed train/val/test lists.

Three modes, so the paper can state exactly which protocol was used:

  # (a0) the published split of Lan et al. (ICCV 2011), 103 train / 47 test (used in the paper)
  python tools/make_ucf_split.py --data data/ucf_sports/ucf_sports_mediapipe25.npz \
      --lan-split splits/ucf_sports_lan2011.txt --val-frac 0.15 --out data/ucf_sports/splits

  # (a) an existing published test list (one video id per line, e.g. Diving-Side/001)
  python tools/make_ucf_split.py --data data/ucf_sports/ucf_sports_mediapipe25.npz \
      --test-list lan2011_test.txt --val-frac 0.15 --out data/ucf_sports/splits

  # (b) a stratified random hold-out, fixed by --seed
  python tools/make_ucf_split.py --data ... --mode stratified --test-frac 0.3 --seed 0 --out ...

  # (c) k-fold cross-validation (recommended for 150 videos): writes fold0..fold{k-1}/
  python tools/make_ucf_split.py --data ... --mode kfold --k 5 --seed 0 --out ...

In every mode the validation set is drawn from the training part only,
stratified by class, and the three lists are checked to be disjoint.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsmamba.data_utils import read_name_list, stratified_holdout  # noqa: E402


def write_split(out_dir, names, labels, tr, va, te, meta):
    os.makedirs(out_dir, exist_ok=True)
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    for fn, idx in (("train.txt", tr), ("val.txt", va), ("test.txt", te)):
        with open(os.path.join(out_dir, fn), "w") as fh:
            fh.write("\n".join(names[i] for i in sorted(idx)) + "\n")
    counts = {s: np.bincount(labels[list(i)], minlength=10).tolist()
              for s, i in (("train", tr), ("val", va), ("test", te))}
    meta = {**meta, "sizes": {"train": len(tr), "val": len(va), "test": len(te)},
            "per_class": counts,
            "test_accuracy_granularity_pp": round(100.0 / max(1, len(te)), 3)}
    json.dump(meta, open(os.path.join(out_dir, "split_meta.json"), "w"), indent=2)
    print(out_dir, meta["sizes"], "1 test video =", meta["test_accuracy_granularity_pp"], "pp")


def carve_val(tr, labels, frac, seed):
    tr = np.array(sorted(tr))
    sub_tr, sub_va = stratified_holdout(labels[tr], frac, seed)
    return tr[sub_tr].tolist(), tr[sub_va].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["list", "stratified", "kfold"], default="list")
    ap.add_argument("--test-list", default=None)
    ap.add_argument("--lan-split", default=None,
                    help="splits/ucf_sports_lan2011.txt (index-based split of Lan et al. 2011)")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d = np.load(a.data, allow_pickle=True)
    names, labels = [str(n) for n in d["names"]], d["y"]
    meta = {"mode": a.mode, "seed": a.seed, "val_frac": a.val_frac, "data": a.data}

    if a.lan_split:
        idx = {}
        for ln in open(a.lan_split):
            if ":" in ln and not ln.startswith("#"):
                k, v = ln.split(":", 1)
                idx[k.strip()] = [int(x) for x in v.split()]
        order = sorted(range(len(names)), key=lambda i: names[i])   # alphabetical: class folder, number
        if len(order) != 150:
            raise SystemExit(f"the Lan split indexes 150 videos, the data has {len(names)}")
        te = sorted(order[j - 1] for j in idx["test"])
        tr_all = sorted(order[j - 1] for j in idx["train"])
        assert len(te) == 47 and len(tr_all) == 103 and not set(te) & set(tr_all)
        tr, va = carve_val(tr_all, labels, a.val_frac, a.seed)
        print("check: video 1 =", names[order[0]], "| video 150 =", names[order[-1]])
        write_split(a.out, names, labels, tr, va, te,
                    {**meta, "mode": "lan2011", "lan_split": a.lan_split,
                     "first_test_videos": [names[i] for i in te[:4]]})
        return
    if a.mode == "list":
        if not a.test_list:
            raise SystemExit("--test-list is required in mode 'list'")
        test_names = set(read_name_list(a.test_list))
        unknown = test_names - set(names)
        if unknown:
            raise SystemExit(f"{len(unknown)} test ids not in the data, e.g. {sorted(unknown)[:3]}")
        te = [i for i, n in enumerate(names) if n in test_names]
        tr_all = [i for i in range(len(names)) if i not in set(te)]
        tr, va = carve_val(tr_all, labels, a.val_frac, a.seed)
        write_split(a.out, names, labels, tr, va, te, {**meta, "test_list": a.test_list})
    elif a.mode == "stratified":
        tr_all, te = stratified_holdout(labels, a.test_frac, a.seed)
        tr, va = carve_val(tr_all, labels, a.val_frac, a.seed + 1)
        write_split(a.out, names, labels, tr, va, te.tolist(), {**meta, "test_frac": a.test_frac})
    else:
        rng = np.random.default_rng(a.seed)
        fold_of = np.empty(len(names), int)
        for c in np.unique(labels):                      # stratified fold assignment
            idx = np.flatnonzero(labels == c)
            rng.shuffle(idx)
            fold_of[idx] = np.arange(idx.size) % a.k
        for f in range(a.k):
            te = np.flatnonzero(fold_of == f).tolist()
            tr_all = np.flatnonzero(fold_of != f).tolist()
            tr, va = carve_val(tr_all, labels, a.val_frac, a.seed + 1 + f)
            write_split(os.path.join(a.out, f"fold{f}"), names, labels, tr, va, te,
                        {**meta, "fold": f, "k": a.k})


if __name__ == "__main__":
    main()
