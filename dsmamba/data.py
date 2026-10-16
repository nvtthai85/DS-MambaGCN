"""PyTorch datasets for NTU RGB+D 60/120 and UCF Sports (MediaPipe skeletons)."""
import numpy as np
import torch
from torch.utils.data import Dataset

from .data_utils import (CompactSkeletons, augment, center_on_spine, load_ntu_npz,
                         ntu_split_mask, read_name_list, stratified_holdout,
                         uniform_resample, valid_length)


class SkeletonDataset(Dataset):
    def __init__(self, x, y, T, train, aug_cfg=None, center=True, seed=0):
        self.x, self.y, self.T = x, np.asarray(y), T
        self.train, self.center = train, center
        self.aug_cfg = aug_cfg or {}
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        seq = np.asarray(self.x[i], dtype=np.float32)
        L = valid_length(seq)
        seq = uniform_resample(seq, L, self.T)
        if self.center:
            seq = center_on_spine(seq)
        if self.train and self.aug_cfg.get("enabled", True):
            # Reproducible per (seed, epoch, sample), independent of worker scheduling.
            rng = np.random.default_rng((self.seed, self.epoch, int(i)))
            seq = augment(seq, rng, self.aug_cfg.get("rot_deg", 15.0),
                          self.aug_cfg.get("shift", 5), self.aug_cfg.get("noise_std", 0.01))
        return torch.from_numpy(seq.astype(np.float32)), int(self.y[i]), int(i)


def build_datasets(dcfg: dict, seed: int):
    """Returns (train_ds, val_ds or None, test_ds, num_classes, info).

    dcfg["validation"] controls model selection (see README, 'Validation policy'):
      {"policy": "holdout", "fraction": 0.05, "split_seed": 0}  -> val carved from train
      {"policy": "final_epoch"}                                   -> no val, last epoch
      {"policy": "files", ...}                                     -> UCF Sports split files
    """
    T, aug = dcfg["T"], dcfg.get("augment", {})
    vcfg = dcfg.get("validation", {"policy": "final_epoch"})
    info = {}
    if dcfg["name"] in ("ntu60", "ntu120") and dcfg.get("format", "npz") == "compact":
        # Raw-skeleton pipeline (tools/ntu_preprocess.py); splits from official id lists.
        full = CompactSkeletons(dcfg["path"])
        num_classes = 60 if dcfg["name"] == "ntu60" else 120
        keep = full.meta["label"] < num_classes
        if dcfg["name"] == "ntu60":            # NTU60 = setups 1-17 only, even in an NTU120 file
            keep &= full.meta["setup"] <= 17
        train_mask = ntu_split_mask(full.meta, dcfg["benchmark"], dcfg["name"])
        tr_all = np.flatnonzero(keep & train_mask)
        te = np.flatnonzero(keep & ~train_mask)
        val_ds = None
        if vcfg["policy"] == "holdout":
            sub_tr, sub_va = stratified_holdout(full.meta["label"][tr_all], vcfg["fraction"],
                                                vcfg.get("split_seed", 0))
            va = tr_all[sub_va]
            tr_all = tr_all[sub_tr]
            info["val_names"] = full.meta["name"][va].tolist()
            v = full.subset(va)
            val_ds = SkeletonDataset(v, v.labels, T, train=False)
        lim = dcfg.get("max_train_samples")
        if lim:  # smoke tests only
            tr_all = np.random.default_rng(0).permutation(tr_all)[:lim]
            te = te[: max(1, lim // 2)]
        tr, ts = full.subset(tr_all), full.subset(te)
        info.update(n_train=len(tr), n_val=0 if val_ds is None else len(val_ds), n_test=len(ts),
                    benchmark=dcfg["benchmark"])
        return (SkeletonDataset(tr, tr.labels, T, True, aug, seed=seed), val_ds,
                SkeletonDataset(ts, ts.labels, T, train=False), num_classes, info)

    if dcfg["name"] in ("ntu60", "ntu120"):
        d = load_ntu_npz(dcfg["path"])
        xtr, ytr = d["x_train"], d["y_train"]
        num_classes = 60 if dcfg["name"] == "ntu60" else 120
        val_ds = None
        if vcfg["policy"] == "holdout":
            tr_idx, va_idx = stratified_holdout(ytr, vcfg["fraction"], vcfg.get("split_seed", 0))
            info["val_indices"] = va_idx.tolist()
            val_ds = SkeletonDataset(xtr[va_idx], ytr[va_idx], T, train=False)
            xtr, ytr = xtr[tr_idx], ytr[tr_idx]
        train_ds = SkeletonDataset(xtr, ytr, T, True, aug, seed=seed)
        test_ds = SkeletonDataset(d["x_test"], d["y_test"], T, train=False)
        return train_ds, val_ds, test_ds, num_classes, info

    if dcfg["name"] == "ucf_sports":
        d = np.load(dcfg["path"], allow_pickle=True)          # from tools/ucf_preprocess.py
        names = list(d["names"])
        pos = {n: k for k, n in enumerate(names)}
        sel = {s: [pos[n] for n in read_name_list(dcfg["splits"][s])]
               for s in ("train", "val", "test")}
        assert not set(sel["test"]) & (set(sel["train"]) | set(sel["val"])), "test overlaps train/val"
        info["split_sizes"] = {k: len(v) for k, v in sel.items()}
        x, y = d["x"], d["y"]
        mk = lambda s, tr: SkeletonDataset(x[sel[s]], y[sel[s]], T, tr, aug, center=False, seed=seed)
        return mk("train", True), mk("val", False), mk("test", False), int(d["num_classes"]), info

    raise ValueError(dcfg["name"])
