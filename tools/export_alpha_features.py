"""Run a trained checkpoint on the test split once and save, per sample, the
fusion gate alpha (Eq. 26), the pooled feature z (Eq. 28), the prediction and
the label. Inference only; no training.

    python tools/export_alpha_features.py --config configs/ntu60_xsub/full.yaml \
        --ckpt runs/ntu60_xsub_full/seed0/best.pt --out analysis/alpha_seed0.npz

Then: tools/analyze_alpha.py (new Fig. 17) and tools/tsne_plot.py (Figs. 14-16).
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsmamba.model import build_model  # noqa: E402
from dsmamba.data import build_datasets  # noqa: E402
from train import load_config  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = load_config(a.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_ds, K, _ = build_datasets(cfg["data"], a.seed)
    model = build_model(cfg["model"], K, cfg["data"]["T"]).to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device))
    model.eval()
    alphas, feats, preds, labels, idxs = [], [], [], [], []
    for x, y, i in DataLoader(test_ds, batch_size=64, num_workers=4):
        logits, extra = model(x.to(device), return_extra=True)
        if extra["alpha"] is not None:
            alphas.append(extra["alpha"].float().cpu())
        feats.append(extra["z"].float().cpu())
        preds.append(logits.argmax(-1).cpu())
        labels.append(y)
        idxs.append(i)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(
        a.out,
        alpha=torch.cat(alphas).numpy() if alphas else np.array([]),
        z=torch.cat(feats).numpy(), pred=torch.cat(preds).numpy(),
        label=torch.cat(labels).numpy(), index=torch.cat(idxs).numpy(),
        dataset=cfg["data"]["name"], ckpt=a.ckpt)
    acc = (torch.cat(preds) == torch.cat(labels)).float().mean().item() * 100
    print(f"saved {a.out}: {len(test_ds)} samples, acc {acc:.2f}% (must match results.json)")


if __name__ == "__main__":
    main()
