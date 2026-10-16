"""Regenerate Figs. 14-16 from real test features (Sec. 5.3.2).

    python tools/tsne_plot.py analysis/alpha_seed0.npz --title "DS-MambaGCN" \
        --classes 0 1 9 22 23 26 7 49 58 11 --per-class 60 --out analysis/fig16_tsne.pdf

The legend uses the official NTU60 names (dsmamba/ntu_labels.py). The sample
indices that were plotted are saved next to the figure so the plot can be
reproduced exactly; keep that file in the code release.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsmamba.ntu_labels import short_name  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("--classes", type=int, nargs="+", required=True, help="0-based class ids")
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = np.load(a.features, allow_pickle=True)
    z, y, idx = d["z"], d["label"], d["index"]
    rng = np.random.default_rng(a.seed)
    pick = []
    for c in a.classes:
        cand = np.flatnonzero(y == c)
        pick.extend(rng.choice(cand, size=min(a.per_class, cand.size), replace=False))
    pick = np.array(pick)

    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    emb = TSNE(n_components=2, perplexity=a.perplexity, init="pca",
               random_state=a.seed).fit_transform(z[pick])
    sil = silhouette_score(z[pick], y[pick])  # in feature space, not in the 2-D map

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    cmap = plt.get_cmap("tab10")
    for k, c in enumerate(a.classes):
        m = y[pick] == c
        ax.scatter(emb[m, 0], emb[m, 1], s=7, alpha=0.75, color=cmap(k % 10), label=short_name(c))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{a.title}  (silhouette in feature space: {sil:.3f})", fontsize=9)
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    json.dump({"features": a.features, "test_indices": idx[pick].tolist(),
               "classes": a.classes, "perplexity": a.perplexity, "seed": a.seed,
               "silhouette": float(sil)},
              open(os.path.splitext(a.out)[0] + "_samples.json", "w"))
    print(f"{a.out}: {len(pick)} points, silhouette {sil:.3f}")


if __name__ == "__main__":
    main()
