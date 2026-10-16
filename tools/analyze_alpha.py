"""Fusion-gate statistics (Sec. 5.3.2 and Fig. 17 of the paper).

    python tools/analyze_alpha.py analysis/alpha_seed0.npz [analysis/alpha_seed1.npz ...] \
        --out analysis/fig17_alpha_per_class.pdf

Input files come from tools/export_alpha_features.py. With one file the script
reports per-class statistics and the interaction vs single-person test; with
several (one per seed) it also reports how stable the per-class ranking is.
Every number is computed from all test samples; nothing is smoothed or selected.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsmamba.ntu_labels import NTU60_CLASSES, NTU60_INTERACTION, short_name  # noqa: E402

EXAMPLES = {"hugging other person": 54, "handshaking": 57, "clapping": 9, "hand waving": 22}


def per_class(alpha, label, K):
    rows = []
    for k in range(K):
        a = alpha[label == k]
        if a.size == 0:
            continue
        rows.append({"class": k, "name": NTU60_CLASSES[k] if K == 60 else str(k),
                     "n": int(a.size), "mean": float(a.mean()), "std": float(a.std(ddof=1)),
                     "median": float(np.median(a)), "q1": float(np.percentile(a, 25)),
                     "q3": float(np.percentile(a, 75))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="analysis/fig17_alpha_per_class.pdf")
    ap.add_argument("--classes", type=int, default=60)
    a = ap.parse_args()

    runs = [np.load(f, allow_pickle=True) for f in a.files]
    for f, r in zip(a.files, runs):
        if r["alpha"].size == 0:
            raise SystemExit(f"{f}: model has no gate (fusion mode without alpha)")
    alpha, label = runs[0]["alpha"].ravel(), runs[0]["label"].ravel()
    K = a.classes
    rows = per_class(alpha, label, K)
    meds = np.array([r["median"] for r in rows])

    inter = np.isin(label, NTU60_INTERACTION)
    u = stats.mannwhitneyu(alpha[inter], alpha[~inter], alternative="two-sided")
    # rank-biserial correlation as an effect size to report next to p
    rbc = 1 - 2 * u.statistic / (inter.sum() * (~inter).sum())
    report = {
        "file": a.files[0], "n_samples": int(alpha.size),
        "alpha_overall": {"mean": float(alpha.mean()), "std": float(alpha.std(ddof=1)),
                          "min": float(alpha.min()), "max": float(alpha.max())},
        "class_median_range": [float(meds.min()), float(meds.max())],
        "interaction_A50_A60": {"mean": float(alpha[inter].mean()),
                                "std": float(alpha[inter].std(ddof=1)), "n": int(inter.sum())},
        "single_person_A1_A49": {"mean": float(alpha[~inter].mean()),
                                 "std": float(alpha[~inter].std(ddof=1)), "n": int((~inter).sum())},
        "mann_whitney_U": float(u.statistic), "p_value": float(u.pvalue),
        "rank_biserial": float(rbc),
        "examples": {n: next((r for r in rows if r["class"] == k), None)
                     for n, k in EXAMPLES.items()} if K == 60 else {},
    }
    if len(runs) > 1:
        rhos = []
        for r in runs[1:]:
            other = per_class(r["alpha"].ravel(), r["label"].ravel(), K)
            rhos.append(stats.spearmanr(meds, [o["median"] for o in other]).statistic)
        report["spearman_class_medians_vs_seed0"] = [float(x) for x in rhos]

    # A gate that barely moves cannot support a per-action interpretation.
    spread = report["class_median_range"][1] - report["class_median_range"][0]
    report["verdict"] = ("alpha varies across classes; per-class discussion is supported"
                         if spread >= 0.1 and report["p_value"] < 0.01 else
                         "alpha varies little or not significantly; use the fallback text "
                         "and remove the per-action interpretation from Sec. 3.6 and Sec. 7")
    print(json.dumps(report, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort(meds)
    data = [alpha[label == rows[i]["class"]] for i in order]
    colors = ["#d55e00" if rows[i]["class"] in NTU60_INTERACTION else "#0072b2" for i in order]
    fig, ax = plt.subplots(figsize=(7.0, 9.0))
    kw = dict(widths=0.6, showfliers=False, patch_artist=True,
              medianprops={"color": "black", "linewidth": 1})
    try:
        bp = ax.boxplot(data, orientation="horizontal", **kw)   # Matplotlib >= 3.10
    except TypeError:
        bp = ax.boxplot(data, vert=False, **kw)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels([short_name(rows[i]["class"]) if K == 60 else str(rows[i]["class"])
                        for i in order], fontsize=5.5)
    ax.set_xlabel(r"fusion gate $\alpha$ (Eq. 26): 1 = temporal stream, 0 = cross-attended spatial context")
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="grey", lw=0.6, ls=":")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#d55e00", label="two-person interaction (A50–A60)"),
                       Patch(color="#0072b2", label="single person (A1–A49)")],
              loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out)
    json.dump(report, open(os.path.splitext(a.out)[0] + "_stats.json", "w"), indent=2)
    print(f"figure: {a.out}")


if __name__ == "__main__":
    main()
