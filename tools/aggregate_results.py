"""Collect runs/<config>/seed*/results.json into mean +/- std table rows.

    python tools/aggregate_results.py --runs runs --reference ntu60_xsub_full

Prints a Markdown summary and LaTeX rows for Tables 13-16. Standard deviation
is the sample std (ddof = 1) over seeds; state this in the table captions.
When --reference is given, a paired t-test over common seeds compares each
configuration with it (seeds share initialisation streams, so pairing is valid).
"""
import argparse
import re
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

TABLE_OF = {
    "t13_": "Table 13", "t14_": "Table 14", "t15_t16_": "Tables 15+16",
    "t15_": "Table 15", "t16_": "Table 16", "full": "all tables",
}


def table_label(name):
    short = re.sub(r"^ntu60_xsub_(lite_|t64_)?", "", name)
    for k, v in TABLE_OF.items():
        if short.startswith(k):
            return v
    return ""


def collect(runs_dir):
    by_cfg = defaultdict(dict)
    for f in glob.glob(os.path.join(runs_dir, "*", "seed*", "results.json")):
        r = json.load(open(f))
        by_cfg[r["config"]][r["seed"]] = r
    return by_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    by_cfg = collect(a.runs)
    if not by_cfg:
        raise SystemExit(f"no results.json under {a.runs}")
    ref = by_cfg.get(a.reference, {}) if a.reference else {}

    rows = []
    for name in sorted(by_cfg):
        seeds = sorted(by_cfg[name])
        acc = np.array([by_cfg[name][s]["test_acc"] for s in seeds])
        std = acc.std(ddof=1) if acc.size > 1 else float("nan")
        params = by_cfg[name][seeds[0]]["params"]["total"] / 1e6
        row = {"config": name, "table": table_label(name), "seeds": seeds,
               "mean": float(acc.mean()), "std": float(std), "params_M": params,
               "per_seed": acc.tolist()}
        common = sorted(set(seeds) & set(ref)) if ref and name != a.reference else []
        if len(common) >= 2:
            x = np.array([by_cfg[name][s]["test_acc"] for s in common])
            y = np.array([ref[s]["test_acc"] for s in common])
            t = stats.ttest_rel(x, y)
            row.update(delta_vs_ref=float((x - y).mean()), p_paired=float(t.pvalue),
                       n_paired=len(common))
        rows.append(row)

    print(f"{'config':<42}{'table':<14}{'n':>3}{'acc':>16}{'params':>9}{'Δref':>8}{'p':>9}")
    for r in rows:
        d = f"{r['delta_vs_ref']:+.2f}" if "delta_vs_ref" in r else ""
        p = f"{r['p_paired']:.3g}" if "p_paired" in r else ""
        print(f"{r['config']:<42}{r['table']:<14}{len(r['seeds']):>3}"
              f"{r['mean']:>10.1f}±{r['std']:.2f}{r['params_M']:>9.2f}{d:>8}{p:>9}")
        if len(r["seeds"]) < 5:
            print(f"{'':<42}  only seeds {r['seeds']} finished")
    print("\nLaTeX rows (accuracy cell):")
    for r in rows:
        print(f"% {r['config']}\n{r['mean']:.1f}$\\pm${r['std']:.2f} & {r['params_M']:.2f} \\\\")
    if a.json_out:
        json.dump(rows, open(a.json_out, "w"), indent=2)


if __name__ == "__main__":
    main()
