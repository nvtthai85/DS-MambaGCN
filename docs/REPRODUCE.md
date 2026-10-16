# Reproducing the paper

This guide goes from raw data to every number in the paper. All runs use one
configuration per table row and seeds 0, 1, 2, 3, 4; no hyperparameter differs
between seeds.

## 0. Environment

```bash
pip install -r requirements.txt einops ninja
pip install mamba-ssm causal-conv1d --no-build-isolation
python -m pytest -q tests/
python -c "import torch, mamba_ssm; print(torch.__version__, torch.version.cuda, mamba_ssm.__version__)"
```

Record the printed versions; `results.json` of each run also stores the PyTorch
version and device.

## 1. Measure the cost before launching anything

```bash
python tools/bench_speed.py --config configs/ntu60_xsub/full.yaml --batch-size 16
python tools/audit_complexity.py --D 256 --T 300 --classes 60
```

`bench_speed.py` prints seconds per step, peak memory and `run_120ep_h_est`, the
estimated hours of one 120-epoch NTU60 X-Sub run on the current GPU. Multiply by
the number of runs you plan (configs × 5 seeds) to size the budget. Reference
point measured on an NVIDIA T4 (16 GB): the paper-size NTU model takes 5.6 s per
step at batch 8. Use `--accum` to keep the effective batch of 32 when memory is short.

## 2. Data

See the README, section *Data*. Checks:

| File | Expected |
|---|---|
| `data/ntu_compact/report.json` | `samples_kept` 56,578, 60 classes |
| NTU60 X-Sub split (printed by `train.py`) | 40,091 train+val / 16,487 test |
| `data/ucf_sports/splits/split_meta.json` | 88 train / 15 val / 47 test, 1 test video = 2.128 pp |

## 3. Model selection protocol

* **NTU RGB+D** has no validation split. `validation: {policy: holdout, fraction: 0.05}`
  holds out 5 % of the training set, stratified by class with a fixed split seed,
  and selects the epoch with the best validation accuracy (patience 20).
* **UCF Sports** uses 15 validation videos drawn from the 103 training videos of
  the Lan et al. split, stratified by class, identical for all seeds.
* In both cases the test set is evaluated once, after training.

## 4. Runs per table

```bash
# Table 10, one family per benchmark (full model only)
for fam in ntu60_xsub ntu60_xview ntu120_xsub ntu120_xset; do
  python tools/run_queue.py --runs runs --data data/<compact folder> --family $fam --configs full
done
# Tables 13-16, NTU60 X-Sub ablations (all rows, all seeds)
python tools/run_queue.py --runs runs --data data/ntu_compact --family ntu60_xsub \
    --configs full,t13_spatial_only,t13_temporal_only,t13_dual_fixed_graph_concat,t13_dual_agtl_concat,t13_unidirectional,t13_no_motion,t14_topo_phys,t14_topo_phys_data,t14_topo_phys_adapt,t15_serial_gcn_mamba,t15_serial_mamba_gcn,t15_t16_parallel_add,t16_gate_only,t16_xattn_only
# Table 11, UCF Sports
for s in 0 1 2 3 4; do python train.py --config configs/ucf_sports/full.yaml --seed $s; done
```

The queue runs seed 0 of every configuration first, so a complete first pass
over all rows exists early; several workers can share one GPU (see `tools/run_queue.py`).

## 5. Tables

```bash
python tools/aggregate_results.py --runs runs --reference ntu60_xsub_full --json-out runs/summary.json
```

| Table | Source |
|---|---|
| 10, 11 | `runs/<benchmark>_full/seed*/results.json` → `summary.json` |
| 13–16 | `summary.json`, paired t-test against `ntu60_xsub_full` |
| 6 | `tools/audit_complexity.py` (params, MACs including the selective scan) |
| 17 | `tools/bench_speed.py` (FPS at batch size 1, GPU name printed) |

## 6. Figures and analyses

```bash
for s in 0 1 2 3 4; do
  python tools/export_alpha_features.py --config configs/ntu60_xsub/full.yaml \
      --ckpt runs/ntu60_xsub_full/seed$s/best.pt --out analysis/alpha_seed$s.npz
done
python tools/analyze_alpha.py analysis/alpha_seed{0,1,2,3,4}.npz --out analysis/fig17_alpha_per_class.pdf
python tools/tsne_plot.py analysis/alpha_seed0.npz --classes 0 1 9 22 23 26 7 49 58 11 \
    --per-class 60 --perplexity 30 --seed 0 --title DS-MambaGCN --out analysis/fig16_tsne.pdf
python tools/adata_edges.py runs/ntu60_xsub_full/seed*/best.pt --k 24
```

`analyze_alpha.py` reports per-class statistics of the gate α, a Mann–Whitney
test between two-person and single-person classes with its effect size, and the
seed-to-seed rank correlation of class medians. `adata_edges.py` reports the share
of the 24 strongest learned A_data edges that are not bones, against the 92.0 %
expected for random pairs.

## 7. Release checklist (v1.0)

1. All runs of sections 4 have `results.json`; `summary.json` regenerated.
2. README results table filled from `summary.json` (no other source).
3. `runs/**/log.jsonl`, `results.json`, `config_resolved.json` and the `best.pt`
   of every full-model run attached to the GitHub release.
4. Tag `v1.0`; archive the release on Zenodo and add the DOI to the README.
