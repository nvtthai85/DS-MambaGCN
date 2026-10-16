# DS-MambaGCN

Dual-Stream Mamba-GCN for skeleton-based action recognition: a bidirectional Mamba
stream models each joint trajectory over time, an adaptive-graph GCN stream models
the joints of every frame, and cross-stream attention fusion (CSAF) with a learned
gate combines the two.

This repository contains the code, configuration files, data-preparation scripts
and analysis tools needed to reproduce every table and figure of the paper:

> Thai Thong Vo Nguyen, Loc Vinh Cu, Hai Thanh Nguyen. *Dual-Stream Mamba-GCN Networks for Efficient Skeleton-Based Action
> Recognition.* Applied Intelligence (under review).

```
Input skeleton (T x V x 3) ──► embedding (position + motion + positional encoding)
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
  Temporal Mamba Stream                     Spatial GCN Stream
  bidirectional selective scan              A_phys + A_data + A_adapt (AGTL)
  per joint, O(T)                           per frame, O(V^2)
              └───────────────────┬───────────────────┘
                                  ▼
             CSAF: per-frame cross-attention + gate α (Eq. 26)
                                  ▼
                       pooling ─► classifier
```

## Results

The numbers below are filled in from `runs/*/seed*/results.json` with
`tools/aggregate_results.py` when release **v1.0** is tagged. Every entry is the
mean ± sample standard deviation over seeds 0–4 of one fixed configuration; model
selection uses a validation set carved from the training data, and each test set
is evaluated exactly once per run.

| Benchmark | Top-1 (%) | Config |
|---|---|---|
| NTU RGB+D 60 X-Sub | pending v1.0 | `configs/ntu60_xsub/full.yaml` |
| NTU RGB+D 60 X-View | pending v1.0 | `configs/ntu60_xview/full.yaml` |
| NTU RGB+D 120 X-Sub | pending v1.0 | `configs/ntu120_xsub/full.yaml` |
| NTU RGB+D 120 X-Set | pending v1.0 | `configs/ntu120_xset/full.yaml` |
| UCF Sports (Lan et al. split) | pending v1.0 | `configs/ucf_sports/full.yaml` |

Training logs, `results.json` files and checkpoints of every run will be attached
to the v1.0 release.

## Installation

```bash
git clone https://github.com/nvtthai85/DS-MambaGCN
cd DS-MambaGCN
pip install -r requirements.txt
pip install einops ninja
pip install mamba-ssm causal-conv1d --no-build-isolation   # CUDA kernels (recommended)
python -m pytest -q tests/
```

Without `mamba-ssm` the model falls back to a pure-PyTorch selective scan
(`dsmamba/mamba.py`), which gives the same results but is much slower. On Google
Colab use `scripts/install_mamba.sh`, which builds the kernels for the runtime's
GPU only and caches them on Drive (see [Colab](#google-colab)).

Tested with Python 3.10–3.12, PyTorch 2.x, CUDA 11.8/12.x, on NVIDIA T4 and A100.

## Data

**NTU RGB+D 60 / 120.** Request the skeleton files (`nturgbd_skeletons_s001_to_s017.zip`,
and `..._s018_to_s032.zip` for NTU120) from the
[ROSE Lab](https://rose1.ntu.edu.sg/dataset/actionRecognition/) together with the
official list of samples with missing skeletons, then:

```bash
# NTU60 (56,578 samples after removing the 302 official missing skeletons)
python tools/ntu_preprocess.py --raw /path/to/nturgb+d_skeletons --out data/ntu_compact \
    --missing NTU_RGBD_samples_with_missing_skeletons.txt --max-action 60
# NTU120 (both archives extracted into one folder)
python tools/ntu_preprocess.py --raw /path/to/all_skeletons --out data/ntu120_compact \
    --missing NTU_RGBD120_samples_with_missing_skeletons.txt --max-action 120
```

The output is a memory-mapped frame array plus per-sample metadata; the X-Sub,
X-View and X-Set splits are derived from the official subject, camera and setup
ids at load time, and `report.json` records the body-selection statistics.

**UCF Sports Actions.** Download the dataset from
[UCF CRCV](https://www.crcv.ucf.edu/data/UCF_Sports_Action.php), then extract
MediaPipe skeletons, map the 33 MediaPipe landmarks to the 25 NTU joints and
write the split of Lan et al. (103 training / 47 test videos, `splits/ucf_sports_lan2011.txt`):

```bash
pip install mediapipe opencv-python
python tools/ucf_extract_mediapipe.py --root /path/to/ucf_sports_actions --out data/ucf_sports/raw
python tools/ucf_preprocess.py --raw data/ucf_sports/raw \
    --out data/ucf_sports/ucf_sports_mediapipe25.npz --coords world
python tools/make_ucf_split.py --data data/ucf_sports/ucf_sports_mediapipe25.npz \
    --lan-split splits/ucf_sports_lan2011.txt --val-frac 0.15 --out data/ucf_sports/splits
```

`extraction_meta.json`, `ucf_sports_mediapipe25_report.json` and `split_meta.json`
record the MediaPipe version and settings, the missing-landmark statistics and
the split sizes.

## Training and evaluation

```bash
python train.py --config configs/ntu60_xsub/full.yaml --seed 0
```

Each run writes `runs/<config>/seed<k>/` with `config_resolved.json`, `log.jsonl`
(one line per epoch), `best.pt`, `last.pt` and `results.json` (test accuracy,
selected epoch, parameter count, environment). The test split is evaluated once,
after training, with the checkpoint chosen on the validation set. Interrupted runs
resume from `resume.pt` when the same command is run again. Useful flags:
`--batch-size`, `--accum` (exact gradient accumulation, the model has no
BatchNorm), `--workers`, `--data-path`.

To run a whole table with all seeds (finished runs are skipped):

```bash
python tools/run_queue.py --runs runs --data data/ntu_compact --family ntu60_xsub
python tools/aggregate_results.py --runs runs --reference ntu60_xsub_full
```

`aggregate_results.py` prints mean ± std per configuration, the paired difference
to the reference with a paired t-test over seeds, and LaTeX rows.

## Reproducing the paper

[`docs/REPRODUCE.md`](docs/REPRODUCE.md) lists, for every table and figure, the
config files, commands, expected compute and the output file each number comes from.

| Paper | Configs / tool |
|---|---|
| Table 10 (NTU comparison) | `configs/{ntu60_xsub,ntu60_xview,ntu120_xsub,ntu120_xset}/full.yaml` |
| Table 11 (UCF Sports) | `configs/ucf_sports/full.yaml` |
| Table 13 (component ablation) | `configs/ntu60_xsub/t13_*.yaml` |
| Table 14 (graph terms) | `configs/ntu60_xsub/t14_*.yaml` |
| Table 15 (parallel vs serial coupling) | `configs/ntu60_xsub/t15_*.yaml` |
| Table 16 (fusion: cross-attention vs gate) | `configs/ntu60_xsub/t16_*.yaml`, `t15_t16_parallel_add.yaml` |
| Tables 6, 17 (params, FLOPs, speed) | `tools/audit_complexity.py`, `tools/bench_speed.py` |
| Figs. 14–16 (t-SNE) | `tools/export_alpha_features.py`, `tools/tsne_plot.py` |
| Fig. 17 (fusion gate α) | `tools/export_alpha_features.py`, `tools/analyze_alpha.py` |
| Sec. 3.5 (A_data edges) | `tools/adata_edges.py` |

## Model switches

All ablations are switches of one model (`dsmamba/model.py`), set in the YAML configs:

| Key | Values |
|---|---|
| `topology` | `phys`, `phys+data`, `phys+adapt`, `full` |
| `coupling` | `parallel`, `serial_gcn_mamba`, `serial_mamba_gcn`, `spatial_only`, `temporal_only` |
| `fusion` | `csaf`, `xattn_only`, `gate_only`, `add`, `concat` |
| `scan` | `bi`, `fwd` |
| `use_motion` | `true`, `false` |

## Google Colab

`notebooks/DS_MambaGCN_colab.ipynb` runs the full pipeline on Colab (T4, L4 or
A100) with code, data, checkpoints and results kept on Google Drive.
`scripts/colab_session.sh` installs the Mamba kernels for the current GPU,
copies code and data to local disk and starts the experiment queue in the
background; after a disconnect, running it again continues where it stopped.

## Repository layout

```
dsmamba/    model.py  mamba.py  graph.py  data.py  data_utils.py  ntu_labels.py
configs/    ntu60_xsub/  ntu60_xview/  ntu120_xsub/  ntu120_xset/  ucf_sports/  ntu60_xsub_lite/
tools/      data preparation, experiment queue, aggregation, analysis, complexity audit
scripts/    Colab session setup, Mamba kernel installer
splits/     UCF Sports split of Lan et al. (2011)
notebooks/  Colab notebook
tests/      NumPy and PyTorch unit tests
docs/       REPRODUCE.md
train.py
```

`configs/ntu60_xsub_lite/` is a reduced-cost variant (T = 64, N = 16, expansion 1,
shared scan weights) for quick experiments on small GPUs; it is not used for the
numbers in the paper.

## Citation

```bibtex
@article{dsmambagcn,
  title   = {Dual-Stream Mamba-GCN Networks for Efficient Skeleton-Based Action Recognition},
  author  = {Nguyen, Thai Thong Vo and Cu, Loc Vinh and Nguyen, Hai Thanh},
  journal = {Applied Intelligence},
  year    = {2026},
  note    = {under review}
}
```

## License

Released under the MIT License (see `LICENSE`). The NTU RGB+D and UCF
Sports datasets are subject to their own terms of use and are not redistributed here.

## Acknowledgements

The selective-scan kernels come from [mamba-ssm](https://github.com/state-spaces/mamba).
The adaptive-graph design builds on 2s-AGCN and CTR-GCN, and the NTU body-selection
procedure follows common practice in ST-GCN and CTR-GCN preprocessing.
