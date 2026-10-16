#!/usr/bin/env bash
# One command for every new Colab session (after mounting Drive). Works on T4, L4, A100, H100.
#   bash /content/drive/MyDrive/DS-MambaGCN_runs/code/DS-MambaGCN/scripts/colab_session.sh
#
# 1. installs the mamba-ssm / causal-conv1d kernels for THIS GPU (Drive cache, see install_mamba.sh);
# 2. copies the code and the preprocessed NTU60 data from Drive to the fast local disk;
# 3. starts the experiment queue in the background; finished runs are skipped and an
#    interrupted run continues from its last epoch, so just run this again after a disconnect.
#
# Settings (environment variables, defaults chosen from the GPU memory):
#   FAMILY    config folder: ntu60_xsub_lite (default) or ntu60_xsub (paper-size model)
#   BATCH     effective batch (default: the value in the config, 32)
#   ACCUM     gradient-accumulation steps; micro-batch = BATCH / ACCUM (exact, no BatchNorm)
#   PARALLEL  number of runs trained side by side on the GPU
#   WORKERS   data-loader processes per run
#   SEEDS     e.g. "0" for the first pass only (default 0,1,2,3,4)
#   CONFIGS   comma-separated config names (default: every row of the family; use "full" for Table 10 families)
#   DATA_NAME data folder under $D/data (default ntu_compact; ntu120_compact for ntu120_* families)
#   START     0 = install and copy only, do not start the queue
set -euo pipefail
D=${D:-/content/drive/MyDrive/DS-MambaGCN_runs}
cd /content       # the code folder is replaced below; never stay inside it
export D
[ -d "$D" ] || { echo "Drive not mounted or $D missing"; exit 1; }

# --- GPU-dependent defaults ---------------------------------------------------
read -r GPU_NAME GPU_MEM < <(python -c "import torch; p=torch.cuda.get_device_properties(0); print(p.name.replace(' ','_'), p.total_memory//2**30)")
NCPU=$(nproc)
if   [ "$GPU_MEM" -ge 70 ]; then DEF_ACCUM=1; DEF_PAR=3    # A100 80GB, H100
elif [ "$GPU_MEM" -ge 35 ]; then DEF_ACCUM=1; DEF_PAR=2    # A100 40GB
elif [ "$GPU_MEM" -ge 20 ]; then DEF_ACCUM=1; DEF_PAR=1    # L4 24GB
else                             DEF_ACCUM=2; DEF_PAR=1    # T4 16GB
fi
FAMILY=${FAMILY:-ntu60_xsub_lite}
if [ "$FAMILY" = "ntu60_xsub" ]; then                      # T=300 paper-size model: one run, smaller micro-batch
  DEF_PAR=1; [ "$GPU_MEM" -ge 35 ] && DEF_ACCUM=2 || DEF_ACCUM=4
fi
BATCH=${BATCH:-32}; ACCUM=${ACCUM:-$DEF_ACCUM}; PARALLEL=${PARALLEL:-$DEF_PAR}
WORKERS=${WORKERS:-$(( NCPU / PARALLEL > 1 ? NCPU / PARALLEL - 1 : 1 ))}
SEEDS=${SEEDS:-0,1,2,3,4}
case "$FAMILY" in ntu120_*) DEF_DATA=ntu120_compact ;; *) DEF_DATA=ntu_compact ;; esac
DATA_NAME=${DATA_NAME:-$DEF_DATA}
EXTRA=()
[ -n "${CONFIGS:-}" ] && EXTRA=(--configs "$CONFIGS")
echo "GPU $GPU_NAME ${GPU_MEM}GB, $NCPU CPUs -> FAMILY=$FAMILY BATCH=$BATCH ACCUM=$ACCUM PARALLEL=$PARALLEL WORKERS=$WORKERS SEEDS=$SEEDS"

# --- kernels, code, data ------------------------------------------------------
bash "$D/code/DS-MambaGCN/scripts/install_mamba.sh"
rm -rf /content/DS-MambaGCN && cp -r "$D/code/DS-MambaGCN" /content/DS-MambaGCN
mkdir -p /content/data /content/logs
[ -f "/content/data/$DATA_NAME/data.npy" ] || cp -r "$D/data/$DATA_NAME" /content/data/
[ "${START:-1}" = "1" ] || { echo "setup done (START=0)"; exit 0; }

# --- queue ----------------------------------------------------------------------
cd /content/DS-MambaGCN
if pgrep -f "tools/run_queue.py" >/dev/null || pgrep -f "python train.py" >/dev/null; then
  echo "a queue or a training run is still active in this session: stop it first (cell 9), then start again"; exit 0
fi
rm -rf /tmp/dsmamba_queue_locks
for i in $(seq 1 "$PARALLEL"); do
  nohup python tools/run_queue.py --runs "$D/runs" --data "/content/data/$DATA_NAME" \
      --family "$FAMILY" --seeds "$SEEDS" "${EXTRA[@]}" --batch-size "$BATCH" --accum "$ACCUM" \
      --workers "$WORKERS" --worker "${GPU_NAME}_w$i" >> "$D/queue.log" 2>&1 &
  sleep 20      # stagger start-up so the workers claim different runs and do not peak together
done
echo "queue started with $PARALLEL worker(s). Progress: tail $D/queue.log"
echo "status table: python tools/run_queue.py --runs $D/runs --data x --family $FAMILY --status"
