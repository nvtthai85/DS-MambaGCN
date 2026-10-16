"""Run the experiment queue (all table rows, all seeds) across several Colab sessions.

    python tools/run_queue.py --runs /content/drive/MyDrive/DS-MambaGCN_runs/runs \
        --data /content/data/ntu_compact --batch-size 16

Order: seed 0 of every configuration first, then seed 1 of every configuration,
and so on, so that a complete first pass over all table rows exists as early as
possible. A run that already has results.json is skipped; a run with resume.pt
continues from its last finished epoch (train.py handles this). Safe to restart
at any time: kill it, open a new session, start it again.

Several workers can share one queue on a large GPU (A100: two lite runs side by
side use it better than one): start run_queue.py N times with the same
arguments. Each run is claimed with a lock file on LOCAL disk (--lock-dir), so
two workers of the same session never train the same run, and locks vanish
with the VM, so a disconnect never leaves a stale lock. Do not run the queue in
two Colab sessions at the same time: they do not see each other's locks.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Full model first (every table compares against it), then the ablation rows of Tables 14-16.
QUEUE = [
    "full",
    "t14_topo_phys",
    "t14_topo_phys_data",
    "t14_topo_phys_adapt",
    "t15_serial_gcn_mamba",
    "t15_serial_mamba_gcn",
    "t15_t16_parallel_add",
    "t16_gate_only",
    "t16_xattn_only",
    "t13_dual_agtl_concat",
]


def status(runs, configs, seeds, family):
    rows = []
    for s in seeds:
        for c in configs:
            d = os.path.join(runs, f"{family}_{c}", f"seed{s}")
            if os.path.exists(os.path.join(d, "results.json")):
                r = json.load(open(os.path.join(d, "results.json")))
                rows.append((c, s, "done", f"{r['test_acc']:.2f}"))
            elif os.path.exists(os.path.join(d, "log.jsonl")):
                n = sum(1 for _ in open(os.path.join(d, "log.jsonl")))
                rows.append((c, s, "partial", f"{n} epochs"))
            else:
                rows.append((c, s, "todo", ""))
    return rows


def try_lock(lock_dir, name):
    os.makedirs(lock_dir, exist_ok=True)
    path = os.path.join(lock_dir, name.replace(os.sep, "__") + ".lock")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--family", default="ntu60_xsub", help="config folder, e.g. ntu60_xsub_lite")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--configs", default=",".join(QUEUE))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=None, help="override (smoke tests only)")
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--status", action="store_true", help="print the table and exit")
    ap.add_argument("--lock-dir", default="/tmp/dsmamba_queue_locks",
                    help="local (not Drive) folder for run claims of parallel workers")
    ap.add_argument("--worker", default="w0", help="name shown in the log")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    configs = a.configs.split(",")
    missing = [c for c in configs if not os.path.exists(os.path.join(ROOT, "configs", a.family, c + ".yaml"))]
    if missing:
        print(f"no config file in configs/{a.family}/ for {missing}; skipped", flush=True)
        configs = [c for c in configs if c not in missing]
    os.makedirs(a.runs, exist_ok=True)

    if a.status:
        for row in status(a.runs, configs, seeds, a.family):
            print("%-24s seed%d  %-8s %s" % row)
        return

    for s in seeds:
        for c in configs:
            d = os.path.join(a.runs, f"{a.family}_{c}", f"seed{s}")
            if os.path.exists(os.path.join(d, "results.json")):
                continue
            lock = try_lock(a.lock_dir, f"{a.family}_{c}_seed{s}")
            if lock is None:            # another worker of this session has it
                continue
            if os.path.exists(os.path.join(d, "results.json")):   # finished meanwhile
                os.remove(lock)
                continue
            os.makedirs(d, exist_ok=True)
            cmd = [sys.executable, "train.py", "--config", f"configs/{a.family}/{c}.yaml",
                   "--seed", str(s), "--out", a.runs, "--data-path", a.data,
                   "--batch-size", str(a.batch_size), "--workers", str(a.workers),
                   "--accum", str(a.accum)]
            if a.epochs:
                cmd += ["--epochs", str(a.epochs)]
            if a.max_train_samples:
                cmd += ["--max-train-samples", str(a.max_train_samples)]
            print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M}] {a.worker} start {c} seed{s}", flush=True)
            t0 = time.time()
            try:
                with open(os.path.join(d, "stdout.log"), "a") as log:
                    rc = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            finally:
                os.remove(lock)
            print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M}] {a.worker} end {c} seed{s} rc={rc} "
                  f"{(time.time() - t0) / 3600:.2f} h", flush=True)
            if rc != 0:
                print(f"run failed, see {d}/stdout.log; stopping the queue", flush=True)
                return rc
    print(f"{a.worker}: nothing left to claim")


if __name__ == "__main__":
    sys.exit(main())
