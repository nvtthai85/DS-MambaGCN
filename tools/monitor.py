"""Progress table for the experiment queue: which run is training, which epoch and
step, losses, validation accuracy, speed and estimated time left.

    python tools/monitor.py --runs /content/drive/MyDrive/DS-MambaGCN_runs/runs \
        --family ntu60_xsub_lite --seeds 0
    python tools/monitor.py ... --watch 60        # refresh every 60 s (Ctrl-C / stop button to quit)

Reads only the files the runs write (log.jsonl, stdout.log, results.json), so it can
be run at any time, also from another Colab session, without touching the training.
"""
import argparse
import datetime
import glob
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.run_queue import QUEUE  # noqa: E402

STEP_RE = re.compile(r"epoch (\d+) step (\d+)/(\d+) loss ([\d.]+) ([\d.]+)s/step")


def fmt_h(h):
    if h is None:
        return "-"
    return f"{h:.1f} h" if h < 48 else f"{h / 24:.1f} d"


def read_run(d, epochs_total):
    r = {"status": "todo", "epoch": None, "step": "", "loss": None, "val": None, "best": None,
         "sps": None, "eta_h": None, "test": None, "age_min": None}
    if os.path.exists(os.path.join(d, "results.json")):
        res = json.load(open(os.path.join(d, "results.json")))
        r.update(status="done", test=res.get("test_acc"), best=res.get("best_val_acc"),
                 epoch=res.get("epochs_run"))
        return r
    logf, outf = os.path.join(d, "log.jsonl"), os.path.join(d, "stdout.log")
    if not os.path.exists(outf) and not os.path.exists(logf):
        return r
    recs = []
    if os.path.exists(logf):
        recs = [json.loads(ln) for ln in open(logf) if ln.strip()]
    epochs_done = len(recs)
    r["status"] = "running"
    if recs:
        r["loss"] = recs[-1].get("train_loss")
        r["val"] = recs[-1].get("val_acc")
        vals = [x["val_acc"] for x in recs if "val_acc" in x]
        r["best"] = max(vals) if vals else None
    cur_epoch, step_txt, sps = epochs_done, "", None
    if os.path.exists(outf):
        r["age_min"] = (time.time() - os.path.getmtime(outf)) / 60
        lines = open(outf, errors="ignore").read().splitlines()[-200:]
        if lines and ("Error" in lines[-1] or "Traceback" in "\n".join(lines[-30:])) \
                and not any(STEP_RE.search(x) for x in lines[-5:]):
            r["status"] = "FAILED"
        for ln in reversed(lines):
            m = STEP_RE.search(ln)
            if m:
                e, s, n, loss, sp = int(m[1]), int(m[2]), int(m[3]), float(m[4]), float(m[5])
                if e >= epochs_done:
                    cur_epoch, step_txt, sps = e, f"{s}/{n}", sp
                    r["loss"] = loss
                    steps_left = (n - s) + (epochs_total - e - 1) * n
                    r["eta_h"] = steps_left * sp * 1.05 / 3600      # +5 % for validation
                break
    if r["eta_h"] is None and recs:
        per_epoch = sum(x["sec"] for x in recs[-5:]) / min(5, len(recs)) * 1.05
        r["eta_h"] = (epochs_total - epochs_done) * per_epoch / 3600
    if r["status"] == "running" and r["age_min"] is not None and r["age_min"] > 20:
        r["status"] = "stopped?"          # no output for 20 min: session ended or crashed
    r.update(epoch=cur_epoch, step=step_txt, sps=sps)
    return r


def gpu_line():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        name, util, used, tot = [x.strip() for x in out.strip().split(",")]
        return f"GPU {name}: {util}% busy, {int(used) / 1024:.1f}/{int(tot) / 1024:.0f} GB"
    except Exception:
        return "GPU: not visible from here"


def render(a):
    cfg_dir = os.path.join(ROOT, "configs", a.family)
    configs = a.configs.split(",") if a.configs else \
        [c for c in QUEUE if os.path.exists(os.path.join(cfg_dir, c + ".yaml"))]
    seeds = [int(s) for s in a.seeds.split(",")]
    rows, total_left, per_run = [], 0.0, []
    for s in seeds:
        for c in configs:
            r = read_run(os.path.join(a.runs, f"{a.family}_{c}", f"seed{s}"), a.epochs)
            rows.append((c, s, r))
            if r["status"] == "running" and r["sps"]:
                per_run.append(r["sps"])
    typical = None
    if per_run:
        steps = next((int(r["step"].split("/")[1]) for _, _, r in rows if r["step"]), 1190)
        typical = sum(per_run) / len(per_run) * steps * a.epochs * 1.05 / 3600
    for _, _, r in rows:
        if r["status"] in ("running", "stopped?") and r["eta_h"]:
            total_left += r["eta_h"]
        elif r["status"] == "todo" and typical:
            total_left += typical

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    done = sum(r["status"] == "done" for *_, r in rows)
    print(f"{now}   family {a.family}   seeds {a.seeds}   {done}/{len(rows)} runs done")
    print(gpu_line())
    hdr = f"{'config':<24}{'seed':>5}  {'status':<9}{'epoch':>9}  {'step':>10}  {'loss':>7}  " \
          f"{'val %':>6}  {'best %':>6}  {'s/step':>6}  {'time left':>9}  {'test %':>6}"
    print(hdr)
    print("-" * len(hdr))
    for c, s, r in rows:
        ep = f"{r['epoch'] + 1 if r['status'] != 'done' else r['epoch']}/{a.epochs}" \
            if r["epoch"] is not None else "-"
        f = lambda v, p=2: "-" if v is None else f"{v:.{p}f}"  # noqa: E731
        print(f"{c:<24}{s:>5}  {r['status']:<9}{ep:>9}  {r['step'] or '-':>10}  {f(r['loss'], 3):>7}  "
              f"{f(r['val']):>6}  {f(r['best']):>6}  {f(r['sps'], 3):>6}  {fmt_h(r['eta_h']):>9}  "
              f"{f(r['test']):>6}")
    if typical:
        print(f"\none run ≈ {fmt_h(typical)} at the current speed; "
              f"all remaining runs of this batch ≈ {fmt_h(total_left)} of training")
    stopped = [f"{c} seed{s}" for c, s, r in rows if r["status"] in ("stopped?", "FAILED")]
    if stopped:
        print("ATTENTION: " + ", ".join(stopped) + " — no progress for 20 min or an error; "
              "see its stdout.log, then restart the queue (cell 6).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--family", default="ntu60_xsub_lite")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--configs", default=None, help="comma-separated; default: every row of the family")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--watch", type=int, default=0, help="refresh every N seconds")
    a = ap.parse_args()
    if not a.watch:
        render(a)
        return
    try:
        from IPython.display import clear_output
    except ImportError:
        clear_output = None
    while True:
        if clear_output:
            clear_output(wait=True)
        else:
            print("\033[2J\033[H", end="")
        render(a)
        sys.stdout.flush()
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
