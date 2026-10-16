"""Measure training speed, inference throughput, peak memory and parameter counts.

    python tools/bench_speed.py --config configs/ntu60_xsub/full.yaml --batch-size 16

Uses random inputs of the configured shape, so it needs no data. Prints one JSON
line per configuration; the numbers feed Table 6 / Table 17 and the time budget.
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsmamba.model import build_model, param_breakdown  # noqa: E402
from train import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--classes", type=int, default=60)
    ap.add_argument("--bodies", type=int, default=2)
    ap.add_argument("--train-samples", type=int, default=37529, help="X-Sub train minus 5% hold-out")
    a = ap.parse_args()
    dev = torch.device("cuda")
    for path in a.config:
        try:
            bench_one(a, path, dev)
        except torch.cuda.OutOfMemoryError:
            print(json.dumps({"config": path, "batch": a.batch_size, "error": "out of memory: lower "
                              "--batch-size and train with --accum"}), flush=True)
        torch.cuda.empty_cache()


def bench_one(a, path, dev):
    cfg = load_config(path)
    T = cfg["data"]["T"]
    model = build_model(cfg["model"], a.classes, T).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.CrossEntropyLoss()
    x = torch.randn(a.batch_size, 3, T, 25, a.bodies, device=dev)
    y = torch.randint(0, a.classes, (a.batch_size,), device=dev)
    torch.cuda.reset_peak_memory_stats()
    for i in range(a.steps + 3):
        if i == 3:
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = crit(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    step = (time.time() - t0) / a.steps
    mem = torch.cuda.max_memory_allocated() / 2**30
    model.eval()
    xi = torch.randn(1, 3, T, 25, a.bodies, device=dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(5):
            model(xi)
        torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(50):
            model(xi)
        torch.cuda.synchronize()
    fps = 50 / (time.time() - t1)
    steps_per_epoch = a.train_samples // a.batch_size
    print(json.dumps({
        "config": cfg.get("name"), "batch": a.batch_size, "T": T,
        "sec_per_step": round(step, 4), "peak_mem_GB": round(mem, 2),
        "epoch_min_est": round(step * steps_per_epoch / 60, 1),
        "run_120ep_h_est": round(step * steps_per_epoch * 120 / 3600, 1),
        "infer_samples_per_s_bs1": round(fps, 1),
        "params": {k: v for k, v in param_breakdown(model).items()},
        "gpu": torch.cuda.get_device_name(0)}), flush=True)


if __name__ == "__main__":
    main()
