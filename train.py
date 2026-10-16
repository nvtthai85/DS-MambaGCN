"""Train and evaluate one configuration with one seed.

    python train.py --config configs/ntu60_xsub/full.yaml --seed 0
    python train.py --config configs/ntu60_xsub/fusion_gate_only.yaml --seed 3 --out runs

Writes runs/<config_name>/seed<k>/{log.jsonl, results.json, best.pt}.
The test split is evaluated exactly once, with the checkpoint chosen by the
validation policy of the config (never by test accuracy).
"""
import argparse
import functools
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from dsmamba.model import build_model, param_breakdown
from dsmamba.data import build_datasets


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _worker_init(worker_id, base):
    np.random.seed(base + worker_id)


def load_config(path):
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    base = cfg.pop("base", None)
    if base:  # shallow-merge each section over the base config
        merged = load_config(os.path.join(os.path.dirname(path), base))
        for k, v in cfg.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
        cfg = merged
    return cfg


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    correct = total = 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--epochs", type=int, default=None, help="override (smoke tests only)")
    ap.add_argument("--batch-size", type=int, default=None, help="override, e.g. 16 on a T4")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--accum", type=int, default=None,
                    help="gradient accumulation steps; the loader uses batch_size // accum per "
                         "step, so the effective batch stays batch_size (exact: no BatchNorm)")
    ap.add_argument("--max-train-samples", type=int, default=None, help="smoke tests only")
    ap.add_argument("--data-path", default=None, help="override data.path")
    ap.add_argument("--log-every", type=int, default=200, help="print progress every N steps")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.workers is not None:
        cfg["train"]["workers"] = args.workers
    if args.accum:
        cfg["train"]["accum"] = args.accum
    if args.max_train_samples:
        cfg["data"]["max_train_samples"] = args.max_train_samples
    if args.data_path:
        cfg["data"]["path"] = args.data_path
    name = cfg.get("name", os.path.splitext(os.path.basename(args.config))[0])
    out_dir = os.path.join(args.out, name, f"seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcfg = cfg["train"]
    epochs = args.epochs or tcfg["epochs"]
    amp = bool(tcfg.get("amp", True)) and device.type == "cuda"

    train_ds, val_ds, test_ds, num_classes, info = build_datasets(cfg["data"], args.seed)
    g = torch.Generator().manual_seed(args.seed)
    nw = tcfg.get("workers", 8)
    accum = int(tcfg.get("accum", 1))
    if tcfg["batch_size"] % accum:
        raise ValueError("batch_size must be divisible by accum")
    micro = tcfg["batch_size"] // accum
    # Same permutation as an un-accumulated loader: micro-batches k*accum..k*accum+accum-1
    # together contain exactly the samples of effective batch k.
    train_ld = DataLoader(train_ds, micro, shuffle=True, generator=g,
                          num_workers=nw, pin_memory=True, drop_last=True,
                          worker_init_fn=functools.partial(_worker_init, base=args.seed * 1000))
    mk = lambda ds: DataLoader(ds, tcfg.get("eval_batch_size", 64), shuffle=False,
                               num_workers=nw, pin_memory=True)
    val_ld = mk(val_ds) if val_ds is not None else None
    test_ld = mk(test_ds)

    model = build_model(cfg["model"], num_classes, cfg["data"]["T"]).to(device)
    params = param_breakdown(model)

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    opt_steps = len(train_ld) // accum           # optimizer steps per epoch
    warm, total_steps = tcfg["warmup_epochs"] * opt_steps, epochs * opt_steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) /
                                                                  max(1, total_steps - warm))))
    crit = nn.CrossEntropyLoss(label_smoothing=tcfg.get("label_smoothing", 0.1))
    try:                                   # PyTorch >= 2.3
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):    # PyTorch 2.0-2.2
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    clip = tcfg.get("grad_clip", None)

    policy = cfg["data"].get("validation", {}).get("policy", "final_epoch")
    if policy != "final_epoch" and val_ld is None:
        raise ValueError("validation policy needs a validation set")
    patience = tcfg.get("patience", None)
    best_val, best_epoch, bad, start_epoch = -1.0, -1, 0, 0
    ckpt_path = os.path.join(out_dir, "resume.pt")
    if os.path.exists(os.path.join(out_dir, "results.json")):
        print(f"{out_dir}: already finished, skipping")
        return
    if os.path.exists(ckpt_path):  # resume after a Colab disconnect
        # load on CPU: RNG states must stay CPU ByteTensors; load_state_dict moves
        # model and optimizer tensors to the parameters' device
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        best_val, best_epoch, bad, start_epoch = ck["best_val"], ck["best_epoch"], ck["bad"], ck["epoch"] + 1
        torch.set_rng_state(ck["torch_rng"])
        if torch.cuda.is_available() and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([r.cpu() for r in ck["cuda_rng"]])
        g.set_state(ck["loader_rng"])
        print(f"resumed from epoch {start_epoch}")
    with open(os.path.join(out_dir, "config_resolved.json"), "w") as fh:
        json.dump({"config": cfg, "args": vars(args), "params": params, "data_info": {
            k: v for k, v in info.items() if not isinstance(v, list)}}, fh, indent=2)
    log = open(os.path.join(out_dir, "log.jsonl"), "a")
    epoch = start_epoch - 1
    stopped_early = start_epoch > 0 and policy != "final_epoch" and patience and bad >= patience

    for epoch in range(start_epoch, epochs):
        if stopped_early:
            break
        model.train()
        train_ds.set_epoch(epoch)
        t0, loss_sum, n = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        step = 0
        for k, (x, y, _) in enumerate(train_ld):
            if k >= opt_steps * accum:              # incomplete last group
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = crit(model(x), y)
            scaler.scale(loss / accum).backward()
            loss_sum += loss.item() * y.numel()
            n += y.numel()
            if (k + 1) % accum:
                continue
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   clip if clip else float("inf"))
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if args.log_every and step % args.log_every == 0:
                print(f"epoch {epoch} step {step}/{opt_steps} loss {loss_sum / n:.4f} "
                      f"{(time.time() - t0) / step:.3f}s/step", flush=True)
        rec = {"epoch": epoch, "train_loss": loss_sum / n, "lr": sched.get_last_lr()[0],
               "grad_norm_preclip": float(gnorm), "sec": time.time() - t0}
        if val_ld is not None:
            rec["val_acc"] = evaluate(model, val_ld, device, amp)
            if rec["val_acc"] > best_val:
                best_val, best_epoch, bad = rec["val_acc"], epoch, 0
                torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
            else:
                bad += 1
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                    "best_val": best_val, "best_epoch": best_epoch, "bad": bad,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "loader_rng": g.get_state()}, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)   # atomic, safe on Drive
        if policy != "final_epoch" and patience and bad >= patience:
            break

    torch.save(model.state_dict(), os.path.join(out_dir, "last.pt"))
    if policy == "final_epoch":
        selected = "last.pt"
    else:
        selected = "best.pt"
        model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=device,
                                         weights_only=True))
    test_acc = evaluate(model, test_ld, device, amp)   # the only test evaluation

    res = {"config": name, "config_path": args.config, "seed": args.seed,
           "test_acc": test_acc, "selection_policy": policy, "selected_checkpoint": selected,
           "best_val_acc": best_val if val_ld is not None else None,
           "selected_epoch": best_epoch if policy != "final_epoch" else epoch,
           "epochs_run": epoch + 1, "params": params, "data_info": info,
           "torch": torch.__version__, "device": str(device)}
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({k: res[k] for k in ("config", "seed", "test_acc", "selected_epoch")}))
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


if __name__ == "__main__":
    main()
