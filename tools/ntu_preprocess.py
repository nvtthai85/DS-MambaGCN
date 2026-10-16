"""Convert raw NTU RGB+D (60 or 120) .skeleton files into a compact, mmap-able format.

    python tools/ntu_preprocess.py --raw /content/data/ntu_raw/nturgb+d_skeletons \
        --out /content/data/ntu_compact --missing NTU_RGBD_samples_with_missing_skeletons.txt

Writes
  <out>/data.npy   float32 (F_total, 2, 25, 3): every kept frame, two bodies (zero-padded)
  <out>/meta.npz   name, offset, length, label (0-based), subject, camera, setup, replication
  <out>/report.json  counts of kept / skipped samples and the body-selection statistics

Benchmark splits (X-Sub, X-View, X-Set) are NOT baked in; dsmamba/data.py derives
them from the subject / camera / setup ids using the official protocol lists, so the
same file serves every protocol.

Body selection (standard practice, cf. ST-GCN and CTR-GCN preprocessing):
  1. bodies tracked by their Kinect bodyID across frames;
  2. bodies present in fewer than --min-frames frames are dropped as noise;
  3. when several bodies remain, those whose x-spread exceeds --spread-ratio times
     their y-spread in most frames are dropped (Kinect ghosts of furniture are wide
     and flat);
  4. the remaining bodies are ranked by motion energy (sum of per-joint coordinate
     variance over time) and the top two are kept, highest energy first.
Steps 2 and 3 only remove a body when another body remains, so a sample is skipped
only when its file contains no body at all (listed in the report). This keeps the
official sample counts (NTU60: 56,578 after the 302 known missing skeletons).
"""
import argparse
import functools
import glob
import json
import multiprocessing
import os
import re
import sys
from collections import OrderedDict

import numpy as np

NAME_RE = re.compile(r"S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})")


def parse_name(name):
    m = NAME_RE.search(name)
    if not m:
        raise ValueError(f"bad NTU file name: {name}")
    s, c, p, r, a = (int(x) for x in m.groups())
    return {"setup": s, "camera": c, "subject": p, "replication": r, "action": a}


def read_skeleton(path):
    """Return (n_frames, OrderedDict bodyID -> {frame_idx: (25,3) array})."""
    with open(path) as fh:
        tokens = fh.read().split("\n")
    it = iter(tokens)
    n_frames = int(next(it))
    bodies = OrderedDict()
    for f in range(n_frames):
        n_bodies = int(next(it))
        for _ in range(n_bodies):
            info = next(it).split()
            body_id = info[0]
            n_joints = int(next(it))
            joints = np.empty((n_joints, 3), np.float32)
            for j in range(n_joints):
                vals = next(it).split()
                joints[j] = (float(vals[0]), float(vals[1]), float(vals[2]))
            bodies.setdefault(body_id, {})[f] = joints
    return n_frames, bodies


def select_bodies(n_frames, bodies, min_frames, spread_ratio, max_bodies=2):
    """Pick up to two bodies. The filters only ever remove a body when another one
    remains, so no sample with at least one detected body is dropped (keeping the
    official test-set sizes intact; a lone actor who is lying down or has arms
    stretched out is wide, not noise)."""
    stats = {"bodies_raw": len(bodies)}
    info = []
    for bid, frames in bodies.items():
        arr = np.stack([frames[f] for f in sorted(frames)])          # (t, 25, 3)
        x_spread = arr[..., 0].max(1) - arr[..., 0].min(1)
        y_spread = arr[..., 1].max(1) - arr[..., 1].min(1)
        wide = (x_spread > spread_ratio * np.maximum(y_spread, 1e-6)).mean() > 0.5
        info.append({"bid": bid, "frames": frames, "n": len(frames), "wide": bool(wide),
                     "energy": float(arr.var(axis=0).sum())})
    cand = info
    # 1) short-lived detections, only if a longer body exists
    longer = [b for b in cand if b["n"] >= min_frames]
    if longer:
        cand = longer
    # 2) wide-and-flat ghosts, only when several bodies compete and one survives
    if len(cand) > 1:
        narrow = [b for b in cand if not b["wide"]]
        if narrow:
            cand = narrow
    cand.sort(key=lambda b: -b["energy"])
    kept = [(b["energy"], b["bid"], b["frames"]) for b in cand[:max_bodies]]
    stats["bodies_kept"] = len(kept)
    stats["filtered"] = len(info) - len(cand)
    if not kept:
        return None, stats
    out = np.zeros((n_frames, 2, 25, 3), np.float32)
    present = np.zeros(n_frames, bool)
    for k, (_, _, frames) in enumerate(kept):
        for f, joints in frames.items():
            out[f, k] = joints
            present[f] = True
    # trim frames where no kept body is present (leading/trailing and gaps)
    out = out[present]
    stats["frames"] = int(out.shape[0])
    return out, stats


def _process_one(job, min_frames, spread_ratio):
    path, name, info = job
    try:
        n_frames, bodies = read_skeleton(path)
    except Exception as e:  # truncated or malformed file
        return name, info, None, None, str(e)
    seq, st = select_bodies(n_frames, bodies, min_frames, spread_ratio)
    return name, info, seq, st, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder with *.skeleton files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--missing", default=None, help="official list of samples with missing skeletons")
    ap.add_argument("--min-frames", type=int, default=11)
    ap.add_argument("--spread-ratio", type=float, default=0.8)
    ap.add_argument("--max-action", type=int, default=120, help="60 to keep only NTU60 classes")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N files (testing)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.raw, "**", "*.skeleton"), recursive=True))
    if a.limit:
        files = files[:a.limit]
    missing = set()
    if a.missing and os.path.exists(a.missing):
        missing = {ln.strip() for ln in open(a.missing) if ln.strip() and not ln.startswith("#")}
    os.makedirs(a.out, exist_ok=True)

    chunks, meta = [], {k: [] for k in ("name", "offset", "length", "label", "subject",
                                         "camera", "setup", "replication")}
    skipped = {"official_missing": [], "no_valid_body": [], "unreadable": [], "action_out_of_range": []}
    n_two_body = n_filtered = 0
    offset = 0
    jobs = []
    for path in files:
        name = os.path.basename(path).split(".")[0]
        info = parse_name(name)
        if info["action"] > a.max_action:
            skipped["action_out_of_range"].append(name)
        elif name in missing:
            skipped["official_missing"].append(name)
        else:
            jobs.append((path, name, info))
    work = functools.partial(_process_one, min_frames=a.min_frames, spread_ratio=a.spread_ratio)
    with multiprocessing.Pool(a.workers) as pool:
        for i, (name, info, seq, st, err) in enumerate(pool.imap(work, jobs, chunksize=64)):
            if err:
                skipped["unreadable"].append(f"{name}: {err}")
                continue
            if seq is None or seq.shape[0] == 0:
                skipped["no_valid_body"].append(name)
                continue
            n_two_body += int(st["bodies_kept"] == 2)
            n_filtered += int(st["filtered"] > 0)
            chunks.append(seq)
            meta["name"].append(name)
            meta["offset"].append(offset)
            meta["length"].append(seq.shape[0])
            meta["label"].append(info["action"] - 1)
            for k in ("subject", "camera", "setup", "replication"):
                meta[k].append(info[k])
            offset += seq.shape[0]
            if (i + 1) % 5000 == 0:
                print(f"{i + 1}/{len(jobs)} files, {len(chunks)} kept", flush=True)

    data = np.lib.format.open_memmap(os.path.join(a.out, "data.npy"), mode="w+",
                                     dtype=np.float32, shape=(offset, 2, 25, 3))
    pos = 0
    for c in chunks:
        data[pos:pos + len(c)] = c
        pos += len(c)
    data.flush()
    np.savez(os.path.join(a.out, "meta.npz"),
             **{k: np.array(v) for k, v in meta.items()})
    lengths = np.array(meta["length"])
    report = {
        "files_found": len(files), "samples_kept": len(chunks),
        "skipped_counts": {k: len(v) for k, v in skipped.items()},
        "skipped": {k: v[:50] for k, v in skipped.items()},
        "two_body_samples": n_two_body,
        "samples_with_filtered_bodies": n_filtered,
        "frames_total": int(offset),
        "frames_per_sample": {"min": int(lengths.min()), "median": int(np.median(lengths)),
                              "max": int(lengths.max())} if len(lengths) else None,
        "classes": int(len(set(meta["label"]))),
        "settings": vars(a),
    }
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=2)
    print(json.dumps({k: report[k] for k in ("files_found", "samples_kept", "skipped_counts",
                                             "frames_per_sample", "classes")}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
