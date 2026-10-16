"""Step 2 of the UCF Sports pipeline: raw MediaPipe landmarks -> 25-joint NTU
layout -> cleaned, normalised, resampled tensor (NumPy only).

    python tools/ucf_preprocess.py --raw data/ucf_sports/raw \
        --out data/ucf_sports/ucf_sports_mediapipe25.npz --coords world

Implements, in this order (each step is a flag, so the paper can state exactly
what was done):
  1. missing keypoints: visibility < --vis-thr, or no person detected -> NaN
  2. temporal linear interpolation of NaNs; leading/trailing gaps take the
     nearest valid frame; a landmark never seen in a clip is filled after
     mapping from its parent joint (counted in the report)
  3. 33 -> 25 mapping of Table 9
  4. image coordinates only: x scaled by width/height, y and z flipped so that
     +y points up (world landmarks are already metric)
  5. centring at the spine base (first frame or every frame)
  6. scaling by the mean shoulder width of the clip
  7. uniform resampling to --T frames
Writes the dataset .npz and <out>_report.json with the missing-data statistics
reported in Sec. 4.2.1 of the paper.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsmamba.data_utils import uniform_resample  # noqa: E402
from dsmamba.graph import BONES  # noqa: E402

# Ten UCF Sports actions; folder names of the official release map onto them.
UCF_CLASSES = ["Diving", "Golf-Swing", "Kicking", "Lifting", "Riding-Horse",
               "Running", "Skateboarding", "Swing-Bench", "Swing-Side", "Walking"]
FOLDER_PREFIX = [("diving", 0), ("golf", 1), ("kicking", 2), ("lifting", 3),
                 ("riding", 4), ("run", 5), ("skateboarding", 6), ("swing-bench", 7),
                 ("swing-side", 8), ("walk", 9)]

# Table 9: NTU joint (0-based) <- MediaPipe landmark(s). Tuples are averaged;
# None marks joints synthesised along the trunk (see map_33_to_25).
MAPPING = {
    0: (23, 24), 1: None, 2: None, 3: 0,
    4: 11, 5: 13, 6: 15, 7: (17, 19),
    8: 12, 9: 14, 10: 16, 11: (18, 20),
    12: 23, 13: 25, 14: 27, 15: 31,
    16: 24, 17: 26, 18: 28, 19: 32,
    20: (11, 12), 21: 19, 22: 21, 23: 20, 24: 22,
}
L_SHOULDER, R_SHOULDER, SPINE_BASE, SPINE_SHOULDER, HEAD = 4, 8, 0, 20, 3


def class_of(folder: str) -> int:
    f = folder.lower()
    for prefix, k in FOLDER_PREFIX:
        if f.startswith(prefix):
            return k
    raise ValueError(f"unknown UCF Sports folder '{folder}': extend FOLDER_PREFIX")


def interpolate_nan(x: np.ndarray) -> np.ndarray:
    """x: (T, ...). Linear interpolation along time per channel; edges = nearest valid."""
    T = x.shape[0]
    flat = x.reshape(T, -1).copy()
    t = np.arange(T)
    for j in range(flat.shape[1]):
        ok = ~np.isnan(flat[:, j])
        if ok.all() or not ok.any():
            continue
        flat[:, j] = np.interp(t, t[ok], flat[ok, j])
    return flat.reshape(x.shape)


def map_33_to_25(lm: np.ndarray, head="nose", spine_mid_t=0.5, neck_t=0.5) -> np.ndarray:
    """lm: (T, 33, 3) -> (T, 25, 3) following Table 9."""
    T = lm.shape[0]
    out = np.full((T, 25, 3), np.nan, np.float32)
    for j, src in MAPPING.items():
        if src is None:
            continue
        out[:, j] = lm[:, list(src)].mean(axis=1) if isinstance(src, tuple) else lm[:, src]
    if head == "face_centroid":
        out[:, HEAD] = np.nanmean(lm[:, 0:11], axis=1)
    base, shoulder, hd = out[:, SPINE_BASE], out[:, SPINE_SHOULDER], out[:, HEAD]
    out[:, 1] = base + spine_mid_t * (shoulder - base)          # spine mid
    out[:, 2] = shoulder + neck_t * (hd - shoulder)              # neck
    return out


def fill_unseen_joints(sk: np.ndarray):
    """Joints with no valid frame in the whole clip take their parent's trajectory."""
    filled = []
    parent = {child: par for child, par in BONES}
    for _ in range(25):
        missing = [j for j in range(25) if np.isnan(sk[:, j]).all()]
        if not missing:
            break
        for j in missing:
            p = parent.get(j)
            if p is not None and not np.isnan(sk[:, p]).all():
                sk[:, j] = sk[:, p]
                filled.append(j)
    return sk, filled


def process_clip(rec, coords, vis_thr, center, head, spine_mid_t, neck_t):
    raw = rec[coords].astype(np.float32)                         # (T, 33, 4)
    det = rec["detected"].astype(bool)
    T0 = raw.shape[0]
    xyz, vis = raw[..., :3].copy(), raw[..., 3]
    miss = (vis < vis_thr) | ~det[:, None] | np.isnan(vis)
    xyz[miss] = np.nan
    stats = {"frames": int(T0), "frames_no_person": int((~det).sum()),
             "landmarks_missing_frac": float(miss.mean())}
    xyz = interpolate_nan(xyz)
    sk = map_33_to_25(xyz, head, spine_mid_t, neck_t)
    sk, filled = fill_unseen_joints(sk)
    stats["joints_never_seen"] = filled
    if np.isnan(sk).any():
        stats["unusable"] = True
        return None, stats
    if coords == "image":
        W, H = float(rec["width"]), float(rec["height"])
        sk[..., 0] *= W / H
        sk[..., 1] *= -1.0
        sk[..., 2] *= -1.0
    else:                                                         # world: MediaPipe y points down
        sk[..., 1] *= -1.0
    origin = sk[:1, SPINE_BASE] if center == "first" else sk[:, SPINE_BASE]
    sk = sk - origin[:, None, :]
    width = np.linalg.norm(sk[:, L_SHOULDER] - sk[:, R_SHOULDER], axis=-1).mean()
    if width > 1e-6:
        sk = sk / width
    stats["shoulder_width_raw"] = float(width)
    return sk, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--coords", choices=["world", "image"], default="world")
    ap.add_argument("--vis-thr", type=float, default=0.5)
    ap.add_argument("--center", choices=["first", "per_frame"], default="first")
    ap.add_argument("--head", choices=["nose", "face_centroid"], default="nose")
    ap.add_argument("--spine-mid-t", type=float, default=0.5)
    ap.add_argument("--neck-t", type=float, default=0.5)
    ap.add_argument("--T", type=int, default=150)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.raw, "*.npz")))
    xs, ys, names, report = [], [], [], {"settings": vars(a), "clips": {}}
    for f in files:
        folder, num = os.path.basename(f)[:-4].split("__")
        name = f"{folder}/{num}"
        sk, st = process_clip(np.load(f), a.coords, a.vis_thr, a.center, a.head,
                              a.spine_mid_t, a.neck_t)
        report["clips"][name] = st
        if sk is None:
            print(f"DISCARDED {name}: {st}")
            continue
        seq = sk.transpose(2, 0, 1)[..., None]                    # (3, T0, 25, 1)
        xs.append(uniform_resample(seq, seq.shape[1], a.T).astype(np.float32))
        ys.append(class_of(folder))
        names.append(name)

    c = report["clips"].values()
    lengths = [s["frames"] for s in c]
    report["summary"] = {
        "clips_total": len(files), "clips_kept": len(names),
        "clips_discarded": [n for n, s in report["clips"].items() if s.get("unusable")],
        "frames_min_max": [int(min(lengths)), int(max(lengths))] if lengths else None,
        "landmarks_missing_frac_mean": float(np.mean([s["landmarks_missing_frac"] for s in c]))
        if lengths else None,
        "frames_without_person_frac": float(sum(s["frames_no_person"] for s in c) /
                                            max(1, sum(lengths))),
        "clips_with_unseen_joints": sum(1 for s in c if s.get("joints_never_seen")),
        "class_counts": {UCF_CLASSES[k]: int(np.sum(np.array(ys) == k)) for k in range(10)},
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, x=np.stack(xs), y=np.array(ys, np.int64), names=np.array(names),
                        class_names=np.array(UCF_CLASSES), num_classes=10)
    json.dump(report, open(os.path.splitext(a.out)[0] + "_report.json", "w"), indent=2)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
