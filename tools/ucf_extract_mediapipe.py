"""Step 1 of the UCF Sports pipeline (revised Sec. 4.2.1): run MediaPipe Pose on
every video and save the raw 33 landmarks per frame, untouched.

    pip install mediapipe opencv-python
    python tools/ucf_extract_mediapipe.py --root /data/ucf_sports_actions \
        --out data/ucf_sports/raw --model-complexity 2

Input layout (official release): <root>/<ClassFolder>/<NNN>/ containing either
an .avi file or the frames as .jpg. Video id = "<ClassFolder>/<NNN>".

For each video it writes <out>/<ClassFolder>__<NNN>.npz with
  world  (T, 33, 4): world landmarks x, y, z in metres (hip-centred) + visibility
  image  (T, 33, 4): normalised image x, y, relative z + visibility
  detected (T,): bool, a person was found in that frame
  width, height, fps
and <out>/extraction_meta.json with the MediaPipe version and every setting:
these are the values reported in Sec. 4.2.1 of the paper.
"""
import argparse
import glob
import json
import os

import numpy as np


def iter_frames(video_dir):
    import cv2
    avis = sorted(glob.glob(os.path.join(video_dir, "*.avi")))
    if avis:
        cap = cv2.VideoCapture(avis[0])
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        return frames, fps, os.path.basename(avis[0])
    jpgs = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
    return [cv2.imread(p) for p in jpgs], 0.0, "jpg-frames"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-complexity", type=int, default=2, choices=[0, 1, 2])
    ap.add_argument("--min-detection-confidence", type=float, default=0.5)
    ap.add_argument("--min-tracking-confidence", type=float, default=0.5)
    ap.add_argument("--no-smooth", action="store_true")
    a = ap.parse_args()

    import cv2
    import mediapipe as mp
    os.makedirs(a.out, exist_ok=True)
    meta = {"mediapipe_version": mp.__version__, "opencv_version": cv2.__version__,
            "api": "mediapipe.solutions.pose (BlazePose GHUM 3D)",
            "model_complexity": a.model_complexity,
            "min_detection_confidence": a.min_detection_confidence,
            "min_tracking_confidence": a.min_tracking_confidence,
            "smooth_landmarks": not a.no_smooth, "static_image_mode": False, "videos": {}}

    video_dirs = sorted(d for d in glob.glob(os.path.join(a.root, "*", "*")) if os.path.isdir(d))
    for vd in video_dirs:
        cls, num = vd.rstrip("/").split(os.sep)[-2:]
        frames, fps, src = iter_frames(vd)
        if not frames:
            print(f"WARNING no frames in {vd}")
            continue
        H, W = frames[0].shape[:2]
        world = np.full((len(frames), 33, 4), np.nan, np.float32)
        image = np.full((len(frames), 33, 4), np.nan, np.float32)
        detected = np.zeros(len(frames), bool)
        # A fresh tracker per video so tracking state never leaks across clips.
        with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=a.model_complexity,
                                    smooth_landmarks=not a.no_smooth,
                                    min_detection_confidence=a.min_detection_confidence,
                                    min_tracking_confidence=a.min_tracking_confidence) as pose:
            for t, f in enumerate(frames):
                res = pose.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                if res.pose_landmarks is None:
                    continue
                detected[t] = True
                image[t] = [[l.x, l.y, l.z, l.visibility] for l in res.pose_landmarks.landmark]
                if res.pose_world_landmarks is not None:
                    world[t] = [[l.x, l.y, l.z, l.visibility]
                                for l in res.pose_world_landmarks.landmark]
        np.savez_compressed(os.path.join(a.out, f"{cls}__{num}.npz"), world=world, image=image,
                            detected=detected, width=W, height=H, fps=fps)
        meta["videos"][f"{cls}/{num}"] = {"frames": len(frames), "fps": fps, "source": src,
                                          "detected_frames": int(detected.sum()),
                                          "width": W, "height": H}
        print(f"{cls}/{num}: {len(frames)} frames, person found in {detected.sum()}")
    json.dump(meta, open(os.path.join(a.out, "extraction_meta.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
