"""Dataset ke videos se MediaPipe keypoints nikal kar .npy mein save karta hai.

Dataset ka folder structure kuch bhi ho sakta hai (maslan squat/..., ya
real/squat/..., ya synthetic/push_up/...). Exercise ka label folder ya file ke
naam mein keywords (squat, push, curl, press) dhoond kar khud lagta hai.
Agar path mein 'synthetic' / 'infinite' / 'avatar' likha ho to video synthetic
maani jati hai, warna real.

Chalane ka tareeqa:
    python -m src.extract_keypoints --data data/raw_videos --out data/keypoints
Kaggle par:
    python -m src.extract_keypoints --data /kaggle/input/<dataset> --out /kaggle/working/keypoints
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.pose_module import PoseDetector


def guess_source(path: Path) -> str:
    p = str(path).lower()
    return "synthetic" if any(k in p for k in ("synthetic", "infinite", "avatar")) else "real"


def label_of(video: Path, data_dir: Path):
    """Pehle folder ke naamon se, phir file ke naam se exercise pehchano."""
    rel = video.relative_to(data_dir)
    for part in list(rel.parts[:-1]) + [rel.stem]:
        lab = config.canonical_label(part)
        if lab:
            return lab
    return None


def extract_video(video_path, detector, frame_stride):
    detector.reset_tracking()          # pichli video ka tracking saaf
    cap = cv2.VideoCapture(str(video_path))
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % frame_stride == 0:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (640, max(1, int(h * 640 / w))))
            kp, _ = detector.detect(frame)
            if kp is not None:
                frames.append(kp)
        i += 1
    cap.release()
    return np.array(frames, dtype=np.float32).reshape(-1, 33, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.RAW_VIDEOS_DIR))
    ap.add_argument("--out", default=str(config.KEYPOINTS_DIR))
    ap.add_argument("--stride", type=int, default=config.FRAME_STRIDE)
    ap.add_argument("--limit", type=int, default=0, help="har exercise ke sirf itne videos (test ke liye)")
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_videos = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in config.VIDEO_EXTS)
    if not all_videos:
        sys.exit(f"{data_dir} mein koi video nahi mili. Path check karein.")

    by_label, skipped = defaultdict(list), []
    for v in all_videos:
        lab = label_of(v, data_dir)
        (by_label[lab] if lab else skipped).append(v)
    if skipped:
        print(f"WARNING: {len(skipped)} videos ka exercise pehchana nahi gaya, skip ho gayin. Misal: {skipped[0]}")
    print({k: len(v) for k, v in by_label.items()})

    detector = PoseDetector(complexity=1)
    rows = []
    for label, videos in by_label.items():
        if args.limit:
            videos = videos[: args.limit]
        for n, v in enumerate(tqdm(videos, desc=label)):
            out_file = out_dir / f"{label}__{n:04d}.npy"
            kp = np.load(out_file) if out_file.exists() else extract_video(v, detector, args.stride)
            if not out_file.exists():
                np.save(out_file, kp)
            rows.append({"file": out_file.name, "label": label, "source": guess_source(v),
                         "frames": len(kp), "video": str(v)})
    detector.close()

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    print(manifest.groupby(["label", "source"]).size())
    print("Manifest saved:", out_dir / "manifest.csv")


if __name__ == "__main__":
    main()
