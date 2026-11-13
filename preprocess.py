"""Keypoints (.npy) se training windows aur train/val/test split banata hai.

Zaroori usool: split VIDEO ke hisab se hota hai, window ke hisab se nahi.
Warna ek hi video ki windows train aur test dono mein ja kar accuracy jhooti
zyada kar deti hain.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.features import sequence_features, mirror_keypoints


def split_videos(manifest, seed=config.RANDOM_STATE):
    """Har row ko 'train' / 'val' / 'test' dena.

    Agar real aur synthetic dono hon: test sirf real videos se banta hai
    (har exercise ki aadhi real videos), synthetic sirf train/val mein jati hai.
    Warna 70/15/15 random split (har exercise mein alag).
    """
    rng = np.random.RandomState(seed)
    manifest = manifest.copy()
    manifest["split"] = "train"
    mixed = {"real", "synthetic"} <= set(manifest["source"])

    for label, g in manifest.groupby("label"):
        idx = rng.permutation(g.index.to_numpy())
        if mixed:
            real = [i for i in idx if manifest.at[i, "source"] == "real"]
            n_test = max(1, round(len(real) * 0.5)) if len(real) > 1 else 0
            test = real[:n_test]
        else:
            n_test = max(1, round(len(idx) * 0.15)) if len(idx) > 2 else 0
            test = list(idx[:n_test])
        rest = [i for i in idx if i not in set(test)]
        n_val = max(1, round(len(rest) * 0.15)) if len(rest) > 3 else 0
        val = rest[:n_val]
        manifest.loc[test, "split"] = "test"
        manifest.loc[val, "split"] = "val"
    return manifest


def make_windows(feats, seq_len=config.SEQ_LEN, stride=config.SEQ_STRIDE):
    """feats (T, F) -> (N, seq_len, F). Chhoti video ko aakhri frame se pad karta hai."""
    T = len(feats)
    if T == 0:
        return np.empty((0, seq_len, feats.shape[-1] if feats.ndim == 2 else 0), np.float32)
    if T < seq_len:
        pad = np.repeat(feats[-1:], seq_len - T, axis=0)
        return np.concatenate([feats, pad])[None]
    starts = range(0, T - seq_len + 1, stride)
    return np.stack([feats[s:s + seq_len] for s in starts])


def build_datasets(kp_dir=config.KEYPOINTS_DIR, augment=True):
    """Return dict: X_train, y_train, X_val, y_val, X_test, y_test, labels, manifest."""
    kp_dir = Path(kp_dir)
    manifest = pd.read_csv(kp_dir / "manifest.csv")
    manifest = manifest[manifest["frames"] >= 5].reset_index(drop=True)   # bohat chhoti videos hata do
    labels = sorted(manifest["label"].unique())
    label_to_id = {l: i for i, l in enumerate(labels)}
    manifest = split_videos(manifest)

    data = {s: ([], []) for s in ("train", "val", "test")}
    for _, row in manifest.iterrows():
        kp = np.load(kp_dir / row["file"])
        variants = [kp]
        if augment and row["split"] == "train":
            variants.append(mirror_keypoints(kp))          # left-right mirror
        for k in variants:
            w = make_windows(sequence_features(k))
            data[row["split"]][0].append(w)
            data[row["split"]][1].append(np.full(len(w), label_to_id[row["label"]]))

    out = {"labels": labels, "manifest": manifest}
    for s, (xs, ys) in data.items():
        out[f"X_{s}"] = np.concatenate(xs).astype(np.float32) if xs else np.empty((0,))
        out[f"y_{s}"] = np.concatenate(ys).astype(np.int64) if ys else np.empty((0,), np.int64)
    return out


if __name__ == "__main__":
    d = build_datasets()
    for s in ("train", "val", "test"):
        print(s, d[f"X_{s}"].shape, np.bincount(d[f"y_{s}"], minlength=len(d["labels"])))
    print(d["manifest"].groupby(["split", "source"]).size())
