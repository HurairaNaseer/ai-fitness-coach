"""Banawati (synthetic) data par poori pipeline ka test. Dataset ya camera ki zaroorat nahi.

    python -m tests.test_pipeline
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src import exercises as ex
from src.angle_utils import calculate_angle
from src.features import N_FEATURES, frame_features, mirror_keypoints
from src.preprocess import build_datasets
from src.rep_counter import RepCounter


def fake_person(t, kind, rng):
    """Side-view insan ka banawati skeleton (33,4). Har exercise mein alag joints hilte hain."""
    kp = np.zeros((33, 4), np.float32)
    kp[:, 3] = 1.0
    s = np.sin(t)
    # khara hua insan (x, y: image coordinates, y neeche ki taraf)
    pts = {0: (0.5, 0.15), 11: (0.5, 0.3), 12: (0.5, 0.3), 13: (0.5, 0.45), 14: (0.5, 0.45),
           15: (0.5, 0.58), 16: (0.5, 0.58), 23: (0.5, 0.55), 24: (0.5, 0.55),
           25: (0.5, 0.75), 26: (0.5, 0.75), 27: (0.5, 0.95), 28: (0.5, 0.95)}
    for i, (x, y) in pts.items():
        kp[i, :2] = (x, y)
    if kind == "squat":
        for i in (23, 24):  kp[i, 1] += 0.12 * (1 + s) / 2; kp[i, 0] -= 0.1 * (1 + s) / 2
        for i in (25, 26):  kp[i, 0] += 0.12 * (1 + s) / 2
    elif kind == "push_up":
        for i in (11, 12, 0):  kp[i, 1] += 0.10 * (1 + s) / 2
        for i in (13, 14):     kp[i, 0] += 0.10 * (1 + s) / 2
    elif kind == "barbell_biceps_curl":
        for i in (15, 16):     kp[i, 1] -= 0.25 * (1 + s) / 2; kp[i, 0] += 0.12 * (1 + s) / 2
    elif kind == "shoulder_press":
        for i in (15, 16):     kp[i, 1] -= 0.4 * (1 + s) / 2
        for i in (13, 14):     kp[i, 1] -= 0.15 * (1 + s) / 2
    kp[:, :2] += rng.normal(0, 0.004, (33, 2))
    return kp


def test_angle():
    assert abs(calculate_angle((0, 1), (0, 0), (1, 0)) - 90) < 1e-3
    assert abs(calculate_angle((1, 0), (0, 0), (-1, 0)) - 180) < 1e-3
    print("OK  calculate_angle")


def test_features():
    rng = np.random.RandomState(0)
    kp = fake_person(0.5, "squat", rng)
    f = frame_features(kp)
    assert f.shape == (N_FEATURES,) and np.isfinite(f).all()
    m = mirror_keypoints(kp)
    assert np.allclose(mirror_keypoints(m)[:, :2], kp[:, :2], atol=1e-6)
    print("OK  features + mirror")


def test_rep_counter():
    for name, start_high in (("squat", True), ("barbell_biceps_curl", True),
                             ("push_up", True), ("shoulder_press", False)):
        s = ex.SETTINGS[name]
        c = RepCounter(s["low"], s["high"])
        lo, hi = s["low"] - 25, s["high"] + 15
        t = np.linspace(0, 5 * 2 * np.pi, 400)
        wave = (hi + lo) / 2 + (hi - lo) / 2 * (np.cos(t) if start_high else -np.cos(t))
        for a in wave:
            c.update(a)
        assert c.reps == 5, (name, c.reps)
    # jitter se ghalat rep nahi ginni chahiye
    c = RepCounter(100, 160)
    rng = np.random.RandomState(1)
    for a in 170 + rng.normal(0, 3, 300):
        c.update(a)
    assert c.reps == 0
    print("OK  rep counter (5 reps sahi, jitter par 0)")


def test_rep_feedback():
    assert ex.rep_check("squat", {"min": 130, "max": 170}) == "depth"
    assert ex.rep_check("squat", {"min": 85, "max": 170}) == "good"
    assert ex.rep_check("shoulder_press", {"min": 90, "max": 120}) == "press_partial"
    print("OK  rep feedback rules")


def test_dataset_and_training():
    rng = np.random.RandomState(0)
    kinds = ["squat", "push_up", "barbell_biceps_curl", "shoulder_press"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rows = []
        for kind in kinds:
            for n in range(14):
                src = "real" if n < 5 else "synthetic"
                T = int(rng.randint(60, 120))
                ph = rng.uniform(0, 6)
                kp = np.stack([fake_person(ph + 0.35 * i * rng.uniform(0.9, 1.1), kind, rng) for i in range(T)])
                f = f"{kind}__{n:04d}.npy"
                np.save(tmp / f, kp)
                rows.append({"file": f, "label": kind, "source": src, "frames": T, "video": "x"})
        pd.DataFrame(rows).to_csv(tmp / "manifest.csv", index=False)

        d = build_datasets(tmp)
        m = d["manifest"]
        tr, te = set(m[m.split == "train"].file), set(m[m.split == "test"].file)
        assert tr.isdisjoint(te), "video train aur test dono mein!"
        assert set(m[m.split == "test"].source) == {"real"}, "test mein synthetic nahi honi chahiye"
        for s in ("train", "val", "test"):
            assert d[f"X_{s}"].ndim == 3 and d[f"X_{s}"].shape[1:] == (config.SEQ_LEN, N_FEATURES), s
        print("OK  preprocess:", {s: d[f"X_{s}"].shape[0] for s in ("train", "val", "test")}, "windows")

        # train.py ko poora chalao (chhote epochs)
        import subprocess
        cmd = [sys.executable, "-m", "src.train", "--kp", str(tmp), "--models", str(tmp / "m"),
               "--results", str(tmp / "r"), "--epochs", "3"]
        try:
            import tensorflow  # noqa: F401
        except Exception:
            cmd.append("--skip-dl")
            print("    (TensorFlow nahi mila: sirf Random Forest test ho raha hai)")
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=config.ROOT)
        assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
        cmp = pd.read_csv(tmp / "r" / "model_comparison.csv")
        print(cmp.to_string(index=False))
        assert (tmp / "m" / "labels.json").exists()
        print("OK  train.py chala, models aur results ban gaye")


if __name__ == "__main__":
    test_angle()
    test_features()
    test_rep_counter()
    test_rep_feedback()
    test_dataset_and_training()
    print("\nSab tests pass.")
