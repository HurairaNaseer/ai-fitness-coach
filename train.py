"""Teen models train karta hai (Random Forest, 1D-CNN, LSTM) aur compare karta hai.

Chalane ka tareeqa:
    python -m src.train --kp data/keypoints
Output:
    models/  (lstm.keras, cnn1d.keras, rf.pkl, labels.json)
    results/ (model_comparison.csv, confusion matrices, loss curves)
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.preprocess import build_datasets


def summary_features(X):
    """Random Forest ke liye: har feature ka mean/std/min/max poore window par."""
    return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1)], axis=1)


def build_lstm(seq_len, n_feat, n_cls):
    from tensorflow import keras
    from tensorflow.keras import layers
    return keras.Sequential([
        layers.Input((seq_len, n_feat)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_cls, activation="softmax"),
    ], name="lstm")


def build_cnn1d(seq_len, n_feat, n_cls):
    from tensorflow import keras
    from tensorflow.keras import layers
    return keras.Sequential([
        layers.Input((seq_len, n_feat)),
        layers.Conv1D(64, 5, padding="same"), layers.BatchNormalization(), layers.ReLU(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 5, padding="same"), layers.BatchNormalization(), layers.ReLU(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_cls, activation="softmax"),
    ], name="cnn1d")


def train_keras(model, d, epochs, batch):
    from tensorflow import keras
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    classes = np.unique(d["y_train"])
    cw = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=d["y_train"])))
    cbs = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
           keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5)]
    val = (d["X_val"], d["y_val"]) if len(d["X_val"]) else None
    hist = model.fit(d["X_train"], d["y_train"], validation_data=val, epochs=epochs,
                     batch_size=batch, class_weight=cw, callbacks=cbs, verbose=2)
    return hist.history


def plot_confusion(cm, labels, title, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_history(hist, title, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(hist["loss"], label="train"); ax[0].plot(hist.get("val_loss", []), label="val")
    ax[0].set_title(f"{title} loss"); ax[0].legend()
    ax[1].plot(hist["accuracy"], label="train"); ax[1].plot(hist.get("val_accuracy", []), label="val")
    ax[1].set_title(f"{title} accuracy"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def evaluate(name, y_true, y_pred, labels, results_dir, rows):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n=== {name} | test accuracy {acc:.3f} | macro-F1 {f1:.3f} ===")
    print(classification_report(y_true, y_pred, labels=range(len(labels)),
                                target_names=labels, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    plot_confusion(cm, labels, f"{name} confusion matrix", results_dir / f"confusion_{name}.png")
    rows.append({"model": name, "test_accuracy": round(acc, 4), "test_macro_f1": round(f1, 4)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp", default=str(config.KEYPOINTS_DIR))
    ap.add_argument("--models", default=str(config.MODELS_DIR))
    ap.add_argument("--results", default=str(config.RESULTS_DIR))
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--skip-dl", action="store_true", help="sirf Random Forest (TensorFlow ke bina)")
    args = ap.parse_args()

    models_dir, results_dir = Path(args.models), Path(args.results)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    d = build_datasets(args.kp)
    labels = d["labels"]
    print("Exercises:", labels)
    for s in ("train", "val", "test"):
        print(f"{s:5s}", d[f"X_{s}"].shape, np.bincount(d[f"y_{s}"], minlength=len(labels)))
    if len(d["X_test"]) == 0:
        sys.exit("Test set khali hai. Har exercise ke kam az kam 3 videos chahiye.")
    (models_dir / "labels.json").write_text(json.dumps(labels))
    d["manifest"].to_csv(results_dir / "split_manifest.csv", index=False)

    rows = []

    # 1) Random Forest (baseline)
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight="balanced",
                                random_state=config.RANDOM_STATE)
    rf.fit(summary_features(d["X_train"]), d["y_train"])
    joblib.dump(rf, models_dir / "rf.pkl")
    evaluate("random_forest", d["y_test"], rf.predict(summary_features(d["X_test"])), labels, results_dir, rows)

    # 2) 1D-CNN aur 3) LSTM
    if not args.skip_dl:
        seq_len, n_feat, n_cls = d["X_train"].shape[1], d["X_train"].shape[2], len(labels)
        for name, builder in (("cnn1d", build_cnn1d), ("lstm", build_lstm)):
            model = builder(seq_len, n_feat, n_cls)
            hist = train_keras(model, d, args.epochs, args.batch)
            model.save(models_dir / f"{name}.keras")
            plot_history(hist, name, results_dir / f"history_{name}.png")
            pred = model.predict(d["X_test"], verbose=0).argmax(1)
            evaluate(name, d["y_test"], pred, labels, results_dir, rows)

    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "model_comparison.csv", index=False)
    print("\n", table.to_string(index=False))
    print("\nSaved:", models_dir, results_dir)


if __name__ == "__main__":
    main()
