"""Live app (OpenCV window): webcam/video -> exercise pehchan + rep count + posture feedback.

Chalane ke tareeqe:
    python -m src.realtime_app --model cnn1d         # webcam, trained model se exercise pehchane
    python -m src.realtime_app --source video.mp4    # video file par
    python -m src.realtime_app --exercise squat      # model ke bina, exercise khud tay (rep counter test)
    python -m src.realtime_app --lang en             # screen ki zubaan English (default: config.FEEDBACK_LANG)
    python -m src.realtime_app --no-voice            # awaz band
'q' dabane par band.  (Browser wali UI ke liye: streamlit run app/streamlit_app.py)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src import exercises as ex_rules
from src import session_log
from src.engine import CoachEngine
from src.i18n import t
from src.pose_module import PoseDetector
from src.voice import Voice


def load_classifier(model_name, models_dir, lang):
    """Return (predict_fn, labels)."""
    models_dir = Path(models_dir)
    labels = json.loads((models_dir / "labels.json").read_text())
    path = models_dir / f"{model_name}.keras"
    if not path.exists():
        sys.exit(t("c_no_model", lang, path=path))
    from tensorflow import keras
    model = keras.models.load_model(path)
    return (lambda x: model(x, training=False).numpy()[0]), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0 = webcam, ya video file ka path")
    ap.add_argument("--model", default="cnn1d", choices=["lstm", "cnn1d"])
    ap.add_argument("--models-dir", default=str(config.MODELS_DIR))
    ap.add_argument("--exercise", choices=list(ex_rules.SETTINGS), help="model ki jagah exercise khud tay karein")
    ap.add_argument("--lang", default=config.FEEDBACK_LANG, choices=["en", "ur"])
    ap.add_argument("--no-voice", action="store_true")
    args = ap.parse_args()
    lang = args.lang

    predict_fn = labels = None
    if not args.exercise:
        predict_fn, labels = load_classifier(args.model, args.models_dir, lang)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(t("c_cam_fail", lang))

    detector = PoseDetector(complexity=1)
    engine = CoachEngine(detector, predict_fn, labels, forced=args.exercise,
                         voice=Voice(enabled=not args.no_voice), lang=lang)

    print(t("c_started", lang))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("Fitness AI Coach", engine.process(frame))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    summary = session_log.append(engine.counters)
    txt = ", ".join(f"{config.DISPLAY_NAMES[k]}: {v}" for k, v in summary.items()) or t("c_no_reps", lang)
    print("\n" + t("c_summary", lang, txt=txt))


if __name__ == "__main__":
    main()