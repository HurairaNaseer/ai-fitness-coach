"""Poore project ki settings ek jagah."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- Paths ----
RAW_VIDEOS_DIR = ROOT / "data" / "raw_videos"   # har exercise ka alag folder
KEYPOINTS_DIR = ROOT / "data" / "keypoints"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# ---- Sequence settings ----
SEQ_LEN = 30          # model ko 30 frames ki sequence milti hai
SEQ_STRIDE = 10       # training windows ke beech gap
FRAME_STRIDE = 2      # har 2nd frame use hota hai (30fps -> ~15fps)
                      # NOTE: realtime app mein bhi yehi stride use hota hai

# ---- Model settings ----
EPOCHS = 60
BATCH_SIZE = 32
RANDOM_STATE = 42

# ---- Realtime app ----
PRED_SMOOTHING = 5       # pichli 5 predictions ka majority vote
MIN_CONFIDENCE = 0.60    # isse kam confidence par exercise "Unknown"
FEEDBACK_LANG = "ur"     # "ur" = Roman Urdu, "en" = English
VOICE_LANG = "en"
MIN_VISIBILITY = 0.5 
LOST_FRAMES = 10

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# ---- Exercise names ----
# Folder/file ke naam mein yeh keywords milen to us exercise ka label lagta hai.
# (order important hai: pehle curl, phir press, phir push)
EXERCISE_KEYWORDS = [
    ("barbell_biceps_curl", ("curl", "bicep")),
    ("shoulder_press", ("shoulder", "press")),
    ("push_up", ("push",)),
    ("squat", ("squat",)),
]
DISPLAY_NAMES = {
    "squat": "Squat",
    "push_up": "Push-up",
    "barbell_biceps_curl": "Bicep Curl",
    "shoulder_press": "Shoulder Press",
}


def canonical_label(text):
    """'Barbell Biceps Curl' / 'push-up' / 'squat_01' -> standard label ya None."""
    t = str(text).lower()
    for label, keys in EXERCISE_KEYWORDS:
        if any(k in t for k in keys):
            return label
    return None
