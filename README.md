# AI-Powered Real-Time Exercise Recognition, Repetition Counting and Posture Feedback System

A computer-vision and deep-learning system that watches a person exercise through a webcam (or an uploaded video), **recognises which exercise is being performed**, **counts the repetitions**, and gives **real-time posture feedback** on screen and by voice.


---

## Table of Contents
1. [Features](#features)
2. [How It Works](#how-it-works)
3. [Dataset](#dataset)
4. [Results](#results)
5. [Project Structure](#project-structure)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Training the Models (Kaggle)](#training-the-models-kaggle)
9. [Configuration and Tuning](#configuration-and-tuning)
10. [Testing](#testing)
11. [Limitations](#limitations)
12. [Future Work](#future-work)
13. [Troubleshooting](#troubleshooting)
14. [Acknowledgements](#acknowledgements)

---

## Features

- **Exercise recognition** for four exercises: Squat, Push-up, Barbell Bicep Curl, Shoulder Press.
- **Automatic repetition counting** from joint angles.
- **Posture feedback** (e.g. "Go deeper", "Keep your chest up") shown on screen and spoken aloud.
- **Live camera mode** and **video upload mode**. The upload mode produces a report with the detected exercise, rep count, a result for every rep, and an angle-over-time chart.
- **Browser interface (Streamlit)** and a lightweight **OpenCV window** app.
- **Bilingual interface**: English and Roman Urdu.
- **Session history** saved to CSV and visualised in the app.
- **Robust to missing people**: when nobody is in view the system resets instead of showing stale results.

## How It Works

```
                 ┌──────────────────────────── Training (offline, Kaggle) ───────────────────────────┐
 Dataset videos →│ MediaPipe Pose → 74 features/frame → 30-frame windows → Random Forest / 1D-CNN / LSTM │
                 └────────────────────────────────────────────────────────────────────────────────────┘

                 ┌──────────────────────────── Inference (live or video) ────────────────────────────┐
 Camera / video →│ MediaPipe Pose → features → 1D-CNN → exercise name (majority vote over 5 predictions) │
                 │                ↘ joint angles → rep counter + posture rules → screen + voice          │
                 └────────────────────────────────────────────────────────────────────────────────────┘
```

| Task | Method | Type |
|---|---|---|
| Body keypoints (33 landmarks) | MediaPipe Pose | Pretrained |
| Exercise classification | 1D-CNN (also LSTM and Random Forest for comparison) | **Custom-trained** |
| Repetition counting | Angle thresholds with a two-state machine | Rule-based |
| Posture feedback | Angle rules per exercise | Rule-based |

**Features per frame (74):** 33 keypoints (x, y), centred on the hips and divided by torso length (66 values), plus 8 joint angles (elbows, shoulders, hips, knees; left and right). This makes the input independent of the person's size and distance from the camera.

**Sequences:** every 2nd frame is used, and the model sees a window of 30 frames (about 2 seconds). Training windows are taken with a stride of 10 frames.

**Rep counting:** a rep is counted when the main joint angle goes below a *low* threshold and then back above a *high* threshold. The angle is smoothed over 5 frames to avoid false counts.

| Exercise | Angle used | Low / High threshold (degrees) |
|---|---|---|
| Squat | Knee | 100 / 160 |
| Push-up | Elbow | 100 / 150 |
| Bicep Curl | Elbow | 60 / 140 |
| Shoulder Press | Elbow | 100 / 150 |

**Posture rules:**

| Exercise | Check | Feedback |
|---|---|---|
| Squat | Not deep enough (per rep) | Go deeper |
| Squat | Torso leaning too far forward | Keep your chest up |
| Push-up | Hips sagging or raised | Keep your body straight |
| Push-up | Not deep enough (per rep) | Lower your chest further |
| Bicep Curl | Elbow drifting away from the body | Keep your elbow close to your body |
| Bicep Curl | Incomplete curl (per rep) | Curl all the way up |
| Shoulder Press | Arms not fully extended (per rep) | Press all the way up |
| Shoulder Press | Leaning back | Keep your back straight |

## Dataset

[Real-Time Exercise Recognition Dataset](https://www.kaggle.com/datasets/riccardoriccio/real-time-exercise-recognition-dataset) (Kaggle, by Riccardo Riccio), covering Squat, Push-up, Barbell Bicep Curl and Shoulder Press. It combines real videos of trainers, additional videos from stock-footage sites, and synthetic avatar videos.

Preparation used in this project:

- Hammer-curl videos and the author's `my_test_video_1` folder were **excluded**.
- **567 videos** were used: 138 bicep curl, 143 push-up, 143 shoulder press, 143 squat.
- The split is done **per video** (not per window) to avoid leakage. The **test set contains only real videos**; synthetic videos are used only for training and validation.
- Training-time augmentation: left-right mirroring.

| Split | Windows |
|---|---|
| Train | 11,044 |
| Validation | 1,053 |
| Test (real videos only) | 763 |

> The dataset has **no labels for correct/incorrect form**, so the system recognises the exercise and gives rule-based posture feedback rather than a learned form score.

## Results

Test set = real videos only (763 windows). Models trained for up to 60 epochs with early stopping and class weighting.

| Model | Test accuracy | Macro F1 |
|---|---|---|
| **1D-CNN** | **93.3%** | **0.927** |
| Random Forest (baseline) | 93.1% | 0.919 |
| LSTM | 90.0% | 0.885 |

Per-class results for the 1D-CNN:

| Exercise | Precision | Recall | F1 |
|---|---|---|---|
| Bicep Curl | 0.78 | 0.95 | 0.86 |
| Push-up | 0.97 | 1.00 | 0.98 |
| Shoulder Press | 0.99 | 0.93 | 0.96 |
| Squat | 0.94 | 0.88 | 0.91 |

Notes:
- The 1D-CNN and Random Forest are within 0.2 percentage points, which is not a meaningful difference on a test set of this size. Both are clearly ahead of the LSTM.
- Validation accuracy (about 98%) is higher than test accuracy because validation includes synthetic videos while the test set is real-only. The test figure is the more realistic one.
- Bicep curl is the hardest class. The confusion matrices are in `results/`.
- These figures are **offline test results**. Accuracy with a live webcam depends on camera angle, lighting and the person, and has not been measured here.

The live app uses the 1D-CNN by default.

## Project Structure

```
fitness-ai/
├── config.py                   # Global settings (paths, sequence length, languages, thresholds)
├── requirements.txt
├── README.md
├── app/
│   └── streamlit_app.py        # Browser interface (live camera, video upload, history)
├── src/
│   ├── pose_module.py          # MediaPipe wrapper and skeleton drawing
│   ├── angle_utils.py          # Angle calculations
│   ├── features.py             # Normalisation, joint angles, mirror augmentation
│   ├── extract_keypoints.py    # Videos -> keypoints (.npy) + manifest.csv
│   ├── preprocess.py           # Windowing and per-video train/val/test split
│   ├── train.py                # Random Forest, 1D-CNN and LSTM training and evaluation
│   ├── rep_counter.py          # Repetition counting
│   ├── exercises.py            # Per-exercise thresholds, posture rules, feedback messages
│   ├── engine.py               # Core logic: one frame in, exercise + reps + feedback out
│   ├── video_analysis.py       # Analysis and report for uploaded videos
│   ├── i18n.py                 # All user-facing text in English and Roman Urdu
│   ├── voice.py                # Voice feedback (pyttsx3, or `say` on macOS)
│   ├── session_log.py          # Session history (CSV)
│   └── realtime_app.py         # OpenCV window application
├── models/                     # Trained models (cnn1d.keras, lstm.keras, rf.pkl, labels.json)
├── results/                    # Confusion matrices, training curves, model_comparison.csv
├── sessions/                   # Saved session history
└── tests/
    └── test_pipeline.py        # Tests on synthetic data (no dataset or camera needed)
```

## Installation

Requirements: **Python 3.10 – 3.12**, a webcam (for live mode), and about 2 GB of free disk space.

```bash
git clone <your-repository-url>
cd fitness-ai

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

The trained models were produced with TensorFlow 2.21.0. Using the same version avoids model-loading problems:

```bash
pip install tensorflow==2.21.0
```

Place the trained files in `models/`: `cnn1d.keras` and `labels.json` are required (`lstm.keras` and `rf.pkl` are optional). The MediaPipe pose model (`pose_landmarker_full.task`) is downloaded automatically on first run if it is not present.

## Usage

### Browser interface

```bash
streamlit run app/streamlit_app.py
```

| Tab | What it does |
|---|---|
| **Live camera** | Turn the camera on and exercise. Shows the exercise, reps, angle and feedback. |
| **Upload video** | Upload a clip and click *Analyse video* to get the exercise, rep count, per-rep results and an angle chart. |
| **History** | Past sessions and total reps per exercise. |

The sidebar lets you choose the language (English / Roman Urdu), automatic or manual exercise selection, the model, the camera number, and voice on/off. Voice feedback is always spoken in English.

### OpenCV window

```bash
python -m src.realtime_app                       # webcam, 1D-CNN, default language
python -m src.realtime_app --lang en             # English on-screen text
python -m src.realtime_app --model lstm          # use the LSTM instead
python -m src.realtime_app --source video.mp4    # run on a video file
python -m src.realtime_app --exercise squat      # skip the classifier (test counting/feedback)
python -m src.realtime_app --no-voice            # mute
```
Press `q` to quit.

### Voice test

```bash
python -m src.voice
```

### Tips for best results

- Film from the **side** for squat, push-up and bicep curl; from the **front** for shoulder press.
- Keep the **whole body** in frame, about 2–3 metres from the camera, with good lighting and no other people in view.

## Training the Models (Kaggle)

Training is done on Kaggle with the dataset attached. Keypoint extraction is CPU-bound; training a model takes only seconds to minutes.

```bash
# 1. Install extra packages
pip install -q mediapipe tqdm

# 2. Extract keypoints (add --limit 3 first as a quick test)
python -m src.extract_keypoints --data <path-to-dataset> --out /kaggle/working/kp

# 3. Train Random Forest, 1D-CNN and LSTM, and evaluate on the real-video test set
python -m src.train --kp /kaggle/working/kp --epochs 60
```

Outputs: `models/` (trained models and `labels.json`) and `results/` (confusion matrices, training curves, `model_comparison.csv`, `split_manifest.csv`). Download them and copy them into the project folders of the same names.

`extract_keypoints` identifies exercises from folder or file names (keywords such as *squat*, *push*, *curl*, *press*), so it works with the dataset's nested folder layout. Folders containing `hammer` or `my_test_video` are skipped by default (`DEFAULT_EXCLUDE` in `config.py`).

## Configuration and Tuning

| File | Setting | Purpose |
|---|---|---|
| `config.py` | `SEQ_LEN`, `FRAME_STRIDE`, `SEQ_STRIDE` | Window length and sampling |
| `config.py` | `MIN_CONFIDENCE`, `PRED_SMOOTHING` | How certain the model must be, and prediction smoothing |
| `config.py` | `MIN_VISIBILITY`, `LOST_FRAMES` | When to decide that nobody is in view |
| `config.py` | `FEEDBACK_LANG`, `VOICE_LANG` | On-screen language (`en` / `ur`) and voice language |
| `src/exercises.py` | `SETTINGS` | Rep-counting thresholds per exercise |
| `src/exercises.py` | `live_check`, `rep_check` | Posture rules |
| `src/i18n.py` | `STRINGS` | All interface text |

The posture and rep thresholds are heuristic starting points. The live app shows the current joint angle on screen, so you can tune `low` and `high` for your own camera position.

If the system sees a person in an empty room, raise `MIN_VISIBILITY` (for example to 0.6 or 0.7).

## Testing

```bash
python -m tests.test_pipeline
```

The tests use synthetic skeleton data, so no dataset or camera is required. They cover angle calculation, features, rep counting, feedback rules, preprocessing (including the no-leakage split) and the training script.

## Limitations

- **No form-quality labels.** The dataset only labels *which* exercise is performed, not whether the form is correct. Posture feedback is therefore rule-based, not learned.
- **Synthetic-to-real gap.** A large part of the training data is synthetic avatar video; accuracy on real people is lower than on validation data.
- **Camera angle.** Joint angles and rep thresholds depend on the viewing angle. Side-on filming is recommended for most exercises.
- **Small, noisy test set.** The test set is made of a limited number of real videos; small accuracy differences between models are not conclusive.
- **Live accuracy not measured.** Reported figures are from offline tests, not from live webcam sessions.
- **Single person, four exercises.** The system supports one person in view and the four exercises above.
- **Pose model limits.** MediaPipe Pose tracks only three points per hand, and false detections are possible in poor lighting.

## Future Work

- Collect form-quality labels (correct / incorrect, with the type of mistake) and train a learned form-scoring model.
- Record additional real videos from different cameras and angles, and fine-tune on them.
- Support more exercises and automatic camera-angle detection.
- Personalised coaching summaries across sessions.
- Mobile deployment (on-device inference).

## Troubleshooting

| Problem | Solution |
|---|---|
| Model fails to load | Install the same TensorFlow version used for training (`pip install tensorflow==2.21.0`). |
| No voice | Run `python -m src.voice`. Install `pyttsx3`; on Linux also `sudo apt install espeak`. |
| "Body not detected" | Improve lighting and keep the whole body in frame. |
| Camera does not open | Try `--source 1` (or another camera number in the sidebar) and close other apps using the camera. |
| Wrong exercise name | Film from the side, keep the whole body in view, or select the exercise manually. |
| Reps are over- or under-counted | Tune the thresholds in `src/exercises.py` (`SETTINGS`). |
| False detection in an empty room | Increase `MIN_VISIBILITY` in `config.py`. |

## Acknowledgements

- **Dataset:** [Real-Time Exercise Recognition Dataset](https://www.kaggle.com/datasets/riccardoriccio/real-time-exercise-recognition-dataset) by Riccardo Riccio, which combines the Kaggle Workout/Exercises Video Dataset, the InfiniteRep synthetic dataset, and stock videos.
- **Pose estimation:** [MediaPipe Pose](https://developers.google.com/mediapipe) by Google.
- Built with Python, OpenCV, TensorFlow/Keras, scikit-learn, NumPy, Pandas and Streamlit.

## License

`<Choose a license, e.g. MIT, and add a LICENSE file>`
