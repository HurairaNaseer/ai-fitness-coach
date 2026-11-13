# AI-Powered Real-Time Exercise Recognition, Rep Counting & Posture Feedback

Camera ke saamne exercise karein. System khud pehchanta hai ke **kaunsi exercise** ho rahi hai
(Squat, Push-up, Bicep Curl, Shoulder Press), **reps ginta hai**, aur **posture ki ghalti** batata hai.

Dataset: https://www.kaggle.com/datasets/riccardoriccio/real-time-exercise-recognition-dataset

## Kaise kaam karta hai

```
PHASE A - Training (ek baar, Kaggle/Colab par)
  Videos --MediaPipe--> keypoints --features--> 30-frame windows --> RF / 1D-CNN / LSTM train

PHASE B - Live (laptop par)
  Webcam --MediaPipe--> keypoints --> LSTM --> exercise ka naam
                              \--> angles --> rep counter + posture rules --> screen + voice
```

| Kaam | Kaun karta hai | Type |
|---|---|---|
| Body ke 33 points nikalna | MediaPipe Pose | Pretrained |
| Exercise pehchanna | LSTM / 1D-CNN (aap train karte hain) | Custom |
| Reps ginna | Angle thresholds (`src/rep_counter.py`) | Rule-based |
| Posture feedback | Angle rules (`src/exercises.py`) | Rule-based |

**Features (har frame ke 74):** 33 keypoints ke (x, y) hip-centered aur torso-length se normalized (66)
+ 8 joint angles (kuhni, kandha, hip, ghutna - left/right).
**Window:** 30 frames (har 2nd frame, ~2 second). **Split:** video ke hisab se; test sirf real videos se.

## Folder structure

```
fitness-ai/
├── config.py               # sab settings (sequence length, paths, language)
├── requirements.txt
├── src/
│   ├── pose_module.py      # MediaPipe wrapper + skeleton drawing
│   ├── angle_utils.py      # angle calculation
│   ├── features.py         # normalization + angles + mirror
│   ├── extract_keypoints.py# videos -> keypoints (.npy) + manifest.csv
│   ├── preprocess.py       # windows + train/val/test split
│   ├── train.py            # Random Forest + 1D-CNN + LSTM, evaluation, plots
│   ├── rep_counter.py      # rep counting
│   ├── exercises.py        # thresholds + posture feedback rules (yahan tune karein)
│   ├── voice.py            # voice feedback (pyttsx3)
│   └── realtime_app.py    # live app
├── tests/test_pipeline.py  # banawati data par poora test
├── data/  models/  results/  sessions/
```

## Setup

Python 3.10 - 3.12 behtar hai.
```bash
pip install -r requirements.txt
python -m tests.test_pipeline        # sab "OK" aana chahiye (dataset ki zaroorat nahi)
```

## Step 1-2: Keypoints nikalna + training (Kaggle par)

1. Kaggle notebook banayein, **Add Data** se upar wala dataset attach karein, aur **Internet ON** karein.
2. Is project ka zip bhi Kaggle Dataset ke taur par upload karke attach karein (ya notebook mein upload karein).
3. Notebook ke cells:

```python
!pip install -q mediapipe tqdm
!cp -r /kaggle/input/<aap-ka-code-dataset>/fitness-ai /kaggle/working/
%cd /kaggle/working/fitness-ai
```
```python
# pehle chhota test (har exercise ke 3 videos) taake pata chale sab theek hai
!python -m src.extract_keypoints --data /kaggle/input/real-time-exercise-recognition-dataset --out /kaggle/working/kp --limit 3
```
Output mein dekhein ke 4 exercises pehchani gayin. Phir poora chalayein (bina `--limit`):
```python
!python -m src.extract_keypoints --data /kaggle/input/real-time-exercise-recognition-dataset --out /kaggle/working/kp
!python -m src.train --kp /kaggle/working/kp --epochs 60
```
4. Training ke baad `models/` folder (`lstm.keras`, `cnn1d.keras`, `rf.pkl`, `labels.json`) aur `results/`
   (confusion matrix, loss curves, `model_comparison.csv`) download karein.

> Agar extraction mein "exercise pehchana nahi gaya" ka warning aaye, to dataset ke folder ke naam
> `config.py` ke `EXERCISE_KEYWORDS` mein add kar dein.

## Step 3: Live app (apne laptop par)

`lstm.keras` aur `labels.json` ko `models/` mein rakhein, phir:
```bash
python -m src.realtime_app                      # webcam, LSTM se exercise pehchan
python -m src.realtime_app --model cnn1d        # 1D-CNN se
python -m src.realtime_app --source video.mp4   # video file par
python -m src.realtime_app --exercise squat     # model ke bina (rep counter/feedback test karne ke liye)
python -m src.realtime_app --no-voice
```
`q` dabane par band. Session ki reps `sessions/session_log.csv` mein save hoti hain.

## Behtar accuracy ke liye

- **Camera angle:** Squat, Push-up, Curl **side view** se; Shoulder Press front se. Poora jism frame mein.
- **Apni 20-30 real videos** (har exercise ki) dataset folder mein `real/` naam ke folder ke andar daal dein.
  Dataset mein synthetic zyada hai, real kam; test hamesha real par hota hai.
- **Thresholds tune karein** (`src/exercises.py` mein `SETTINGS`): live app mein angle screen par dikhta hai,
  apni exercise ke neeche/upar wale angle dekh kar `low`/`high` set karein.
- Training mein accuracy kam aaye to epochs/`SEQ_LEN` (`config.py`) badal kar dekhein, aur confusion matrix dekhein
  ke kaunsi do exercises aapas mein mix ho rahi hain.

## Limitations (report mein honestly likhein)

- Dataset mein **form ke correct/incorrect labels nahi**, is liye posture feedback **rule-based** hai, ML-based nahi.
- Model sirf exercise **pehchanta** hai. Feedback thresholds andaze hain jo tune karne parte hain.
- Synthetic (avatar) videos par train hone se real insaan par accuracy kam ho sakti hai.
- Camera angle badalne se angles aur accuracy par asar parta hai.

## Troubleshooting

| Masla | Hal |
|---|---|
| `mediapipe` install/import error | Python 3.10-3.12 use karein, `pip install -U mediapipe` |
| Pehli baar pose model download hota hai | Internet chahiye (sirf ek baar, `models/pose_landmarker_full.task`) |
| Webcam nahi khulta | `--source 1` try karein |
| "Body detect nahi hui" | Roshni theek karein, poora jism frame mein layein |
| Naam baar baar badalta hai | `config.py` mein `PRED_SMOOTHING` badhayein |
| Voice nahi aati | `pip install pyttsx3`, ya `--no-voice` |
