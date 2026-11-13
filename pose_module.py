"""MediaPipe Pose ka wrapper (pretrained body detection).

MediaPipe ke do API hain, dono support hain:
  * purana  : mp.solutions.pose          (mediapipe <= ~0.10.21)
  * naya    : Tasks API PoseLandmarker   (naye versions; model file pehli baar khud download hoti hai)
Jo available ho wohi khud chun leta hai. Dono se output ek jaisa milta hai.
"""
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_full.task"

# MediaPipe Pose ke standard skeleton connections
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


# Sirf asli jode: kandhe, kuhniyan, kalaiyan, hip, ghutne, takhne (+ naak)
BODY_POINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
BODY_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
                    (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]


def _has_legacy():
    return hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")


def _ensure_model():
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print("Pose model download ho raha hai (sirf pehli baar)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return str(MODEL_PATH)


class PoseDetector:
    def __init__(self, complexity=1, det_conf=0.5, track_conf=0.5):
        self.complexity, self.det_conf, self.track_conf = complexity, det_conf, track_conf
        self.legacy = _has_legacy()
        self._ts = 0
        self._build()

    def _build(self):
        if self.legacy:
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False, model_complexity=self.complexity,
                min_detection_confidence=self.det_conf, min_tracking_confidence=self.track_conf)
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            opts = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=_ensure_model()),
                running_mode=vision.RunningMode.VIDEO, num_poses=1,
                min_pose_detection_confidence=self.det_conf,
                min_tracking_confidence=self.track_conf)
            self.pose = vision.PoseLandmarker.create_from_options(opts)
            self._ts = 0

    def reset_tracking(self):
        """Nayi video shuru karne se pehle call karein (pichli video ka tracking khatam)."""
        self.close()
        self._build()

    def detect(self, frame_bgr):
        """Return (kp, kp): kp = (33,4) array [x,y,z,visibility] ya None."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.legacy:
            res = self.pose.process(rgb)
            lms = res.pose_landmarks.landmark if res.pose_landmarks else None
        else:
            self._ts += 33                                  # ms, hamesha barhta hua
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            res = self.pose.detect_for_video(img, self._ts)
            lms = res.pose_landmarks[0] if res.pose_landmarks else None
        if lms is None:
            return None, None
        kp = np.array([[l.x, l.y, l.z, getattr(l, "visibility", 1.0) or 0.0] for l in lms],
                      dtype=np.float32)
        return kp, kp

    @staticmethod
    def draw(frame, kp, bad_points=(), full=False):
        """Skeleton draw karta hai. Default: sirf asli jode (chehra aur ungliyan nahi).
        bad_points (landmark index) red dikhte hain."""
        if kp is None:
            return frame
        h, w = frame.shape[:2]
        pt = lambda i: (int(kp[i, 0] * w), int(kp[i, 1] * h))
        conns = POSE_CONNECTIONS if full else BODY_CONNECTIONS
        points = range(33) if full else BODY_POINTS
        for a, b in conns:
            if kp[a, 3] > 0.3 and kp[b, 3] > 0.3:
                cv2.line(frame, pt(a), pt(b), (255, 255, 255), 3, cv2.LINE_AA)
        for i in points:
            if kp[i, 3] > 0.3:
                cv2.circle(frame, pt(i), 5, (0, 220, 0), -1, cv2.LINE_AA)
        for i in bad_points:
            cv2.circle(frame, pt(i), 10, (0, 0, 255), -1, cv2.LINE_AA)
        return frame

    def close(self):
        try:
            self.pose.close()
        except Exception:
            pass