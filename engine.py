"""CoachEngine: ek frame lo -> exercise pehchan + rep count + feedback + overlay.

Live OpenCV app (realtime_app.py) aur Streamlit frontend dono yahi use karte hain.
"""
import time
from collections import Counter, deque

import cv2
import numpy as np

import config
from src import exercises as ex_rules
from src.features import frame_features
from src.i18n import t
from src.pose_module import PoseDetector
from src.rep_counter import RepCounter

GREEN, RED, YELLOW, WHITE = (0, 220, 0), (0, 0, 255), (0, 220, 255), (255, 255, 255)
_CORE = [11, 12, 23, 24]   # kandhe aur hip: inki visibility se "insan maujood hai?" tay hota hai


def put(frame, text, org, color=WHITE, scale=0.7, thick=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


class CoachEngine:
    def __init__(self, detector, predict_fn=None, labels=None, forced=None, voice=None,
                 lang=config.FEEDBACK_LANG):
        """predict_fn(x[1,30,74]) -> probabilities (n_classes,). forced = exercise khud tay (model ke bina).
        lang = screen ki zubaan ('en'/'ur'). Awaz hamesha config.VOICE_LANG (English) mein bolti hai."""
        self.detector, self.predict_fn, self.labels = detector, predict_fn, labels
        self.forced, self.voice, self.lang = forced, voice, lang
        self.buffer = deque(maxlen=config.SEQ_LEN)
        self.votes = deque(maxlen=config.PRED_SMOOTHING)
        self.counters = {}
        self.current = forced
        self.conf = 1.0 if forced else 0.0
        self.feedback, self.feedback_until = "", 0.0
        self.frame_i, self.missing = 0, 0
        self.present, self.bad = False, []
        self.fps, self._t = 0.0, time.time()
        # Video analysis ke liye records
        self.rep_log = []              # har rep: {rep, exercise, result, min, max}
        self.issue_frames = Counter()  # posture ghalti ki id -> kitni frames mein hui
        self.active_frames = 0         # jin frames mein exercise pehchani hui thi
        self._announced = None         # jis exercise ka naam awaz mein bol chuke hain

    # ---------- helpers ----------
    @staticmethod
    def _person_visible(kp):
        return kp is not None and float(kp[_CORE, 3].mean()) >= config.MIN_VISIBILITY

    def _lost(self):
        """Body kuch der se nahi mili: purani cheezein saaf (reps barkarar)."""
        self.buffer.clear()
        self.votes.clear()
        if not self.forced:
            self.current, self.conf = None, 0.0
        for c in self.counters.values():
            c.pause()
        self.feedback = ""
        self.bad = []
        self._announced = None

    def _classify(self, kp):
        if self.forced:
            return
        if self.frame_i % config.FRAME_STRIDE == 0:        # training jaisi frame rate
            self.buffer.append(frame_features(kp))
        if len(self.buffer) == config.SEQ_LEN and self.frame_i % (config.FRAME_STRIDE * 3) == 0:
            probs = np.asarray(self.predict_fn(np.array(self.buffer)[None]))
            k = int(probs.argmax())
            self.votes.append(k if probs[k] >= config.MIN_CONFIDENCE else -1)
            top, _ = Counter(self.votes).most_common(1)[0]
            self.current = config.canonical_label(self.labels[top]) if top >= 0 else None
            self.conf = float(probs[k])

    def _count_and_check(self, kp):
        if not self.current:
            return
        if self.current not in self.counters:
            s = ex_rules.SETTINGS[self.current]
            self.counters[self.current] = RepCounter(s["low"], s["high"])
        counter = self.counters[self.current]
        self.active_frames += 1
        event = counter.update(ex_rules.primary_angle(self.current, kp))
        if event:
            mid = ex_rules.rep_check(self.current, event)
            self.rep_log.append({"rep": event["rep"], "exercise": self.current, "result": mid,
                                 "min": round(event["min"]), "max": round(event["max"])})
            self._say(mid, 3.0, spoken=str(event["rep"]) if mid == "good" else None)
        msg_id, self.bad = ex_rules.live_check(self.current, kp)
        if msg_id:
            self.issue_frames[msg_id] += 1
            self._say(msg_id, 2.0)

    def _say(self, msg_id, seconds, spoken=None):
        """Screen par feedback (self.lang mein) aur awaz (hamesha English mein)."""
        self.feedback = ex_rules.message(msg_id, self.lang)
        self.feedback_until = time.time() + seconds
        if self.voice:
            self.voice.say(spoken or ex_rules.message(msg_id, config.VOICE_LANG))

    # ---------- main ----------
    def process(self, frame):
        """BGR frame lo, annotated frame wapas do."""
        self.frame_i += 1
        kp, _ = self.detector.detect(frame)
        self.present = self._person_visible(kp)
        self.bad = []

        if not self.present:
            self.missing += 1
            if self.missing >= config.LOST_FRAMES:
                self._lost()
        else:
            self.missing = 0
            self._classify(kp)
            if self.voice and self.current and self.current != self._announced:
                self._announced = self.current          # exercise ka naam ek baar bolo
                self.voice.say(config.DISPLAY_NAMES[self.current])
            self._count_and_check(kp)
            PoseDetector.draw(frame, kp, bad_points=self.bad)

        now = time.time()
        self.fps = 0.9 * self.fps + 0.1 / max(now - self._t, 1e-6)
        self._t = now
        self._overlay(frame)
        return frame

    def state(self):
        c = self.counters.get(self.current)
        return {
            "present": self.present,
            "exercise": self.current,
            "name": config.DISPLAY_NAMES[self.current] if self.current else t("unknown", self.lang),
            "confidence": self.conf,
            "reps": c.reps if c else 0,
            "angle": c.angle if c else None,
            "all_reps": {k: v.reps for k, v in self.counters.items()},
            "feedback": self.feedback if time.time() < self.feedback_until else "",
            "fps": self.fps,
        }

    def _overlay(self, frame):
        s, L = self.state(), self.lang
        w = frame.shape[1]
        cv2.rectangle(frame, (0, 0), (w, 110), (30, 30, 30), -1)
        if not s["present"]:
            put(frame, t("ov_no_body", L), (10, 30), YELLOW, 0.65)
            put(frame, t("ov_session_reps", L, n=sum(s["all_reps"].values())), (10, 62), WHITE, 0.7)
        else:
            txt = f"{t('exercise', L)}: {s['name']}"
            if not self.forced and self.current:
                txt += f" ({s['confidence'] * 100:.0f}%)"
            put(frame, txt, (10, 28), GREEN)
            if self.current:
                put(frame, f"{t('reps', L)}: {s['reps']}", (10, 62), WHITE, 0.9)
                if s["angle"] is not None:
                    put(frame, f"{t('angle', L)}: {s['angle']:.0f}", (210, 62), WHITE)
            if s["feedback"]:
                put(frame, s["feedback"], (10, 98), RED if self.bad else YELLOW, 0.8)
        put(frame, f"{s['fps']:.0f} FPS", (w - 110, 28), WHITE, 0.6, 1)