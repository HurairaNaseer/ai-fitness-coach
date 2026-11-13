"""Angle ke upar-neeche hone se reps ginta hai (koi ML model nahi)."""
from collections import deque


class RepCounter:
    """Do thresholds: angle 'low' se neeche gaya, phir 'high' se upar aaya = 1 rep.

    Squat/Push-up/Curl: seedha (high) -> mura (low) -> seedha (high) = 1 rep.
    Shoulder press: kuhni mura (low) -> haath upar (high) = 1 rep.
    """

    def __init__(self, low, high, smooth=5):
        self.low, self.high = low, high
        self.buf = deque(maxlen=smooth)
        self.reps = 0
        self.state = None            # None / "low" / "high"
        self.angle = None            # smooth kiya hua angle
        self._min = self._max = None  # is rep ke dauran sab se kam / zyada angle

    def reset(self):
        self.__init__(self.low, self.high, self.buf.maxlen)

    def pause(self):
        """Body gayab hone par: smoothing/state saaf, lekin reps (count) barkarar."""
        self.buf.clear()
        self.state = None
        self.angle = None
        self._min = self._max = None

    def update(self, raw_angle):
        """Naya angle do. Rep poora hone par {'min','max'} dict, warna None."""
        self.buf.append(raw_angle)
        a = sum(self.buf) / len(self.buf)
        self.angle = a
        self._min = a if self._min is None else min(self._min, a)
        self._max = a if self._max is None else max(self._max, a)

        event = None
        if a < self.low:
            self.state = "low"
        elif a > self.high and self.state == "low":
            self.state = "high"
            self.reps += 1
            event = {"min": self._min, "max": self._max, "rep": self.reps}
            self._min = self._max = a          # agli rep ke liye reset
        elif a > self.high and self.state is None:
            self.state = "high"
        return event