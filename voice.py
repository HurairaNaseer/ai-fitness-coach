"""Awaz mein feedback (pyttsx3, offline). Agar install na ho to chup chaap band rehta hai."""
import queue
import threading
import time


class Voice:
    def __init__(self, enabled=True, cooldown=4.0):
        self.enabled = enabled
        self.cooldown = cooldown
        self._last = {}
        self._q = queue.Queue(maxsize=2)
        if enabled:
            try:
                import pyttsx3  # noqa: F401
                threading.Thread(target=self._worker, daemon=True).start()
            except Exception:
                print("pyttsx3 nahi mila: voice feedback band.")
                self.enabled = False

    def _worker(self):
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        while True:
            text = self._q.get()
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass

    def say(self, text):
        """Ek hi message cooldown se pehle dobara nahi bolta."""
        if not self.enabled or not text:
            return
        now = time.time()
        if now - self._last.get(text, 0) < self.cooldown:
            return
        self._last[text] = now
        try:
            self._q.put_nowait(text)
        except queue.Full:
            pass
