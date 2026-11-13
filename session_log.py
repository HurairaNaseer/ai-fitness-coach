"""Session ki reps ko CSV mein save/load karna (history ke liye)."""
import csv
import time

import pandas as pd

import config

LOG = config.ROOT / "sessions" / "session_log.csv"


def append_summary(summary):
    """summary = {exercise: reps}. Sirf reps > 0 save hoti hain."""
    summary = {k: v for k, v in summary.items() if v > 0}
    if not summary:
        return summary
    LOG.parent.mkdir(exist_ok=True)
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "exercise", "reps"])
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        for k, reps in summary.items():
            w.writerow([now, k, reps])
    return summary


def append(counters):
    """counters = {exercise: RepCounter}. Return {exercise: reps}."""
    return append_summary({k: c.reps for k, c in counters.items()})


def load():
    if not LOG.exists():
        return pd.DataFrame(columns=["time", "exercise", "reps"])
    return pd.read_csv(LOG)