"""Upload ki hui exercise video ka analysis: exercise, reps, aur form ki report."""
from collections import Counter

import cv2

import config
from src import exercises as ex_rules
from src.i18n import t


def analyze_video(path, engine, progress_cb=None, frame_cb=None, max_width=640, preview_every=6,
                  lang=config.FEEDBACK_LANG):
    """Video ke har frame par engine chalata hai. Return report (dict).

    progress_cb(0..1) aur frame_cb(annotated_frame, state) optional hain (UI ke liye).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(t("d_open_fail", lang))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    exercise_frames, person_frames, timeline = Counter(), 0, []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w > max_width:
            frame = cv2.resize(frame, (max_width, int(h * max_width / w)))
        out = engine.process(frame)
        s = engine.state()
        if s["present"]:
            person_frames += 1
            if s["exercise"]:
                exercise_frames[s["exercise"]] += 1
        if s["angle"] is not None:
            timeline.append((i / fps, float(s["angle"])))
        if frame_cb and i % preview_every == 0:
            frame_cb(out, s)
        if progress_cb and total:
            progress_cb(min(1.0, (i + 1) / total))
        i += 1
    cap.release()

    main = exercise_frames.most_common(1)[0] if exercise_frames else (None, 0)
    return {
        "frames": i,
        "fps": fps,
        "duration_s": i / fps if fps else 0.0,
        "person_frames": person_frames,
        "exercise_frames": dict(exercise_frames),
        "main_exercise": main[0],
        "main_share": (main[1] / person_frames) if person_frames else 0.0,
        "reps_by_exercise": {k: c.reps for k, c in engine.counters.items()},
        "rep_log": list(engine.rep_log),
        "issue_frames": dict(engine.issue_frames),
        "active_frames": engine.active_frames,
        "timeline": timeline,
    }


def describe(report, lang=config.FEEDBACK_LANG):
    """Report ko insaani zubaan (English ya Roman Urdu) ki lines mein badalta hai.
    Jin lines ki shuruaat "  - " se ho woh upar wali line ki zayli (sub) lines hain."""
    if report["person_frames"] == 0:
        return [t("d_no_person", lang)]
    ex = report["main_exercise"]
    if not ex:
        return [t("d_no_exercise", lang)]

    lines = [t("d_exercise", lang, name=config.DISPLAY_NAMES[ex], p=f"{report['main_share'] * 100:.0f}")]
    reps = report["reps_by_exercise"].get(ex, 0)
    lines.append(t("d_reps", lang, n=reps))

    others = {k: v for k, v in report["reps_by_exercise"].items() if k != ex and v > 0}
    if others:
        txt = ", ".join(f"{config.DISPLAY_NAMES[k]} ({v})" for k, v in others.items())
        lines.append(t("d_others", lang, txt=txt))

    rep_log = [r for r in report["rep_log"] if r["exercise"] == ex]
    bad = [r for r in rep_log if r["result"] != "good"]
    if rep_log:
        lines.append(t("d_form", lang, good=len(rep_log) - len(bad), bad=len(bad)))
        for msg_id, n in Counter(r["result"] for r in bad).most_common():
            lines.append("  - " + t("d_form_item", lang, n=n, msg=ex_rules.message(msg_id, lang)))
    elif reps == 0:
        lines.append(t("d_norep", lang))

    active = max(report["active_frames"], 1)
    flagged = False
    for msg_id, n in sorted(report["issue_frames"].items(), key=lambda x: -x[1]):
        pct = 100 * n / active
        if pct >= 5:
            flagged = True
            lines.append(t("d_posture", lang, msg=ex_rules.message(msg_id, lang), p=f"{pct:.0f}"))

    if rep_log and not bad and not flagged:
        lines.append(t("d_good", lang))
    lines.append(t("d_caveat", lang))
    return lines