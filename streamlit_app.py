"""Browser frontend: live camera, video upload analysis, aur session history.

Chalane ka tareeqa (project ke root folder se):
    streamlit run app/streamlit_app.py
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import pandas as pd
import streamlit as st

import config
from src import exercises as ex_rules
from src import session_log
from src.engine import CoachEngine
from src.i18n import t
from src.pose_module import PoseDetector
from src.video_analysis import analyze_video, describe
from src.voice import Voice

st.set_page_config(page_title="AI Fitness Coach", page_icon="🏋️", layout="wide")


@st.cache_resource(show_spinner=False)
def load_model(name):
    from tensorflow import keras
    model = keras.models.load_model(config.MODELS_DIR / f"{name}.keras")
    labels = json.loads((config.MODELS_DIR / "labels.json").read_text())
    return model, labels


@st.cache_resource(show_spinner=False)
def load_detector():
    return PoseDetector(complexity=1)


@st.cache_resource
def get_voice():
    return Voice(enabled=True)


def make_engine(forced, model_name, voice, lang):
    """CoachEngine banata hai. Automatic mode mein trained model load hota hai."""
    predict_fn = labels = None
    if forced is None:
        model, labels = load_model(model_name)
        predict_fn = lambda x: model(x, training=False).numpy()[0]
    detector = load_detector()
    detector.reset_tracking()
    return CoachEngine(detector, predict_fn, labels, forced=forced, voice=voice, lang=lang)


def show_report(report, lang):
    if report["person_frames"] == 0:
        st.error(describe(report, lang)[0])
        return
    ex = report["main_exercise"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("exercise", lang), config.DISPLAY_NAMES.get(ex, t("unknown", lang)))
    c2.metric(t("reps", lang), report["reps_by_exercise"].get(ex, 0) if ex else 0)
    c3.metric(t("video_length", lang), t("n_seconds", lang, n=f"{report['duration_s']:.0f}"))
    c4.metric(t("body_found", lang),
              t("p_frames", lang, p=f"{100 * report['person_frames'] / max(report['frames'], 1):.0f}"))

    st.subheader(t("result", lang))
    for line in describe(report, lang):
        if line.startswith("  - "):
            st.markdown("&nbsp;&nbsp;&nbsp;• " + line[4:])
        else:
            st.markdown(line)

    rep_rows = [{t("col_rep", lang): r["rep"],
                 t("exercise", lang): config.DISPLAY_NAMES[r["exercise"]],
                 t("col_result", lang): t("good", lang) if r["result"] == "good" else ex_rules.message(r["result"], lang),
                 t("col_min", lang): r["min"],
                 t("col_max", lang): r["max"]}
                for r in report["rep_log"]]
    if rep_rows:
        st.subheader(t("per_rep", lang))
        st.dataframe(pd.DataFrame(rep_rows), use_container_width=True, hide_index=True)

    if report["timeline"]:
        st.subheader(t("angle_chart", lang))
        df = pd.DataFrame(report["timeline"], columns=["second", t("angle", lang)]).set_index("second")
        st.line_chart(df)
        st.caption(t("angle_caption", lang))

    if st.button(t("save_hist", lang)):
        saved = session_log.append_summary(report["reps_by_exercise"])
        if saved:
            st.success(t("saved", lang))
        else:
            st.info(t("nothing_to_save", lang))


# ---------------- Sidebar ----------------
lang = st.sidebar.selectbox("Language / Zubaan", ["en", "ur"],
                            index=0 if config.FEEDBACK_LANG == "en" else 1,
                            format_func=lambda x: "English" if x == "en" else "Roman Urdu")
st.sidebar.title(t("settings", lang))
mode = st.sidebar.radio(t("detect_mode", lang), ["auto", "manual"],
                        format_func=lambda m: t("mode_" + m, lang), help=t("mode_help", lang))
if mode == "auto":
    model_name = st.sidebar.selectbox(t("model", lang), ["cnn1d", "lstm"])
    forced = None
else:
    model_name = None
    forced = st.sidebar.selectbox(t("exercise", lang), list(ex_rules.SETTINGS),
                                  format_func=lambda k: config.DISPLAY_NAMES[k])
st.sidebar.markdown("**" + t("live_settings", lang) + "**")
cam_index = st.sidebar.number_input(t("camera_no", lang), min_value=0, max_value=5, value=0, step=1)
voice_on = st.sidebar.toggle(t("voice", lang), value=True)
if st.sidebar.button(t("voice_test", lang)):
    get_voice().say("Voice is working")

st.title("🏋️ AI Fitness Coach")
st.caption(t("app_caption", lang))

tab_live, tab_video, tab_hist = st.tabs([t("tab_live", lang), t("tab_video", lang), t("tab_hist", lang)])

# ---------------- History ----------------
with tab_hist:
    df = session_log.load()
    if df.empty:
        st.info(t("hist_empty", lang))
    else:
        df["exercise"] = df["exercise"].map(lambda k: config.DISPLAY_NAMES.get(k, k))
        st.subheader(t("hist_by_ex", lang))
        st.bar_chart(df.groupby("exercise")["reps"].sum())
        st.subheader(t("hist_sessions", lang))
        st.dataframe(df.sort_values("time", ascending=False), use_container_width=True)

# ---------------- Video upload ----------------
with tab_video:
    st.subheader(t("video_title", lang))
    st.caption(t("video_caption", lang))
    up = st.file_uploader(t("choose_video", lang), type=["mp4", "mov", "avi", "mkv", "webm"])
    if up is not None:
        st.video(up)
        if st.button(t("analyze", lang), type="primary"):
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix or ".mp4") as tmp:
                    tmp.write(up.getbuffer())
                    tmp_path = tmp.name
                engine = make_engine(forced, model_name, None, lang)
                bar = st.progress(0.0, text=t("processing", lang))
                preview = st.empty()
                st.session_state["report"] = analyze_video(
                    tmp_path, engine, lang=lang,
                    progress_cb=lambda p: bar.progress(p, text=t("processing", lang)),
                    frame_cb=lambda frame, s: preview.image(frame, channels="BGR"))
                bar.empty()
                preview.empty()
            except Exception as e:
                st.error(t("analysis_failed", lang, e=e))
            finally:
                if tmp_path:
                    Path(tmp_path).unlink(missing_ok=True)
    if st.session_state.get("report"):
        show_report(st.session_state["report"], lang)

# ---------------- Live camera (sab se aakhir mein, kyunke loop chalta rehta hai) ----------------
with tab_live:
    run = st.toggle(t("camera_on", lang), value=False)
    col_video, col_stats = st.columns([3, 1])
    video_ph = col_video.empty()
    with col_stats:
        ex_ph, reps_ph, angle_ph, fb_ph, total_ph = (st.empty() for _ in range(5))

    if not run:
        video_ph.info(t("camera_hint", lang))
    else:
        try:
            engine = make_engine(forced, model_name, get_voice() if voice_on else None, lang)
        except Exception as e:
            st.error(t("model_error", lang, e=e))
            st.stop()
        cap = cv2.VideoCapture(int(cam_index))
        if not cap.isOpened():
            st.error(t("camera_error", lang))
            st.stop()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    st.warning(t("frame_error", lang))
                    break
                video_ph.image(engine.process(frame), channels="BGR")
                s = engine.state()
                ex_ph.metric(t("exercise", lang), s["name"],
                             t("confidence", lang, p=f"{s['confidence'] * 100:.0f}")
                             if forced is None and s["exercise"] else None)
                reps_ph.metric(t("reps", lang), s["reps"])
                angle_ph.metric(t("angle", lang), f"{s['angle']:.0f}°" if s["angle"] is not None else "-")
                if not s["present"]:
                    fb_ph.warning(t("no_body_warn", lang))
                elif s["feedback"]:
                    fb_ph.error(s["feedback"])
                else:
                    fb_ph.success(t("posture_ok", lang) if s["exercise"] else t("start_exercise", lang))
                total_ph.metric(t("session_total", lang), sum(s["all_reps"].values()))
        finally:
            cap.release()
            session_log.append(engine.counters)