"""Saara user-facing text ek jagah. Har cheez ke do version: proper English ("en") aur proper Roman Urdu ("ur").

Istemal:  t("reps", "ur")  ->  "Dohraav"
          t("d_reps", "en", n=5)  ->  "Reps: 5"
"""

STRINGS = {
    # ---------- Overlay (video par likha text) ----------
    "exercise": {"en": "Exercise", "ur": "Warzish"},
    "reps": {"en": "Reps", "ur": "Dohraav"},
    "angle": {"en": "Angle", "ur": "Zaviya"},
    "unknown": {"en": "Unknown", "ur": "Na-maloom"},
    "ov_no_body": {"en": "Body not detected - step into the camera view",
                   "ur": "Jism nazar nahi aaya - camera ke saamne aayein"},
    "ov_session_reps": {"en": "Reps this session: {n}", "ur": "Is session ke dohraav: {n}"},

    # ---------- Streamlit: sidebar ----------
    "settings": {"en": "Settings", "ur": "Tarteebaat"},
    "language": {"en": "Language / Zubaan", "ur": "Language / Zubaan"},
    "detect_mode": {"en": "Exercise detection", "ur": "Warzish ki pehchan"},
    "mode_auto": {"en": "Automatic (AI model)", "ur": "Khud ba khud (AI model)"},
    "mode_manual": {"en": "Manual", "ur": "Haath se chunein"},
    "mode_help": {"en": "Automatic: the trained model recognises the exercise. Manual: you choose it yourself.",
                  "ur": "Khud ba khud: trained model warzish pehchanta hai. Haath se chunein: aap khud warzish chunte hain."},
    "model": {"en": "Model", "ur": "Model"},
    "live_settings": {"en": "For the live camera", "ur": "Live camera ke liye"},
    "camera_no": {"en": "Camera number", "ur": "Camera ka number"},
    "voice": {"en": "Voice feedback", "ur": "Awaaz mein hidayat"},
    "voice_test": {"en": "🔊 Test voice", "ur": "🔊 Awaaz ka test"},

    # ---------- Streamlit: main ----------
    "app_caption": {"en": "Real-time exercise recognition, rep counting and posture feedback",
                    "ur": "Warzish ki pehchan, dohraav ki ginti aur jism ke andaaz par live mashwara"},
    "tab_live": {"en": "📷 Live camera", "ur": "📷 Live camera"},
    "tab_video": {"en": "🎞️ Upload video", "ur": "🎞️ Video upload karein"},
    "tab_hist": {"en": "📈 History", "ur": "📈 Purana record"},
    "camera_on": {"en": "📷 Camera ON", "ur": "📷 Camera chalayein"},
    "camera_hint": {"en": "Turn the camera on above. Stand side-on, keep your whole body in view, "
                          "and stay about 2-3 metres from the camera.",
                    "ur": "Upar se camera chalayein. Side se khade hon, poora jism nazar aaye, "
                          "aur camera se 2-3 meter door rahein."},
    "model_error": {"en": "Could not load the model: {e}", "ur": "Model load nahi ho saka: {e}"},
    "camera_error": {"en": "Could not open the camera. Try another camera number, or close any other app using it.",
                     "ur": "Camera nahi khul saka. Camera ka dusra number try karein, ya jo dusri app camera "
                           "istemal kar rahi ho use band karein."},
    "frame_error": {"en": "No picture received from the camera.", "ur": "Camera se tasveer nahi mili."},
    "confidence": {"en": "{p}% confidence", "ur": "{p}% yaqeen"},
    "no_body_warn": {"en": "Body not detected. Keep your whole body in view.",
                     "ur": "Jism nazar nahi aa raha. Poora jism camera mein layein."},
    "posture_ok": {"en": "Posture looks fine", "ur": "Jism ka andaaz theek hai"},
    "start_exercise": {"en": "Start exercising", "ur": "Warzish shuru karein"},
    "session_total": {"en": "Total reps this session", "ur": "Is session ke kul dohraav"},

    # ---------- Streamlit: history ----------
    "hist_empty": {"en": "No sessions saved yet. Use the live camera or analyse a video and the reps will be saved here.",
                   "ur": "Abhi koi session mehfooz nahi hua. Live camera chalayein ya video ka tajziya karein, "
                         "dohraav yahan mehfooz ho jayenge."},
    "hist_by_ex": {"en": "Total reps by exercise", "ur": "Warzish ke hisaab se kul dohraav"},
    "hist_sessions": {"en": "Past sessions", "ur": "Pichle sessions"},

    # ---------- Streamlit: video upload ----------
    "video_title": {"en": "Upload an exercise video", "ur": "Warzish ki video upload karein"},
    "video_caption": {"en": "Film from the side with your whole body in view. A short video (30-60 seconds) "
                            "is best because every frame is processed.",
                      "ur": "Video side se banayein aur poora jism nazar aaye. Chhoti video (30 se 60 second) "
                            "behtar hai, kyunke har frame process hota hai."},
    "choose_video": {"en": "Choose a video", "ur": "Video chunein"},
    "analyze": {"en": "🔍 Analyse video", "ur": "🔍 Video ka tajziya karein"},
    "processing": {"en": "Processing the video...", "ur": "Video process ho rahi hai..."},
    "analysis_failed": {"en": "Analysis failed: {e}", "ur": "Tajziya nahi ho saka: {e}"},
    "result": {"en": "Result", "ur": "Nateeja"},
    "video_length": {"en": "Video length", "ur": "Video ki lambai"},
    "n_seconds": {"en": "{n} sec", "ur": "{n} second"},
    "body_found": {"en": "Body detected", "ur": "Jism nazar aaya"},
    "p_frames": {"en": "{p}% of frames", "ur": "{p}% frames mein"},
    "per_rep": {"en": "Result of each rep", "ur": "Har dohraav ka nateeja"},
    "col_rep": {"en": "Rep", "ur": "Dohraav"},
    "col_result": {"en": "Result", "ur": "Nateeja"},
    "good": {"en": "Good", "ur": "Theek"},
    "col_min": {"en": "Lowest angle", "ur": "Sab se kam zaviya"},
    "col_max": {"en": "Highest angle", "ur": "Sab se zyada zaviya"},
    "angle_chart": {"en": "Angle over time", "ur": "Zaviya (waqt ke saath)"},
    "angle_caption": {"en": "Each wave is one rep. The lower the angle, the deeper the movement.",
                      "ur": "Har lehar ek dohraav hai. Zaviya jitna chhota, harkat utni gehri."},
    "save_hist": {"en": "Save to history", "ur": "Purane record mein mehfooz karein"},
    "saved": {"en": "Saved.", "ur": "Mehfooz ho gaya."},
    "nothing_to_save": {"en": "There are no reps to save.", "ur": "Mehfooz karne ke liye koi dohraav nahi."},

    # ---------- Video report (describe) ----------
    "d_open_fail": {"en": "The video could not be opened. Please use an mp4, mov or avi file.",
                    "ur": "Video nahi khul saki. Meharbani karke mp4, mov ya avi file istemal karein."},
    "d_no_person": {"en": "No body was detected in this video. Your whole body should be in frame and the lighting should be good.",
                    "ur": "Is video mein jism nazar nahi aaya. Poora jism frame mein hona chahiye aur roshni achhi honi chahiye."},
    "d_no_exercise": {"en": "The exercise could not be recognised. Try a side-view video that shows your whole body, "
                            "or choose the exercise yourself in Manual mode.",
                      "ur": "Warzish pehchani nahi ja saki. Side se banayi hui, poore jism wali video try karein, "
                            "ya 'Haath se chunein' mode mein warzish khud chunein."},
    "d_exercise": {"en": "Exercise: {name} (recognised in {p}% of the video)",
                   "ur": "Warzish: {name} (video ke {p}% hissay mein pehchani gayi)"},
    "d_reps": {"en": "Reps: {n}", "ur": "Dohraav: {n}"},
    "d_others": {"en": "Note: the model sometimes thought it saw other exercises too. Their reps: {txt}. "
                       "The rep count may be slightly off.",
                 "ur": "Note: model ne kabhi kabhi dusri warzishein bhi samjhin. Un ke dohraav: {txt}. "
                       "Dohraav ki ginti thori kam ya zyada ho sakti hai."},
    "d_form": {"en": "Form: {good} good reps, {bad} with a mistake",
               "ur": "Tareeqa: {good} dohraav theek, {bad} mein ghalti"},
    "d_form_item": {"en": "{n} reps: \"{msg}\"", "ur": "{n} dohraav mein: \"{msg}\""},
    "d_norep": {"en": "No complete rep was counted. The video may be too short, or the angle thresholds "
                      "(src/exercises.py) may need tuning for this camera angle.",
                "ur": "Koi mukammal dohraav nahi gina gaya. Ho sakta hai video chhoti ho, ya is camera angle ke liye "
                      "src/exercises.py ke zaviye tune karne parein."},
    "d_posture": {"en": "Posture: \"{msg}\" (during {p}% of the video)",
                  "ur": "Jism ka andaaz: \"{msg}\" (video ke {p}% hissay mein)"},
    "d_good": {"en": "Your form looks fine.", "ur": "Tareeqa theek lag raha hai."},
    "d_caveat": {"en": "This result is based on angle rules and can be off if the camera is not side-on.",
                 "ur": "Yeh nateeja zaviyon ke usoolon par hai, aur camera side se na ho to kam ya zyada ho sakta hai."},

    # ---------- Console (realtime_app) ----------
    "c_started": {"en": "Started. Press 'q' to quit.", "ur": "Shuru ho gaya. Band karne ke liye 'q' dabayein."},
    "c_cam_fail": {"en": "Could not open the camera or video.", "ur": "Camera ya video nahi khul saka."},
    "c_no_model": {"en": "{path} not found. Train the model first (python -m src.train).",
                   "ur": "{path} nahi mila. Pehle model train karein (python -m src.train)."},
    "c_summary": {"en": "Session summary: {txt}", "ur": "Session ka khulasa: {txt}"},
    "c_no_reps": {"en": "no reps", "ur": "koi dohraav nahi"},
}


def t(key, lang="ur", **kw):
    """Text lo. lang = 'en' ya 'ur'. {placeholders} kw se bharte hain."""
    entry = STRINGS[key]
    s = entry.get(lang) or entry["en"]
    return s.format(**kw) if kw else s