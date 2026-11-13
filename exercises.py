"""Har exercise ke rules: kaunsa angle ginna hai, thresholds, aur posture feedback.

NOTE: thresholds andaze hain. Apne videos par dekh kar tune karein
(sab ek jagah yahin likhe hain).
"""
import numpy as np

from src.angle_utils import calculate_angle, angle_from_vertical
from src.features import (L_SH, R_SH, L_EL, R_EL, L_WR, R_WR,
                          L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN)

# Landmark indices: (left, right)
SH, EL, WR, HIP, KN, AN = (L_SH, R_SH), (L_EL, R_EL), (L_WR, R_WR), (L_HIP, R_HIP), (L_KN, R_KN), (L_AN, R_AN)

# low/high = rep counting ke thresholds (degrees)
SETTINGS = {
    "squat":               {"low": 100, "high": 160},
    "push_up":             {"low": 100, "high": 150},
    "barbell_biceps_curl": {"low": 60,  "high": 140},
    "shoulder_press":      {"low": 100, "high": 150},
}

# Feedback messages: id -> (Roman Urdu, English)
MESSAGES = {
    "depth":         ("Aur neeche jayein", "Go deeper"),
    "back_lean":     ("Seena seedha rakhein", "Keep your chest up"),
    "body_line":     ("Jism ko seedha rakhein", "Keep your body straight"),
    "push_depth":    ("Seene ko zameen ke aur qareeb layein", "Lower your chest further"),
    "elbow_drift":   ("Kohni ko jism ke qareeb rakhein", "Keep your elbow close to your body"),
    "curl_partial":  ("Haath ko poora upar tak layein", "Curl all the way up"),
    "press_partial": ("Haath poore upar tak seedhe karein", "Press all the way up"),
    "press_lean":    ("Kamar ko seedha rakhein", "Keep your back straight"),
    "good":          ("Bohat khoob", "Good rep"),
}


def message(msg_id, lang="ur"):
    return MESSAGES[msg_id][0 if lang == "ur" else 1]


def pick_side(kp, ex):
    """Left ya right: jis taraf ke relevant joints zyada visible hon."""
    joints = {"squat": (SH, HIP, KN, AN), "push_up": (SH, EL, WR, HIP, AN),
              "barbell_biceps_curl": (SH, EL, WR, HIP), "shoulder_press": (SH, EL, WR, HIP)}[ex]
    left = sum(kp[j[0], 3] for j in joints)
    right = sum(kp[j[1], 3] for j in joints)
    return 0 if left >= right else 1


def primary_angle(ex, kp):
    """Rep counting ka angle (degrees)."""
    s = pick_side(kp, ex)
    if ex == "squat":
        return calculate_angle(kp[HIP[s]], kp[KN[s]], kp[AN[s]])
    return calculate_angle(kp[SH[s]], kp[EL[s]], kp[WR[s]])


def live_check(ex, kp):
    """Har frame ki posture check. Return (msg_id ya None, [ghalat landmarks])."""
    s = pick_side(kp, ex)
    sh, hip = kp[SH[s]], kp[HIP[s]]
    if ex == "squat":
        knee_angle = calculate_angle(kp[HIP[s]], kp[KN[s]], kp[AN[s]])
        if knee_angle < 140 and angle_from_vertical(sh, hip) > 65:
            return "back_lean", [SH[s], HIP[s]]
    elif ex == "push_up":
        if calculate_angle(sh, hip, kp[AN[s]]) < 150:
            return "body_line", [HIP[s]]
    elif ex == "barbell_biceps_curl":
        if calculate_angle(kp[EL[s]], sh, hip) > 40:
            return "elbow_drift", [EL[s]]
    elif ex == "shoulder_press":
        if angle_from_vertical(sh, hip) > 25:
            return "press_lean", [SH[s], HIP[s]]
    return None, []


def rep_check(ex, event):
    """Rep poori hone par us ke min/max angle se feedback. Return msg_id."""
    if ex == "squat" and event["min"] > 110:
        return "depth"
    if ex == "push_up" and event["min"] > 110:
        return "push_depth"
    if ex == "barbell_biceps_curl" and event["min"] > 70:
        return "curl_partial"
    if ex == "shoulder_press" and event["max"] < 150:
        return "press_partial"
    return "good"
