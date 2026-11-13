"""Keypoints se model ke features banana (normalization + angles)."""
import numpy as np
from src.angle_utils import calculate_angle

# MediaPipe landmark indices
NOSE = 0
L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_HIP, R_HIP = 23, 24
L_KN, R_KN = 25, 26
L_AN, R_AN = 27, 28

N_FEATURES = 33 * 2 + 8   # 66 normalized coords + 8 angles = 74

# Mirror augmentation ke liye left/right pairs
_LR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
             (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]


def mirror_keypoints(kp):
    """kp (...,33,C): left-right flip (x -> 1-x, aur L/R landmarks swap)."""
    out = kp.copy()
    out[..., 0] = 1.0 - out[..., 0]
    for a, b in _LR_PAIRS:
        out[..., [a, b], :] = out[..., [b, a], :]
    return out


def normalize_keypoints(kp):
    """Hip ko center banao, torso length se divide karo. Return (66,)."""
    xy = kp[:, :2]
    hip_c = (xy[L_HIP] + xy[R_HIP]) / 2.0
    sh_c = (xy[L_SH] + xy[R_SH]) / 2.0
    torso = np.linalg.norm(sh_c - hip_c)
    if torso < 1e-6:
        torso = 1.0
    return ((xy - hip_c) / torso).reshape(-1)


def joint_angles(kp):
    """8 angles (degrees): elbow, shoulder, hip, knee (left aur right)."""
    return np.array([
        calculate_angle(kp[L_SH], kp[L_EL], kp[L_WR]),
        calculate_angle(kp[R_SH], kp[R_EL], kp[R_WR]),
        calculate_angle(kp[L_EL], kp[L_SH], kp[L_HIP]),
        calculate_angle(kp[R_EL], kp[R_SH], kp[R_HIP]),
        calculate_angle(kp[L_SH], kp[L_HIP], kp[L_KN]),
        calculate_angle(kp[R_SH], kp[R_HIP], kp[R_KN]),
        calculate_angle(kp[L_HIP], kp[L_KN], kp[L_AN]),
        calculate_angle(kp[R_HIP], kp[R_KN], kp[R_AN]),
    ], dtype=np.float32)


def frame_features(kp):
    """Ek frame ke 74 features."""
    return np.concatenate([normalize_keypoints(kp), joint_angles(kp) / 180.0]).astype(np.float32)


def sequence_features(kp_seq):
    """kp_seq (T,33,4) -> (T,74)."""
    return np.stack([frame_features(k) for k in kp_seq])
