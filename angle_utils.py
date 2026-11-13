import numpy as np


def calculate_angle(a, b, c):
    """b par bana angle (degrees) a-b-c ke beech. Sirf x,y use hote hain."""
    a, b, c = np.asarray(a)[:2], np.asarray(b)[:2], np.asarray(c)[:2]
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-9:
        return 0.0
    cos = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def angle_from_vertical(p_top, p_bottom):
    """p_bottom se p_top tak ki line vertical se kitne degrees hati hui hai."""
    v = np.asarray(p_top)[:2] - np.asarray(p_bottom)[:2]
    n = np.linalg.norm(v)
    if n < 1e-9:
        return 0.0
    cos = np.clip(abs(v[1]) / n, -1.0, 1.0)  # image mein y neeche ki taraf barhta hai
    return float(np.degrees(np.arccos(cos)))
