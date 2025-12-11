import numpy as np

def calc_angle(a, b, c):
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cos = np.dot(ba, bc) / denom
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def angle_message(name: str, angle: float, min_: float, max_: float) -> str:
    ok = "OK ✅" if min_ <= angle <= max_ else "⚠ Fora do ideal"
    return f"{name}: {angle:.2f}° ({ok})"