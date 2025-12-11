import numpy as np
import math

def compute_velocity(history, key):
    """Retorna lista de velocidades (°/s) da articulação `key`."""
    vals = []
    for i in range(1, len(history)):
        t0 = history[i - 1]["t"]
        t1 = history[i]["t"]
        v0 = history[i - 1]["angles"].get(key)
        v1 = history[i]["angles"].get(key)
        if v0 is None or v1 is None:
            continue
        dt = t1 - t0
        if dt <= 0:
            continue
        vals.append((v1 - v0) / dt)
    return vals


def rom_of_joint(history, key):
    """Amplitude de movimento (max - min) da articulação `key`."""
    vals = [s["angles"].get(key) for s in history if key in s["angles"]]
    vals = [v for v in vals if v is not None]
    if not vals:
        return 0.0
    return max(vals) - min(vals)


def lateral_deviation_px(history, point_name):
    """Desvio padrão da coordenada X de um ponto (px)."""
    xs = []
    for s in history:
        kp = s.get("kp")
        if not kp or point_name not in kp:
            continue
        xs.append(kp[point_name][0])
    if not xs:
        return 0.0
    return float(np.std(xs))


def symmetry_measure(history, left_key, right_key):
    """Diferença média absoluta entre articulações esq/dir."""
    diffs = []
    for s in history:
        a = s["angles"]
        if left_key in a and right_key in a:
            diffs.append(abs(a[left_key] - a[right_key]))
    return float(np.mean(diffs)) if diffs else 0.0


def trajectory_straightness(history, point_name):
    """Net-distance / path-length (1 = reta perfeita)."""
    pts = []
    for s in history:
        kp = s.get("kp")
        if not kp or point_name not in kp:
            continue
        pts.append(tuple(kp[point_name]))
    if len(pts) < 3:
        return 1.0
    path_len = 0.0
    for i in range(1, len(pts)):
        path_len += math.dist(pts[i - 1], pts[i])
    net = math.dist(pts[0], pts[-1])
    if path_len == 0:
        return 1.0
    return net / path_len


def stacking_check(history, elbow_key, wrist_key):
    """Distância média (px) entre cotovelo e punho na mesma direção X."""
    el_x, wr_x = [], []
    for s in history:
        kp = s.get("kp")
        if not kp:
            continue
        if elbow_key in kp and wrist_key in kp:
            el_x.append(kp[elbow_key][0])
            wr_x.append(kp[wrist_key][0])
    if not el_x or not wr_x:
        return 0.0
    diffs = [abs(wr_x[i] - el_x[i]) for i in range(min(len(wr_x), len(el_x)))]
    return float(np.mean(diffs))


def detect_sticking_point(history, axis_key, low_ratio=0.2, time_ratio=0.15):
    """True se >15 % do tempo a velocidade < 20 % da mediana."""
    vels = compute_velocity(history, axis_key)
    if not vels:
        return False
    mag = [abs(v) for v in vels]
    median = np.median(mag)
    if median == 0:
        return False
    low_thresh = median * low_ratio
    durations = []
    for i in range(1, len(history)):
        durations.append(history[i]["t"] - history[i - 1]["t"])
    total = sum(durations) if durations else 0.0
    low_time = 0.0
    for i, m in enumerate(mag):
        if m < low_thresh and i < len(durations):
            low_time += durations[i]
    if total <= 0:
        return False
    return (low_time / total) > time_ratio


def sequence_check(history, joint_primary, joint_secondary, frac=0.5):
    """
    Verifica se joint_primary atinge fração 'frac' do ROM antes
    de joint_secondary (heurística simples).
    """
    def time_to_frac(joint):
        vals = [s["angles"].get(joint) for s in history if joint in s["angles"]]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        mn, mx = min(vals), max(vals)
        if (mx - mn) == 0:
            return None
        target = mn + frac * (mx - mn)
        for s in history:
            v = s["angles"].get(joint)
            if v is None:
                continue
            if abs(v - target) < 1.0:
                return s["t"]
        return None

    t1 = time_to_frac(joint_primary)
    t2 = time_to_frac(joint_secondary)
    if t1 is None or t2 is None:
        return True
    return t1 <= t2


def rep_time(history):
    """Duração da repetição (s)."""
    if not history:
        return 0.0
    return history[-1]["t"] - history[0]["t"]