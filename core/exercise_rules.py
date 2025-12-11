from .config import REP_CONFIGS
import numpy as np

def get_exercise_config(ex: str):
    return REP_CONFIGS.get(ex)

def axis_value(angles: dict, axis):
    if isinstance(axis, list):
        vals = [angles[k] for k in axis if k in angles]
        return float(np.mean(vals)) if vals else None
    return float(angles.get(axis)) if axis in angles else None