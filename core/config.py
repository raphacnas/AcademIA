import cv2
import platform

YOLO_PATH = "assets/yolo11n-pose.pt"

KEYPOINT_MAP = {
    "ombro_esq": 5,
    "cotovelo_esq": 7,
    "punho_esq": 9,
    "ombro_dir": 6,
    "cotovelo_dir": 8,
    "punho_dir": 10,
    "quadril_esq": 11,
    "joelho_esq": 13,
    "tornozelo_esq": 15,
    "quadril_dir": 12,
    "joelho_dir": 14,
    "tornozelo_dir": 16,
}

REP_CONFIGS = {
    "supino": {
        "axis": ["cotovelo_esq", "cotovelo_dir"],
        "top_thresh": 140,
        "bottom_thresh": 60,
        "bottom_enter": 100,
        "top_enter": 130,
    },
    "agachamento": {
        "axis": "joelho_esq",
        "top_thresh": 170,
        "bottom_thresh": 100,
        "bottom_enter": 130,
        "top_enter": 160,
    },
    "terra": {
        "axis": "quadril_esq",
        "top_thresh": 140,
        "bottom_thresh": 100,
        "bottom_enter": 120,
        "top_enter": 130,
    },
    "leg45": {
        "axis": "joelho_esq",
        "top_thresh": 170,
        "bottom_thresh": 90,
        "bottom_enter": 110,
        "top_enter": 160,
    },
    "leg90": {
        "axis": "joelho_esq",
        "top_thresh": 170,
        "bottom_thresh": 90,
        "bottom_enter": 110,
        "top_enter": 160,
    },
    "puxada_alta": {
        "axis": "ombro_esq",
        "top_thresh": 100,
        "bottom_thresh": 40,
        "bottom_enter": 60,
        "top_enter": 90,
    },
    "cadeira_romana": {
        "axis": "quadril_esq",
        "top_thresh": 170,
        "bottom_thresh": 130,
        "bottom_enter": 150,
        "top_enter": 160,
    },
    "hack": {
        "axis": "joelho_esq",
        "top_thresh": 170,
        "bottom_thresh": 90,
        "bottom_enter": 110,
        "top_enter": 160,
    },
    "remada_maquina": {
        "axis": ["cotovelo_esq", "cotovelo_dir"],
        "top_thresh": 100,
        "bottom_thresh": 60,
        "bottom_enter": 80,
        "top_enter": 95,
    },
    "remada_baixa": {
        "axis": ["cotovelo_esq", "cotovelo_dir"],
        "top_thresh": 100,
        "bottom_thresh": 60,
        "bottom_enter": 80,
        "top_enter": 95,
    },
    "remada_alta": {
        "axis": ["cotovelo_esq", "cotovelo_dir"],
        "top_thresh": 100,
        "bottom_thresh": 60,
        "bottom_enter": 80,
        "top_enter": 95,
    },
    "desenvolvimento de ombro": {
        "axis": ["ombro_esq", "ombro_dir"],
        "top_thresh": 160,
        "bottom_thresh": 0,
        "bottom_enter": 20,
        "top_enter": 150,
    },
}

def cam_backend():
    return {
        "Windows": cv2.CAP_DSHOW,
        "Linux": cv2.CAP_V4L2,
        "Darwin": cv2.CAP_AVFOUNDATION,
    }.get(platform.system(), cv2.CAP_ANY)