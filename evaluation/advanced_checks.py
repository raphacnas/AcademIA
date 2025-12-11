from typing import List, Tuple
import numpy as np

from core.metrics import *


def evaluate_rep_metrics(ex: str, history) -> Tuple[List[str], List[str]]:
    """
    Avaliações avançadas após o término de uma repetição.
    Retorna: (lista de chaves de erro, lista de mensagens descritivas)
    """
    errors: List[str] = []
    msgs: List[str] = []

    if not history:
        return errors, msgs

    # ---------- thresholds ----------
    lateral_px_thresh = 15.0
    symmetry_thresh_deg = 12.0
    traj_straight_thresh = 0.82
    stacking_px_thresh = 25.0

    # ---------- braços ----------
    arm_ex = [
        "supino",
        "remada_maquina",
        "remada_baixa",
        "remada_alta",
        "desenvolvimento de ombro",
        "puxada_alta",
    ]
    if ex in arm_ex:
        sym = symmetry_measure(history, "cotovelo_esq", "cotovelo_dir")
        if sym > symmetry_thresh_deg:
            errors.append("simetria_braços")
            msgs.append(f"Assimetria braços média {sym:.1f}°")

        for side, punho in (("esq", "punho_esq"), ("dir", "punho_dir")):
            traj = trajectory_straightness(history, punho)
            if traj < traj_straight_thresh:
                errors.append(f"trajetoria_{punho}")
                msgs.append(f"Trajetória {punho} curvilínea ({traj:.2f})")

        for side, coto, punho in (
            ("esq", "cotovelo_esq", "punho_esq"),
            ("dir", "cotovelo_dir", "punho_dir"),
        ):
            stack = stacking_check(history, coto, punho)
            if stack > stacking_px_thresh:
                errors.append(f"stacking_{side}")
                msgs.append(f"Punho {side} desalinhado do cotovelo ~{stack:.0f}px")

        # sticking point no eixo principal (primeiro da lista ou string)
        from core.exercise_rules import get_exercise_config
        cfg = get_exercise_config(ex)
        if cfg:
            axis = cfg["axis"][0] if isinstance(cfg["axis"], list) else cfg["axis"]
            if detect_sticking_point(history, axis):
                errors.append("sticking_point")
                msgs.append("Sticking point detectado no eixo principal")

    # ---------- pernas ----------
    leg_ex = ["agachamento", "leg45", "leg90", "hack", "terra", "cadeira_romana"]
    if ex in leg_ex:
        sym_k = symmetry_measure(history, "joelho_esq", "joelho_dir")
        if sym_k > 10.0:
            errors.append("simetria_pernas")
            msgs.append(f"Assimetria joelhos média {sym_k:.1f}°")

        lat_hip = lateral_deviation_px(history, "quadril_esq")
        if lat_hip > lateral_px_thresh:
            errors.append("instabilidade_lateral")
            msgs.append(f"Oscilação lateral quadril ~{lat_hip:.1f}px")

        rom_k = rom_of_joint(history, "joelho_esq")
        if rom_k < 20.0:
            errors.append("rom_pequeno")
            msgs.append(f"Baixa amplitude joelho esq {rom_k:.1f}°")

        cfg = get_exercise_config(ex)
        if cfg:
            axis = cfg["axis"][0] if isinstance(cfg["axis"], list) else cfg["axis"]
            if detect_sticking_point(history, axis):
                errors.append("sticking_point")
                msgs.append("Sticking point detectado em articulação principal")

    # ---------- tempo ----------
    t = rep_time(history)
    if t > 6.0:
        errors.append("rep_muito_lenta")
        msgs.append(f"Repetição muito lenta ({t:.2f}s)")
    if t < 0.6:
        errors.append("rep_muito_rapida")
        msgs.append(f"Repetição muito rápida ({t:.2f}s)")

    # ---------- consistência com últimas reps ----------
    # last_reps fica em RepStateMachine.last_reps – passado pelo cliente
    # (ver MainWindow)

    # ---------- sequência ----------
    if ex == "agachamento":
        if not sequence_check(history, "quadril_esq", "joelho_esq"):
            errors.append("sequencia_errada")
            msgs.append("Sequência incorreta: joelho iniciou antes do quadril")

    if ex == "supino":
        if not sequence_check(history, "cotovelo_esq", "ombro_esq"):
            errors.append("sequencia_errada")
            msgs.append("Sequência supino incomum")

    return errors, msgs