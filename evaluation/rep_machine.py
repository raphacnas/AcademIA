from collections import deque
from core.exercise_rules import get_exercise_config, axis_value
from core.metrics import rep_time, rom_of_joint
import time


class RepStateMachine:
    """
    Máquina de estados para controle de repetições de UM exercício.
    Guarda histórico, fase, flags e lista de erros da rep atual.
    """

    def __init__(self, exercise: str):
        self.ex = exercise
        self.history = deque(maxlen=600)
        self.last_reps = []  # dicionários resumo das últimas 20 reps
        self.phase = "top"  # top | down | bottom | up
        self.reached_bottom = False
        self.errors = set()
        self.reps = 0

    # ---------- API pública ----------
    def add_sample(self, angles: dict, kp: dict, t: float | None = None):
        """Registra amostra de frame na máquina."""
        t = t or time.time()
        self.history.append({"t": t, "angles": angles, "kp": kp})
        self._tick(angles)

    def finish_rep(self) -> dict:
        """Chamado quando se completa uma rep: retorna resumo."""
        roms = {
            k: rom_of_joint(self.history, k)
            for k in ["joelho_esq", "joelho_dir", "cotovelo_esq", "cotovelo_dir"]
        }
        rep_t = rep_time(self.history)
        summary = {
            "t": time.time(),
            "errors": list(self.errors),
            "rom": roms,
            "rep_time": rep_t,
        }
        self.last_reps.append(summary)
        if len(self.last_reps) > 20:
            self.last_reps.pop(0)

        self.reps += 1
        self._reset_cycle()
        return summary

    def reset_full(self):
        """Zera tudo (uso em reset de dashboard)."""
        self.history.clear()
        self.last_reps.clear()
        self.phase = "top"
        self.reached_bottom = False
        self.errors = set()
        self.reps = 0

    # ---------- interno ----------
    def _tick(self, angles: dict):
        cfg = get_exercise_config(self.ex)
        if not cfg:
            return
        val = axis_value(angles, cfg["axis"])
        if val is None:
            return

        top_thresh = cfg["top_thresh"]
        bottom_thresh = cfg["bottom_thresh"]
        top_enter = cfg.get("top_enter", top_thresh)
        bottom_enter = cfg.get("bottom_enter", bottom_thresh)

        if self.phase == "top":
            if val < bottom_enter:
                self.phase = "down"

        elif self.phase == "down":
            if val <= bottom_thresh:
                self.phase = "bottom"
                self.reached_bottom = True
            elif val >= top_enter:
                self.phase = "top"
                self.reached_bottom = False

        elif self.phase == "bottom":
            if val > bottom_thresh:
                self.phase = "up"

        elif self.phase == "up":
            if val >= top_thresh and self.reached_bottom:
                # rep completa – quem chama finish_rep() é o cliente
                pass
            elif val <= bottom_enter:
                self.phase = "bottom"

    def _reset_cycle(self):
        """Limpa só os dados da rep atual, mantendo last_reps."""
        self.phase = "top"
        self.reached_bottom = False
        self.errors = set()
        self.history.clear()

    # ---------- helpers ----------
    @property
    def is_complete(self) -> bool:
        """Indica que a rep acabou de completar (usado pelo cliente)."""
        cfg = get_exercise_config(self.ex)
        if not cfg:
            return False
        val = axis_value(self.history[-1]["angles"], cfg["axis"]) if self.history else None
        if val is None:
            return False
        return (
            self.phase == "up"
            and val >= cfg["top_thresh"]
            and self.reached_bottom
        )