import json
import os
from collections import defaultdict
from datetime import datetime


class ErrorTracker:
    """
    Persiste contadores de erros por exercício em JSON.
    Format do arquivo:
    {
      "errors": {"supino": {"cotovelo_dir": 3, "simetria_braços": 1}},
      "total":  {"supino": 12},
      "last":   "2025-12-10T20:05:00"
    }
    """

    def __init__(self, file_path: str = "exercise_errors.json"):
        self.file = file_path
        self.errors = defaultdict(lambda: defaultdict(int))
        self.total = defaultdict(int)
        self._load()

    # ---------- leitura ----------
    def _load(self):
        if not os.path.exists(self.file):
            return
        try:
            with open(self.file, encoding="utf-8") as f:
                data = json.load(f)
            for ex, err in data.get("errors", {}).items():
                for k, v in err.items():
                    self.errors[ex][k] = v
            for ex, v in data.get("total", {}).items():
                self.total[ex] = v
        except Exception as e:
            print(f"[ErrorTracker] falha ao carregar: {e}")

    # ---------- escrita ----------
    def _save(self):
        try:
            with open(self.file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "errors": {k: dict(v) for k, v in self.errors.items()},
                        "total": dict(self.total),
                        "last": datetime.now().isoformat(),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:
            print(f"[ErrorTracker] falha ao salvar: {e}")

    # ---------- API ----------
    def add_rep(self, ex: str, errors_list: list[str]):
        """
        Registra o término de uma repetição:
        – incrementa total de reps para o exercício
        – incrementa contador de cada erro em errors_list
        """
        self.total[ex] += 1
        for err in errors_list:
            self.errors[ex][err] += 1
        self._save()

    def stats(self, ex: str) -> dict:
        """Retorna dict {erro: {c: int, p: float}} para o exercício."""
        t = self.total[ex]
        if not t:
            return {}
        return {k: {"c": v, "p": v / t * 100} for k, v in self.errors[ex].items()}

    def file_reset(self):
        """Apaga o arquivo físico e limpa memória."""
        try:
            os.remove(self.file)
        except FileNotFoundError:
            pass
        self.errors.clear()
        self.total.clear()