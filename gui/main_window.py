from core.metrics import rom_of_joint, rep_time
import cv2
import numpy as np
import time

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QComboBox, QTextEdit,
    QHBoxLayout, QVBoxLayout, QPushButton, QTableWidget, QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCharts import QChart, QChartView, QBarSeries, QBarCategoryAxis, QValueAxis

from gui.led_driver import LedDriver
from threads.camera import CameraThread
from threads.model import ModelThread
from persistence.error_tracker import ErrorTracker
from evaluation.rep_machine import RepStateMachine
from evaluation.advanced_checks import evaluate_rep_metrics
from core.angles import calc_angle, angle_message
from core.config import KEYPOINT_MAP, REP_CONFIGS
from gui.dashboard import update_dashboard, reset_data_action


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.led_red_timer = QTimer()
        self.led_red_timer.setInterval(150)  # 150 ms
        self.led_red_timer.timeout.connect(self._led_red_tick)
        self.led_red_on = False



        self.setWindowTitle("AcademIA")
        self.resize(1200, 800)

        self.tracker = ErrorTracker()
        # fábrica de máquinas por exercício
        self.machines = {}

        self._build_ui()
        self._start_threads()

        self.tracker = ErrorTracker()
        self.machines = {}
        self.led = LedDriver("COM5")  # mesmo nome da porta do teste


    def _led_red_tick(self):
        self.led_red_on = not self.led_red_on
        self.led.red() if self.led_red_on else self.led.off()

    # --------------------------------------------------
    # UI
    # --------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        lay_main = QHBoxLayout(central)

        # ------ left ------
        left = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.combo = QComboBox()
        self.combo.addItems(list(REP_CONFIGS.keys()))
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setMaximumHeight(300)
        left.addWidget(self.video_label)
        left.addWidget(self.combo)
        left.addWidget(self.text_output)
        lay_main.addLayout(left, stretch=2)

        # ------ right (dashboard) ------
        right = QVBoxLayout()
        right.addWidget(QLabel("Dashboard de Erros", alignment=Qt.AlignmentFlag.AlignCenter))

        self.chart = QChart()
        self.chart.setTitle("Erros por exercício")
        self.series = QBarSeries()
        self.chart.addSeries(self.series)
        self.axis_x = QBarCategoryAxis()
        self.axis_y = QValueAxis()
        self.chart.addAxis(self.axis_x, Qt.AlignmentFlag.AlignBottom)
        self.chart.addAxis(self.axis_y, Qt.AlignmentFlag.AlignLeft)
        self.series.attachAxis(self.axis_x)
        self.series.attachAxis(self.axis_y)
        self.chart_view = QChartView(self.chart)
        self.chart_view.setMinimumHeight(300)
        right.addWidget(self.chart_view)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Exercício", "Total Reps", "Erros", "Taxa (%)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        right.addWidget(self.table)

        btn_lay = QHBoxLayout()
        refresh = QPushButton("Atualizar")
        refresh.clicked.connect(self._update_dashboard)
        reset = QPushButton("Resetar")
        reset.clicked.connect(self._reset_data)
        btn_lay.addWidget(refresh)
        btn_lay.addWidget(reset)
        right.addLayout(btn_lay)
        lay_main.addLayout(right, stretch=1)

    # --------------------------------------------------
    # Threads
    # --------------------------------------------------
    def _start_threads(self):
        self.cam_thread = CameraThread()
        self.model_thread = ModelThread(self.cam_thread)
        self.model_thread.result_ready.connect(self._process_results)
        self.cam_thread.start()
        self.model_thread.start()

    # --------------------------------------------------
    # Slots
    # --------------------------------------------------
    def _process_results(self, results, frame):
        ex = self.combo.currentText()
        frame_time = time.time()
        machine = self._get_machine(ex)

        if frame is None or frame.size == 0:
            print("[PROCESS] frame vazio – retornando")
            return

        # ---------- ângulo 100 % protegido ----------
        def safe_angle(a, b, c, _name):
            if a is None or b is None or c is None:
                return 0.0
            ba, bc = a - b, c - b
            norm = np.linalg.norm(ba) * np.linalg.norm(bc)
            if abs(norm) < 1e-6:
                return 0.0
            cos = np.dot(ba, bc) / norm
            return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

        # ---------- desenho ----------
        def safe_draw(img, results):
            bones = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12),
                     (11, 13), (13, 15), (12, 14), (14, 16), (5, 11), (6, 12)]
            for r in results:
                if r.keypoints is None or len(r.keypoints.xy) == 0:
                    continue
                k = r.keypoints.xy[0]
                if len(k) < 17:
                    continue
                for x, y in k:
                    cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
                for i, j in bones:
                    if i < len(k) and j < len(k):
                        pt1 = (int(k[i][0]), int(k[i][1]))
                        pt2 = (int(k[j][0]), int(k[j][1]))
                        cv2.line(img, pt1, pt2, (0, 255, 0), 2)

        # ---------- tabela de faixas dinâmicas ----------
        DYN_RANGE = {
            "supino": {
                "deep": {"cotovelo_esq": (45, 60), "cotovelo_dir": (45, 60)},
                "peak": {"cotovelo_esq": (100, 110), "cotovelo_dir": (100, 110)},
            },
            "agachamento": {
                "deep": {"joelho_esq": (90, 120), "joelho_dir": (90, 120)},
                "peak": {"joelho_esq": (170, 180), "joelho_dir": (170, 180)},
            },
            "terra": {
                "deep": {"quadril_esq": (100, 120), "quadril_dir": (100, 120)},
                "peak": {"quadril_esq": (160, 180), "quadril_dir": (160, 180)},
            },
            "leg45": {
                "deep": {"joelho_esq": (90, 120), "joelho_dir": (90, 120),
                         "quadril_esq": (60, 80), "quadril_dir": (60, 80)},
                "peak": {"joelho_esq": (170, 180), "joelho_dir": (170, 180),
                         "quadril_esq": (160, 180), "quadril_dir": (160, 180)},
            },
            "leg90": {
                "deep": {"joelho_esq": (90, 120), "joelho_dir": (90, 120),
                         "quadril_esq": (60, 100), "quadril_dir": (60, 100)},
                "peak": {"joelho_esq": (170, 180), "joelho_dir": (170, 180),
                         "quadril_esq": (160, 180), "quadril_dir": (160, 180)},
            },
            "puxada_alta": {
                "deep": {"ombro_esq": (60, 90), "ombro_dir": (60, 90),
                         "cotovelo_esq": (80, 100), "cotovelo_dir": (80, 100)},
                "peak": {"ombro_esq": (30, 60), "ombro_dir": (30, 60),
                         "cotovelo_esq": (100, 120), "cotovelo_dir": (100, 120)},
            },
            "cadeira_romana": {
                "deep": {"quadril_esq": (120, 140), "quadril_dir": (120, 140)},
                "peak": {"quadril_esq": (160, 180), "quadril_dir": (160, 180)},
            },
            "hack": {
                "deep": {"joelho_esq": (90, 120), "joelho_dir": (90, 120),
                         "quadril_esq": (60, 100), "quadril_dir": (60, 100)},
                "peak": {"joelho_esq": (170, 180), "joelho_dir": (170, 180),
                         "quadril_esq": (160, 180), "quadril_dir": (160, 180)},
            },
            "remada_alta": {
                "deep": {"ombro_esq": (60, 90), "ombro_dir": (60, 90),
                         "cotovelo_esq": (80, 100), "cotovelo_dir": (80, 100)},
                "peak": {"ombro_esq": (30, 60), "ombro_dir": (30, 60),
                         "cotovelo_esq": (100, 120), "cotovelo_dir": (100, 120)},
            },
            "remada_baixa": {
                "deep": {"cotovelo_esq": (60, 80), "cotovelo_dir": (60, 80)},
                "peak": {"cotovelo_esq": (100, 120), "cotovelo_dir": (100, 120)},
            },
            "remada_maquina": {
                "deep": {"cotovelo_esq": (60, 80), "cotovelo_dir": (60, 80)},
                "peak": {"cotovelo_esq": (100, 120), "cotovelo_dir": (100, 120)},
            },
            "desenvolvimento de ombro": {
                "deep": {"ombro_esq": (0, 30), "ombro_dir": (0, 30),
                         "cotovelo_esq": (60, 90), "cotovelo_dir": (60, 90)},
                "peak": {"ombro_esq": (140, 160), "ombro_dir": (140, 160),
                         "cotovelo_esq": (160, 180), "cotovelo_dir": (160, 180)},
            },
        }

        out_lines = []
        for r in results:
            kp = r.keypoints.xy
            if len(kp) == 0 or len(kp[0]) < 17:
                continue
            p0 = kp[0]

            c = {name: np.array(p0[idx]) if idx < len(p0) else np.array([0.0, 0.0])
                 for name, idx in KEYPOINT_MAP.items()}

            angles = {}
            for side in ("esq", "dir"):
                angles[f"cotovelo_{side}"] = safe_angle(c[f"ombro_{side}"], c[f"cotovelo_{side}"], c[f"punho_{side}"],
                                                        "")
                angles[f"joelho_{side}"] = safe_angle(c[f"quadril_{side}"], c[f"joelho_{side}"], c[f"tornozelo_{side}"],
                                                      "")
                angles[f"quadril_{side}"] = safe_angle(c[f"ombro_{side}"], c[f"quadril_{side}"], c[f"joelho_{side}"],
                                                       "")
                angles[f"ombro_{side}"] = safe_angle(c[f"cotovelo_{side}"], c[f"ombro_{side}"], c[f"quadril_{side}"],
                                                     "")

            machine.add_sample(angles, {k: (float(v[0]), float(v[1])) for k, v in c.items()}, frame_time)

            # ---------- verificação dinâmica ----------
            if ex not in DYN_RANGE:
                out_lines.append(f"=== {ex.upper()} (sem faixas) ===")
            else:
                phase = machine.phase
                rng = DYN_RANGE[ex]["deep"] if phase in {"down", "bottom"} else DYN_RANGE[ex]["peak"]
                out_lines.append(f"=== {ex.upper()} – fase {phase} ===")

                ok_counter = 0
                for joint, (min_a, max_a) in rng.items():
                    angle = angles.get(joint, 0)
                    if min_a <= angle <= max_a:
                        out_lines.append(angle_message(joint, angle, min_a, max_a))
                        ok_counter += 1
                    else:
                        machine.errors.add(joint)
                        out_lines.append(angle_message(joint, angle, min_a, max_a))

                # ---------- LED ----------
                if ok_counter == len(rng):  # todos OK
                    self.led_red_timer.stop()
                    self.led.green()
                else:  # algum fora
                    if not self.led_red_timer.isActive():
                        self.led_red_timer.start()
                        

            break  # primeira pessoa detectada

        # ---------- texto ----------
        if out_lines:
            header = f"\n{'-' * 50}\n"
            block = header + "\n".join(out_lines) + header
            prev = self.text_output.toPlainText()
            self.text_output.setText(block + "\n" + prev)

        # ---------- desenho ----------
        try:
            annotated = frame.copy()
            safe_draw(annotated, results)
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            # garante que a memória é contígua antes de passar para o Qt
            if not rgb.data.contiguous:
                rgb = np.ascontiguousarray(rgb)

            h, w, ch = rgb.shape
            qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            if qt_img.isNull():
                print("[DRAW] QImage nula – ignorando frame")
                return
            self.video_label.setPixmap(QPixmap.fromImage(qt_img))
        except Exception as e:
            print(f"[DRAW] erro: {e}")
            return

        # ---------- rep completa? ----------
        if machine.is_complete:
            self._complete_rep(ex)

    # --------------------------------------------------
    # Avaliações
    # --------------------------------------------------
    def _simple_eval(self, ex: str, angles: dict):
        """Apenas prints de ângulos – mesmas regras do original."""
        lines = []
        cfg = REP_CONFIGS.get(ex)
        if not cfg:
            return lines

        if ex == "supino":
            lines += ["=== SUPINO ==="]
            lines += [angle_message("Cotovelo Esq", angles["cotovelo_esq"], 45, 90)]
            lines += [angle_message("Cotovelo Dir", angles["cotovelo_dir"], 45, 90)]

        elif ex == "agachamento":
            lines += ["=== AGACHAMENTO ==="]
            lines += [angle_message("Joelho Esq", angles["joelho_esq"], 80, 140)]
            lines += [angle_message("Joelho Dir", angles["joelho_dir"], 80, 140)]

        # (adicionar demais exercícios idem ao original se desejar)
        return lines

    def _complete_rep(self, ex: str):
        machine = self._get_machine(ex)
        hist = list(machine.history)

        # avaliação avançada
        adv_errors, adv_msgs = evaluate_rep_metrics(ex, hist)

        # consolida erros
        all_errors = list(machine.errors) + adv_errors
        all_errors = list(dict.fromkeys(all_errors))  # uniq

        # tracking
        self.tracker.add_rep(ex, all_errors)

        # monta texto
        header = f"=== RESUMO REP - {ex.upper()} ==="
        simple_txt = f"Erros simples: {', '.join(machine.errors) if machine.errors else 'Nenhum'}"
        adv_txt = "\n".join(adv_msgs) if adv_msgs else "Correções avançadas: Nada a reportar ✔"
        roms = {k: rom_of_joint(hist, k) for k in ["joelho_esq", "joelho_dir", "cotovelo_esq", "cotovelo_dir"]}
        rep_t = rep_time(hist)
        summary = (
            f"{header}\n{simple_txt}\n{adv_txt}\n"
            f"ROMs: {', '.join([f'{k}={v:.1f}°' for k, v in roms.items()])}\n"
            f"Tempo rep: {rep_t:.2f}s\n----------------------\n"
        )
        print(summary)

        self.text_output.setText(summary)

        # guarda resumo na máquina
        machine.finish_rep()

        # dashboard
        self._update_dashboard()

        # ---------- salva no tracker ----------
        self.tracker.add_rep(ex, all_errors)

        # ---------- log / texto ----------
        print(summary)

    # --------------------------------------------------
    # Utilidades
    # --------------------------------------------------
    def _get_machine(self, ex: str) -> RepStateMachine:
        if ex not in self.machines:
            self.machines[ex] = RepStateMachine(ex)
        return self.machines[ex]

    def _draw_skeleton(self, frame, results):
        annotated = frame.copy()
        bones = [
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (5, 6),
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
            (5, 11),
            (6, 12),
        ]
        for r in results:
            if r.keypoints is not None and len(r.keypoints.xy):
                k = r.keypoints.xy[0]
                for x, y in k:
                    cv2.circle(annotated, (int(x), int(y)), 4, (0, 255, 0), -1)
                for i, j in bones:
                    if i < len(k) and j < len(k):
                        pt1 = (int(k[i][0]), int(k[i][1]))
                        pt2 = (int(k[j][0]), int(k[j][1]))
                        cv2.line(annotated, pt1, pt2, (0, 255, 0), 2)

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        qt_img = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888
        )
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    def _update_dashboard(self):
        update_dashboard(
            self.chart,
            self.series,
            self.axis_x,
            self.axis_y,
            self.table,
            self.tracker,
        )

    def _reset_data(self):
        ok = reset_data_action(self, self.tracker, self.machines)
        if ok:
            self._update_dashboard()
            QMessageBox.information(self, "Sucesso", "Dados resetados com sucesso!")

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------
    def closeEvent(self, event):
        self.cam_thread.stop()
        self.model_thread.stop()
        event.accept()