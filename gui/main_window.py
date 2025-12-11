from core.metrics import rom_of_joint, rep_time
import cv2
import numpy as np
import time

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QComboBox, QTextEdit,
    QHBoxLayout, QVBoxLayout, QPushButton, QTableWidget, QMessageBox,
)
from PyQt6.QtCore import Qt
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

        # ---------- ângulo 100 % protegido ----------
        def safe_angle(a, b, c, name):
            if a is None or b is None or c is None:
                return 0.0
            ba, bc = a - b, c - b
            norm = np.linalg.norm(ba) * np.linalg.norm(bc)
            if abs(norm) < 1e-6:          # divisão por zero
                return 0.0
            cos = np.dot(ba, bc) / norm
            return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

        # ---------- desenho blindado ----------
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

        # ---------- processa ----------
        out_lines = []
        for r in results:
            kp = r.keypoints.xy
            if len(kp) == 0 or len(kp[0]) < 17:   # <── evita acesso inválido
                continue
            p0 = kp[0]

            c = {}
            for name, idx in KEYPOINT_MAP.items():
                c[name] = np.array(p0[idx]) if idx < len(p0) else np.array([0.0, 0.0])

            angles = {}
            for side in ("esq", "dir"):
                o, coto, punho = f"ombro_{side}", f"cotovelo_{side}", f"punho_{side}"
                q, joelho, torn = f"quadril_{side}", f"joelho_{side}", f"tornozelo_{side}"

                if all(idx < len(p0) for idx in (KEYPOINT_MAP[o], KEYPOINT_MAP[coto], KEYPOINT_MAP[punho])):
                    angles[f"cotovelo_{side}"] = safe_angle(c[o], c[coto], c[punho], f"cotovelo_{side}")
                else:
                    angles[f"cotovelo_{side}"] = 0.0

                if all(idx < len(p0) for idx in (KEYPOINT_MAP[q], KEYPOINT_MAP[joelho], KEYPOINT_MAP[torn])):
                    angles[f"joelho_{side}"] = safe_angle(c[q], c[joelho], c[torn], f"joelho_{side}")
                else:
                    angles[f"joelho_{side}"] = 0.0

                if all(idx < len(p0) for idx in (KEYPOINT_MAP[o], KEYPOINT_MAP[q], KEYPOINT_MAP[joelho])):
                    angles[f"quadril_{side}"] = safe_angle(c[o], c[q], c[joelho], f"quadril_{side}")
                else:
                    angles[f"quadril_{side}"] = 0.0

                if all(idx < len(p0) for idx in (KEYPOINT_MAP[coto], KEYPOINT_MAP[o], KEYPOINT_MAP[q])):
                    angles[f"ombro_{side}"] = safe_angle(c[coto], c[o], c[q], f"ombro_{side}")
                else:
                    angles[f"ombro_{side}"] = 0.0

            machine.add_sample(angles, {k: (float(v[0]), float(v[1])) for k, v in c.items()}, frame_time)

            # ---------- contador geral para "permanece certo" ----------
            if not hasattr(self, "_ok_counter"):
                self._ok_counter = 0

            # ------------ verificações por exercício ------------
            if ex == "supino":
                val_e, val_d = angles.get("cotovelo_esq", 0), angles.get("cotovelo_dir", 0)
                out_lines.append("=== SUPINO ===")
                # esquerdo
                if val_e < 45 or val_e > 90:
                    machine.errors.add("cotovelo_esq")
                    self.led.blink_error()
                    out_lines.append(angle_message("Cotovelo Esq", val_e, 45, 90))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Cotovelo Esq", val_e, 45, 90))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)
                # direito
                if val_d < 45 or val_d > 90:
                    machine.errors.add("cotovelo_dir")
                    self.led.blink_error()
                    out_lines.append(angle_message("Cotovelo Dir", val_d, 45, 90))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Cotovelo Dir", val_d, 45, 90))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)

            elif ex == "agachamento":
                val_e, val_d = angles.get("joelho_esq", 0), angles.get("joelho_dir", 0)
                out_lines.append("=== AGACHAMENTO ===")
                # esquerdo
                if val_e < 90 or val_e > 120:
                    machine.errors.add("joelho_esq")
                    self.led.blink_error()
                    out_lines.append(angle_message("Joelho Esq", val_e, 90, 120))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Joelho Esq", val_e, 90, 120))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)
                # direito
                if val_d < 90 or val_d > 120:
                    machine.errors.add("joelho_dir")
                    self.led.blink_error()
                    out_lines.append(angle_message("Joelho Dir", val_d, 90, 120))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Joelho Dir", val_d, 90, 120))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)

            elif ex == "terra":
                val_e, val_d = angles.get("quadril_esq", 0), angles.get("quadril_dir", 0)
                out_lines.append("=== LEVANTAMENTO TERRA ===")
                # esquerdo
                if val_e < 100 or val_e > 120:
                    machine.errors.add("quadril_esq")
                    self.led.blink_error()
                    out_lines.append(angle_message("Quadril Esq", val_e, 100, 120))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Quadril Esq", val_e, 100, 120))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)
                # direito
                if val_d < 100 or val_d > 120:
                    machine.errors.add("quadril_dir")
                    self.led.blink_error()
                    out_lines.append(angle_message("Quadril Dir", val_d, 100, 120))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Quadril Dir", val_d, 100, 120))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)

            elif ex == "leg45":
                out_lines.append("=== LEG PRESS 45° ===")
                # joelhos
                val_e, val_d = angles.get("joelho_esq", 0), angles.get("joelho_dir", 0)
                if val_e < 90 or val_e > 120:
                    machine.errors.add("joelho_esq")
                    self.led.blink_error()
                    out_lines.append(angle_message("Joelho Esq", val_e, 90, 120))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Joelho Esq", val_e, 90, 120))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)
                if val_d < 90 or val_d > 120:
                    machine.errors.add("joelho_dir")
                    self.led.blink_error()
                    out_lines.append(angle_message("Joelho Dir", val_d, 90, 120))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Joelho Dir", val_d, 90, 120))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)
                # quadril
                val_e, val_d = angles.get("quadril_esq", 0), angles.get("quadril_dir", 0)
                if val_e < 60 or val_e > 80:
                    machine.errors.add("quadril_esq")
                    self.led.blink_error()
                    out_lines.append(angle_message("Quadril Esq", val_e, 60, 80))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Quadril Esq", val_e, 60, 80))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)
                if val_d < 60 or val_d > 80:
                    machine.errors.add("quadril_dir")
                    self.led.blink_error()
                    out_lines.append(angle_message("Quadril Dir", val_d, 60, 80))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Quadril Dir", val_d, 60, 80))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)

            elif ex == "leg90":
                out_lines.append("=== LEG PRESS HORIZONTAL (90°) ===")
                # joelhos
                val_e, val_d = angles.get("joelho_esq", 0), angles.get("joelho_dir", 0)
                if val_e < 90 or val_e > 120:
                    machine.errors.add("joelho_esq")
                    self.led.blink_error()
                    out_lines.append(angle_message("Joelho Esq", val_e, 90, 120))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Joelho Esq", val_e, 90, 120))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)
                if val_d < 90 or val_d > 120:
                    machine.errors.add("joelho_dir")
                    self.led.blink_error()
                    out_lines.append(angle_message("Joelho Dir", val_d, 90, 120))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Joelho Dir", val_d, 90, 120))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)
                # quadril
                val_e, val_d = angles.get("quadril_esq", 0), angles.get("quadril_dir", 0)
                if val_e < 60 or val_e > 100:
                    machine.errors.add("quadril_esq")
                    self.led.blink_error()
                    out_lines.append(angle_message("Quadril Esq", val_e, 60, 100))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Quadril Esq", val_e, 60, 100))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)
                if val_d < 60 or val_d > 100:
                    machine.errors.add("quadril_dir")
                    self.led.blink_error()
                    out_lines.append(angle_message("Quadril Dir", val_d, 60, 100))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Quadril Dir", val_d, 60, 100))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)

            elif ex == "puxada_alta":
                val_e, val_d = angles.get("ombro_esq", 0), angles.get("ombro_dir", 0)
                out_lines.append("=== PUXADA ALTA FRONTAL ===")
                # ombro
                if val_e < 60 or val_e > 90:
                    if "ombro_esq" not in machine.errors:
                        machine.errors.add("ombro_esq")
                        self.led.blink_error()
                    out_lines.append(angle_message("Ombro Esq", val_e, 60, 90))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Ombro Esq", val_e, 60, 90))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)
                if val_d < 60 or val_d > 90:
                    if "ombro_dir" not in machine.errors:
                        machine.errors.add("ombro_dir")
                        self.led.blink_error()
                    out_lines.append(angle_message("Ombro Dir", val_d, 60, 90))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Ombro Dir", val_d, 60, 90))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)
                # cotovelo
                val_e, val_d = angles.get("cotovelo_esq", 0), angles.get("cotovelo_dir", 0)
                if val_e < 80 or val_e > 100:
                    if "cotovelo_esq" not in machine.errors:
                        machine.errors.add("cotovelo_esq")
                        self.led.blink_error()
                    out_lines.append(angle_message("Cotovelo Esq", val_e, 80, 100))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Cotovelo Esq", val_e, 80, 100))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)
                if val_d < 80 or val_d > 100:
                    if "cotovelo_dir" not in machine.errors:
                        machine.errors.add("cotovelo_dir")
                        self.led.blink_error()
                    out_lines.append(angle_message("Cotovelo Dir", val_d, 80, 100))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Cotovelo Dir", val_d, 80, 100))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)

            elif ex == "cadeira_romana":
                val_e, val_d = angles.get("quadril_esq", 0), angles.get("quadril_dir", 0)
                avg_q = (val_e + val_d) / 2.0
                out_lines.append("=== CADEIRA ROMANA (BACK EXTENSION) ===")
                # esquerdo
                if not ((120 <= val_e <= 140) or (160 <= val_e <= 180)):
                    if "quadril_esq_fora_flex_ext" not in machine.errors:
                        machine.errors.add("quadril_esq_fora_flex_ext")
                        self.led.blink_error()
                    out_lines.append(angle_message("Quadril Esq (flex/ext)", val_e, 120, 140))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Quadril Esq (flex/ext)", val_e, 120, 140))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)
                # direito
                if not ((120 <= val_d <= 140) or (160 <= val_d <= 180)):
                    if "quadril_dir_fora_flex_ext" not in machine.errors:
                        machine.errors.add("quadril_dir_fora_flex_ext")
                        self.led.blink_error()
                    out_lines.append(angle_message("Quadril Dir (flex/ext)", val_d, 120, 140))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Quadril Dir (flex/ext)", val_d, 120, 140))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)
                # média
                if avg_q > 185:
                    if "hiperextensao" not in machine.errors:
                        machine.errors.add("hiperextensao")
                        self.led.blink_error()
                    out_lines.append(angle_message("Quadril Médio (ext)", avg_q, 160, 180))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Quadril Médio (ext)", avg_q, 160, 180))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)

            elif ex == "hack":
                val_e, val_d = angles.get("joelho_esq", 0), angles.get("joelho_dir", 0)
                out_lines.append("=== HACK SQUAT ===")
                # joelho
                if val_e < 90 or val_e > 120:
                    if "joelho_esq" not in machine.errors:
                        machine.errors.add("joelho_esq")
                        self.led.blink_error()
                    out_lines.append(angle_message("Joelho Esq", val_e, 90, 120))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Joelho Esq", val_e, 90, 120))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)
                if val_d < 90 or val_d > 120:
                    if "joelho_dir" not in machine.errors:
                        machine.errors.add("joelho_dir")
                        self.led.blink_error()
                    out_lines.append(angle_message("Joelho Dir", val_d, 90, 120))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Joelho Dir", val_d, 90, 120))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)
                # quadril
                val_e, val_d = angles.get("quadril_esq", 0), angles.get("quadril_dir", 0)
                if val_e < 60 or val_e > 100:
                    if "quadril_esq" not in machine.errors:
                        machine.errors.add("quadril_esq")
                        self.led.blink_error()
                    out_lines.append(angle_message("Quadril Esq", val_e, 60, 100))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Quadril Esq", val_e, 60, 100))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)
                if val_d < 60 or val_d > 100:
                    if "quadril_dir" not in machine.errors:
                        machine.errors.add("quadril_dir")
                        self.led.blink_error()
                    out_lines.append(angle_message("Quadril Dir", val_d, 60, 100))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Quadril Dir", val_d, 60, 100))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)

            elif ex == "remada_alta":
                val_e, val_d = angles.get("ombro_esq", 0), angles.get("ombro_dir", 0)
                out_lines.append("=== REMADA MÁQUINA ALTA (WIDE GRIP) ===")
                # ombro
                if val_e < 45 or val_e > 60:
                    if "ombro_esq" not in machine.errors:
                        machine.errors.add("ombro_esq")
                        self.led.blink_error()
                    out_lines.append(angle_message("Ombro Esq", val_e, 45, 60))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Ombro Esq", val_e, 45, 60))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)
                if val_d < 45 or val_d > 60:
                    if "ombro_dir" not in machine.errors:
                        machine.errors.add("ombro_dir")
                        self.led.blink_error()
                    out_lines.append(angle_message("Ombro Dir", val_d, 45, 60))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Ombro Dir", val_d, 45, 60))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)
                # cotovelo
                val_e, val_d = angles.get("cotovelo_esq", 0), angles.get("cotovelo_dir", 0)
                if val_e < 60 or val_e > 100:
                    if "cotovelo_esq" not in machine.errors:
                        machine.errors.add("cotovelo_esq")
                        self.led.blink_error()
                    out_lines.append(angle_message("Cotovelo Esq", val_e, 60, 100))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Cotovelo Esq", val_e, 60, 100))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)
                if val_d < 60 or val_d > 100:
                    if "cotovelo_dir" not in machine.errors:
                        machine.errors.add("cotovelo_dir")
                        self.led.blink_error()
                    out_lines.append(angle_message("Cotovelo Dir", val_d, 60, 100))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Cotovelo Dir", val_d, 60, 100))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)

            elif ex == "desenvolvimento de ombro":
                # ombro
                val_e, val_d = angles.get("ombro_esq", 0), angles.get("ombro_dir", 0)
                out_lines.append("=== DESENVOLVIMENTO DE OMBRO ===")
                if val_e < 0 or val_e > 160:
                    if "ombro_esq" not in machine.errors:
                        machine.errors.add("ombro_esq")
                        self.led.blink_error()
                    out_lines.append(angle_message("Ombro Esq", val_e, 0, 160))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Ombro Esq", val_e, 0, 160))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)
                if val_d < 0 or val_d > 160:
                    if "ombro_dir" not in machine.errors:
                        machine.errors.add("ombro_dir")
                        self.led.blink_error()
                    out_lines.append(angle_message("Ombro Dir", val_d, 0, 160))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Ombro Dir", val_d, 0, 160))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)
                # cotovelo
                val_e, val_d = angles.get("cotovelo_esq", 0), angles.get("cotovelo_dir", 0)
                if val_e < 60 or val_e > 180:
                    if "cotovelo_esq" not in machine.errors:
                        machine.errors.add("cotovelo_esq")
                        self.led.blink_error()
                    out_lines.append(angle_message("Cotovelo Esq", val_e, 60, 180))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Cotovelo Esq", val_e, 60, 180))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)
                if val_d < 60 or val_d > 180:
                    if "cotovelo_dir" not in machine.errors:
                        machine.errors.add("cotovelo_dir")
                        self.led.blink_error()
                    out_lines.append(angle_message("Cotovelo Dir", val_d, 60, 180))
                    self._ok_counter = 0
                else:
                    out_lines.append(angle_message("Cotovelo Dir", val_d, 60, 180))
                    self._ok_counter += 1
                    if self._ok_counter % 30 == 1:
                        self.led.blink_success(times=1, on_ms=40)

            break  # primeira pessoa detectada

            # ---------- texto ----------
        if out_lines:
            prev = self.text_output.toPlainText()
            new_text = "\n".join(out_lines) + ("\n\n" + prev if prev else "")
            self.text_output.setText(new_text)

            # ---------- desenho ----------
        annotated = frame.copy()
        safe_draw(annotated, results)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        qt_img = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888
        )
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

        # ---------- rep completa? ----------
        if machine.is_complete:
            self._complete_rep(ex)
            # ---------- texto ----------

        # ---------- acumula no topo (sem sobrescrever) ----------
        if out_lines:
            header = f"\n{'-' * 50}\n"
            block = header + "\n".join(out_lines) + header
            prev = self.text_output.toPlainText()
            self.text_output.setText(block + "\n" + prev)  # << acrescenta no TOPO

        # ---------- desenho ----------
        annotated = frame.copy()
        safe_draw(annotated, results)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        qt_img = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888
        )
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

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

        # ---------- LED: verde se perfeita ----------
        if not all_errors:  # lista vazia → sem erro
            self.led.blink_success()  # pisca verde 2×

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