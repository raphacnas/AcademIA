import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QComboBox, QTextEdit
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from ultralytics import YOLO


# Funções do seu código
def CalcAngulo(a, b, c):
    ba = a - b
    bc = c - b
    cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def printReturn(nome, angulo, minimo, maximo):
    status = "OK ✅" if minimo <= angulo <= maximo else "⚠ Fora do ideal"
    return f"{nome}: {angulo:.2f}° ({status})"


# Keypoints IDs
exercicios_ids = {
    "ombro_esq": 5, "cotovelo_esq": 7, "punho_esq": 9,
    "ombro_dir": 6, "cotovelo_dir": 8, "punho_dir": 10,
    "quadril_esq": 11, "joelho_esq": 13, "tornozelo_esq": 15,
    "quadril_dir": 12, "joelho_dir": 14, "tornozelo_dir": 16
}

# Modelo
model = YOLO("yolo11n-pose.pt")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Argus IA - Monitor de Exercícios")

        # Layout principal
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)

        self.combo = QComboBox()
        self.combo.addItems([
            "supino", "agachamento", "terra", "leg45", "leg90",
            "extensora", "flexora", "puxada_alta", "puxada_baixa",
            "abdutora", "adutora", "hack", "remada_maquina",
            "remada_baixa", "remada_alta"
        ])

        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.combo)
        layout.addWidget(self.text_output)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Timer para atualizar vídeo
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = cv2.VideoCapture(0)  # câmera
        self.timer.start(30)  # cerca de 30 FPS

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Faz predição com YOLO
        results = model.predict(frame, verbose=False)
        output_text = ""
        for fps in results:
            keypoints = fps.keypoints.xy
            if len(keypoints) == 0:
                continue
            pessoa0 = keypoints[0]
            coords = {nome: np.array(pessoa0[idx]) for nome, idx in exercicios_ids.items()}

            # Ângulos básicos
            ang_cotovelo_esq = CalcAngulo(coords["ombro_esq"], coords["cotovelo_esq"], coords["punho_esq"])
            ang_cotovelo_dir = CalcAngulo(coords["ombro_dir"], coords["cotovelo_dir"], coords["punho_dir"])

            ang_joelho_esq = CalcAngulo(coords["quadril_esq"], coords["joelho_esq"], coords["tornozelo_esq"])
            ang_joelho_dir = CalcAngulo(coords["quadril_dir"], coords["joelho_dir"], coords["tornozelo_dir"])

            ang_quadril_esq = CalcAngulo(coords["ombro_esq"], coords["quadril_esq"], coords["joelho_esq"])
            ang_quadril_dir = CalcAngulo(coords["ombro_dir"], coords["quadril_dir"], coords["joelho_dir"])

            ang_ombro_esq = CalcAngulo(coords["cotovelo_esq"], coords["ombro_esq"], coords["quadril_esq"])
            ang_ombro_dir = CalcAngulo(coords["cotovelo_dir"], coords["ombro_dir"], coords["quadril_dir"])

            exercicio_atual = self.combo.currentText()

            output_text = ""

            if exercicio_atual == "supino":
                output_text += "=== SUPINO ===\n"
                output_text += printReturn("Cotovelo Esq", ang_cotovelo_esq, 45, 90) + "\n"
                output_text += printReturn("Cotovelo Dir", ang_cotovelo_dir, 45, 90) + "\n"

            elif exercicio_atual == "agachamento":
                output_text += "=== AGACHAMENTO ===\n"
                output_text += printReturn("Joelho Esq", ang_joelho_esq, 90, 120) + "\n"
                output_text += printReturn("Joelho Dir", ang_joelho_dir, 90, 120) + "\n"

            elif exercicio_atual == "terra":
                output_text += "=== LEVANTAMENTO TERRA ===\n"
                output_text += printReturn("Quadril Esq", ang_quadril_esq, 100, 120) + "\n"
                output_text += printReturn("Quadril Dir", ang_quadril_dir, 100, 120) + "\n"

            elif exercicio_atual == "leg45":
                output_text += "=== LEG PRESS ===\n"
                output_text += printReturn("Joelho Esq", ang_joelho_esq, 90, 120) + "\n"
                output_text += printReturn("Joelho Dir", ang_joelho_dir, 90, 120) + "\n"
                output_text += printReturn("Quadril Esq", ang_quadril_esq, 60, 80) + "\n"
                output_text += printReturn("Quadril Dir", ang_quadril_dir, 60, 80) + "\n"

            elif exercicio_atual == "leg90":
                output_text += "=== LEG PRESS HORIZONTAL (90°) ===\n"
                output_text += printReturn("Joelho Esq", ang_joelho_esq, 90, 120) + "\n"
                output_text += printReturn("Joelho Dir", ang_joelho_dir, 90, 120) + "\n"
                output_text += printReturn("Quadril Esq", ang_quadril_esq, 60, 100) + "\n"
                output_text += printReturn("Quadril Dir", ang_quadril_dir, 60, 100) + "\n"

            elif exercicio_atual == "flexora":
                output_text += "=== FLEXORA ===\n"
                output_text += printReturn("Joelho Esq", ang_joelho_esq, 0, 120) + "\n"
                output_text += printReturn("Joelho Dir", ang_joelho_dir, 0, 120) + "\n"

            elif exercicio_atual == "extensora":
                output_text += "=== EXTENSORA ===\n"
                output_text += printReturn("Joelho Esq", ang_joelho_esq, 30, 90) + "\n"
                output_text += printReturn("Joelho Dir", ang_joelho_dir, 30, 90) + "\n"

            elif exercicio_atual == "puxada_alta":
                output_text += "=== PUXADA ALTA FRONTAL ===\n"
                output_text += printReturn("Ombro Esq", ang_ombro_esq, 60, 90) + "\n"
                output_text += printReturn("Ombro Dir", ang_ombro_dir, 60, 90) + "\n"
                output_text += printReturn("Cotovelo Esq", ang_cotovelo_esq, 80, 100) + "\n"
                output_text += printReturn("Cotovelo Dir", ang_cotovelo_dir, 80, 100) + "\n"

            elif exercicio_atual == "puxada_baixa":
                output_text += "=== PUXADA BAIXA ===\n"
                output_text += printReturn("Ombro Esq", ang_quadril_esq, 0, 30) + "\n"
                output_text += printReturn("Ombro Dir", ang_quadril_dir, 0, 30) + "\n"
                output_text += printReturn("Cotovelo Esq", ang_cotovelo_esq, 60, 100) + "\n"
                output_text += printReturn("Cotovelo Dir", ang_cotovelo_dir, 60, 100) + "\n"

            elif exercicio_atual == "abdutora":
                output_text += "=== ABDUTORA ===\n"
                output_text += printReturn("Quadril Esq (Abdução)", ang_quadril_esq, 0, 40) + "\n"
                output_text += printReturn("Quadril Dir (Abdução)", ang_quadril_dir, 0, 40) + "\n"

            elif exercicio_atual == "adutora":
                output_text += "=== ADUTORA ===\n"
                output_text += printReturn("Quadril Esq (Adução)", ang_quadril_esq, 0, 30) + "\n"
                output_text += printReturn("Quadril Dir (Adução)", ang_quadril_dir, 0, 30) + "\n"

            elif exercicio_atual == "hack":
                output_text += "=== HACK SQUAT ===\n"
                output_text += printReturn("Joelho Esq", ang_joelho_esq, 90, 120) + "\n"
                output_text += printReturn("Joelho Dir", ang_joelho_dir, 90, 120) + "\n"
                output_text += printReturn("Quadril Esq", ang_quadril_esq, 60, 100) + "\n"
                output_text += printReturn("Quadril Dir", ang_quadril_dir, 60, 100) + "\n"

            elif exercicio_atual == "remada_maquina":
                output_text += "=== REMADA MÁQUINA ===\n"
                output_text += printReturn("Cotovelo Esq", ang_cotovelo_esq, 60, 100) + "\n"
                output_text += printReturn("Cotovelo Dir", ang_cotovelo_dir, 60, 100) + "\n"
                output_text += printReturn("Ombro Esq", ang_ombro_esq, 20, 40) + "\n"
                output_text += printReturn("Ombro Dir", ang_ombro_dir, 20, 40) + "\n"

            elif exercicio_atual == "remada_baixa":
                output_text += "=== REMADA MÁQUINA BAIXA (CLOSE GRIP) ===\n"
                output_text += printReturn("Cotovelo Esq", ang_cotovelo_esq, 60, 100) + "\n"
                output_text += printReturn("Cotovelo Dir", ang_cotovelo_dir, 60, 100) + "\n"
                output_text += printReturn("Ombro Esq", ang_ombro_esq, 0, 30) + "\n"
                output_text += printReturn("Ombro Dir", ang_ombro_dir, 0, 30) + "\n"

            elif exercicio_atual == "remada_alta":
                output_text += "=== REMADA MÁQUINA ALTA (WIDE GRIP) ===\n"
                output_text += printReturn("Cotovelo Esq", ang_cotovelo_esq, 60, 100) + "\n"
                output_text += printReturn("Cotovelo Dir", ang_cotovelo_dir, 60, 100) + "\n"
                output_text += printReturn("Ombro Esq", ang_ombro_esq, 45, 60) + "\n"
                output_text += printReturn("Ombro Dir", ang_ombro_dir, 45, 60) + "\n"


        self.text_output.setText(output_text)

        # Converte frame para exibir no QLabel
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())
