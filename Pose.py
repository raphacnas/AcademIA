import sys, cv2, numpy as np, json, time
from collections import defaultdict
from datetime import datetime
from PyQt6.QtWidgets import *
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap
from ultralytics import YOLO

# ---------------------------------------------------------
# Escolha de backend no Windows (DSHOW geralmente funciona)
# ---------------------------------------------------------
def cam_backend():
    return cv2.CAP_DSHOW

# ---------------------------------------------------------
# THREAD DA CÂMERA
# ---------------------------------------------------------
class CameraThread(QThread):
    frame_captured = pyqtSignal(np.ndarray)

    def __init__(self, src=0):
        super().__init__()
        self.src = src
        self.running = True
        self.queue = []

    def run(self):
        # Tentativa inicial
        cap = cv2.VideoCapture(self.src, cam_backend())

        # Fallback sem backend forçado
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.src)

        if not cap.isOpened():
            print(f"[CameraThread] ERRO: não abriu a câmera (src={self.src})")
            self.running = False
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("[CameraThread] aviso: frame não recebido, tentando reabrir...")
                cap.release()
                time.sleep(0.5)
                cap = cv2.VideoCapture(self.src, cam_backend())
                continue

            self.queue = [frame.copy()]
            self.frame_captured.emit(frame)
            self.msleep(30)

        cap.release()

    def stop(self):
        self.running = False


# ---------------------------------------------------------
# THREAD DO MODELO
# ---------------------------------------------------------
class ModelThread(QThread):
    detections_ready = pyqtSignal(list)

    def __init__(self, model, camera_thread):
        super().__init__()
        self.model = model
        self.running = True
        self.camera_thread = camera_thread

    def run(self):
        while self.running:
            if self.camera_thread.queue:
                frame = self.camera_thread.queue[-1]

                try:
                    results = self.model.predict(frame, verbose=False, device="cpu")
                except Exception as e:
                    print("[ModelThread] predict falhou:", e)
                    self.msleep(200)
                    continue

                detections = []
                for r in results:
                    for box in r.boxes:
                        detections.append({
                            "cls": int(box.cls[0]),
                            "conf": float(box.conf[0]),
                            "bbox": box.xyxy[0].tolist()
                        })
                self.detections_ready.emit(detections)

            self.msleep(50)

    def stop(self):
        self.running = False


# ---------------------------------------------------------
# JANELA PRINCIPAL (PyQt)
# ---------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Argus IA – Detecção de EPI")

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.label)

        # Iniciar câmera
        self.camera_thread = CameraThread(src=0)
        self.camera_thread.frame_captured.connect(self.update_frame)
        self.camera_thread.start()

        # Iniciar modelo
        self.model = YOLO("yolo11n.pt")
        self.model_thread = ModelThread(self.model, self.camera_thread)
        self.model_thread.detections_ready.connect(self.process_detections)
        self.model_thread.start()

    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(img))

    def process_detections(self, detections):
        for d in detections:
            cls = d["cls"]
            conf = d["conf"]

            # ----------------------------------------
            # CONDICIONAIS DE EXERCÍCIOS (ATUALIZADAS)
            # ----------------------------------------

            # Exemplo de decisão: cadeira romana
            if cls == 0:  # supondo classe 0 → pessoa usando cadeira romana
                if conf > 0.6:
                    print("Cadeira Romana detectada — execução dentro do padrão.")
                else:
                    print("Cadeira Romana detectada, baixa confiança. Checar postura.")

            # Exemplo para outra classe
            if cls == 1:
                print("Óculos de EPI detectado")

            # Caso ocorra um exercício trocado ou incorreto
            if cls not in [0,1]:
                print("Objeto inesperado detectado")

    def closeEvent(self, event):
        self.camera_thread.stop()
        self.model_thread.stop()
        event.accept()


# Execução
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(800, 600)
    w.show()
    sys.exit(app.exec())
