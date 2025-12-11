from PyQt6.QtCore import QThread, pyqtSignal
import cv2
import time
import numpy as np
from core.config import cam_backend


class CameraThread(QThread):
    frame_captured = pyqtSignal(np.ndarray)

    def __init__(self, src: int = 0, qsize: int = 2):
        super().__init__()
        self.src = src
        self.qsize = qsize
        self.running = True
        self.queue: list = []

    def run(self):
        print("[CAM] iniciando...")
        cap = cv2.VideoCapture(self.src, cam_backend())
        if not cap.isOpened():
            print(f"[CAM] FAIL – câmera {self.src} não abriu")
            self.running = False
            return
        print("[CAM] câmera aberta")


        # tentativa de reduzir latência
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("[CAM] frame vazio – reconectando")
                cap.release()
                time.sleep(0.5)
                # tenta reconectar
                cap = cv2.VideoCapture(self.src, cam_backend())
                if not cap.isOpened():
                    cap = cv2.VideoCapture(self.src)
                if not cap.isOpened():
                    self.msleep(200)
                    continue
            else:
                print("[CAM] frame capturado", frame.shape)

            self.queue = [frame.copy()]
            self.frame_captured.emit(frame)
            self.msleep(30)

        try:
            cap.release()
        except Exception:
            pass

    def get_latest(self):
        return self.queue[-1] if self.queue else None

    def stop(self):
        self.running = False
        self.wait()