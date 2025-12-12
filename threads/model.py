from PyQt6.QtCore import QThread, pyqtSignal
import cv2
import numpy as np
import time
from ultralytics import YOLO
from core.config import YOLO_PATH


model = YOLO(YOLO_PATH)


class ModelThread(QThread):
    result_ready = pyqtSignal(object, np.ndarray)  # results, frame

    def __init__(self, cam_thread: "CameraThread"):
        super().__init__()
        self.cam_thread = cam_thread
        self.running = True

    def run(self):
        while self.running:
            frame = self.cam_thread.get_latest()
            if frame is not None:
                # frame = cv2.flip(frame, 0)
                try:
                    results = model.predict(frame, verbose=False, device="cpu")
                except Exception as e:
                    print("[ModelThread] predict falhou:", e)
                    self.msleep(200)
                    continue
                self.result_ready.emit(results, frame)
            self.msleep(30)

    def stop(self):
        self.running = False
        self.wait()