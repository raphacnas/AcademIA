import serial
from PyQt6.QtCore import QTimer

class LedDriver:
    def __init__(self, port: str = "COM5", baud: int = 115200):
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
        except serial.SerialException:
            self.ser = None   # Arduino não conectado → ignora silenciosamente

    def send_color(self, r: int, g: int, b: int, brightness: int = 150):
        if self.ser and self.ser.is_open:
            self.ser.write(bytes([r, g, b, brightness]))

    def blink_error(self, times: int = 2, on_ms: int = 120, off_ms: int = 80):
        """Pisca vermelho <times> vezes sem bloquear."""
        if not self.ser:
            return
        for i in range(times):
            QTimer.singleShot(i * (on_ms + off_ms), lambda: self.send_color(255, 0, 0, 150))
            QTimer.singleShot(i * (on_ms + off_ms) + on_ms, lambda: self.send_color(0, 0, 0, 0))

    def blink_success(self, times: int = 2, on_ms: int = 60, off_ms: int = 60):
        """Pisca verde quando rep foi perfeita."""
        if not self.ser:
            return
        for i in range(times):
            QTimer.singleShot(i * (on_ms + off_ms),
                              lambda: self.send_color(0, 255, 0, 80))   # verde
            QTimer.singleShot(i * (on_ms + off_ms) + on_ms,
                              lambda: self.send_color(0, 0, 0, 0))