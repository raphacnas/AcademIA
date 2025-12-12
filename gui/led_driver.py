# led_driver.py
import serial
from typing import Optional

class LedDriver:
    """
    Driver minimalista para LED RGB via serial (4 bytes: R, G, B, brightness).
    Quem controla o piscar é o MainWindow.
    """

    def __init__(self, port: str = "COM5", baud: int = 115200):
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
        except serial.SerialException:
            self.ser = None   # Arduino não conectado → ignora silenciosamente

    # --------------------------------------------------
    # primitives
    # --------------------------------------------------
    def send(self, r: int, g: int, b: int, brightness: int = 150) -> None:
        """Envia cor instantânea (0-255)."""
        if self.ser and self.ser.is_open:
            self.ser.write(bytes([r, g, b, brightness]))

    def off(self) -> None:
        self.send(0, 0, 0, 0)

    def green(self, brightness: int = 150) -> None:
        self.send(0, 255, 0, brightness)

    def red(self, brightness: int = 150) -> None:
        self.send(255, 0, 0, brightness)
