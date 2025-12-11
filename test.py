import serial, time, random

PORT = "COM5"          # troque para o seu (Linux: /dev/ttyUSB0)
BAUD   = 115200

def send_color(ser: serial.Serial, r: int, g: int, b: int, brightness: int):
    ser.write(bytes([r, g, b, brightness]))

def rainbow_cycle(ser: serial.Serial, delay=0.05):
    for j in range(256):
        r = int((255 & (j << 0)) * 1)
        g = int((255 & (j << 8)) * 1)
        b = int((255 & (j << 16)) * 1)
        send_color(ser, r, g, b, 80)
        time.sleep(delay)

def blink_red(ser: serial.Serial, times: int = 3, on_ms: float = 0.15, off_ms: float = 0.15, brightness: int = 70):
    """Pisca fita em vermelho <times> vezes (padrão 3)."""
    for _ in range(times):
        send_color(ser, 255, 0, 0, brightness)   # liga
        time.sleep(on_ms)
        send_color(ser, 0, 0, 0, 0)
        time.sleep(off_ms)
    # volta ao estado anterior (opcional)
    send_color(ser, 0, 0, 0, 0)

if __name__ == "__main__":
    with serial.Serial(PORT, BAUD, timeout=1) as ser:
        time.sleep(2)  # boot Arduino
        print("Piscando vermelho...")
        while True:
            blink_red(ser, times=5, on_ms=0.1, off_ms=0.1)  # 5x rápido