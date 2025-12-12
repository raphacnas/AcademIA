# main.py
import faulthandler, traceback, sys
faulthandler.enable()                       # crash → stdout
sys.excepthook = lambda t, v, tb: traceback.print_exception(t, v, tb)

from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())