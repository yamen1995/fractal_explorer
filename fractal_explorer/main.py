import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from fractal_explorer.ui import set_dark_palette, FractalExplorer

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    set_dark_palette(app)
    window = FractalExplorer()
    window.show()
    sys.exit(app.exec_())