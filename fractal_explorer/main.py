import sys
from PyQt5.QtWidgets import QApplication
from fractal_explorer.ui import set_dark_palette, FractalExplorer

if __name__ == "__main__":
    """
    Main entry point for the Fractal Explorer application.
    Initializes the QApplication, applies a dark theme, creates,
    and shows the main FractalExplorer window.
    """
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Recommended for a consistent look

    # Apply the custom dark palette
    # This function should be defined in ui.py or a utility module
    # and should set the QPalette for the application.
    set_dark_palette(app)

    window = FractalExplorer()
    window.show()

    sys.exit(app.exec_())