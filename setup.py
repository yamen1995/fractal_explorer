import os
import PyInstaller.__main__

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(BASE_DIR, "fractal_explorer", "main.py")
ICON_PATH = os.path.join(BASE_DIR, "fractal_explorer", "resources", "icon.ico")

APP_NAME = "FractalExplorer"

def build_exe():
    cmd = [
        f"--name={APP_NAME}",
        "--onefile",
        "--windowed",
        f"--icon={ICON_PATH}",
        SCRIPT
    ]
    PyInstaller.__main__.run(cmd)

if __name__ == "__main__":
    build_exe()