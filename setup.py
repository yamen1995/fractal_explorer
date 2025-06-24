import os
import PyInstaller.__main__


SCRIPT = "fractal_explorer/fractal_explorer/main.py"


APP_NAME = "FractalExplorer"


ICON_PATH = "fractal_explorer/fractal_explorer/resources/icon.ico"


def build_exe():
    cmd = [
        "--name=%s" % APP_NAME,
        "--onefile",
        "--windowed", 
    ]

    if ICON_PATH:
        cmd.append(f"--icon={ICON_PATH}")

    cmd.append(SCRIPT)

    PyInstaller.__main__.run(cmd)

if __name__ == "__main__":
    build_exe()