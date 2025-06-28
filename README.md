
# 🌀 Fractal Explorer: Infinite Mathematical Beauty

![Fractal Visualization](fractal_explorer/resources/Screenshot.png)

**Fractal Explorer** is an interactive desktop application that lets you explore the infinite complexity of mathematical fractals with real-time rendering and beautiful visualizations. Discover the hidden beauty of the Mandelbrot set and other fractals with intuitive controls and stunning color palettes.

---

## ✨ Features

- ⚡ Real-time fractal exploration with smooth zooming and panning
- 🔄 Expanded fractal types: Mandelbrot, Julia, Burning Ship, Tricorn, Celtic Mandelbrot, Buffalo, Lyapunov, **Mandelbar**, **Perpendicular Burning Ship**, **Perpendicular Buffalo**
- 🎨 Customizable color maps with scientifically curated palettes
- ✨ **Real-time Fractal Evolution**: Animate fractal parameters (Julia C, Exponent, Iterations) to see them morph and change dynamically.
- 🧮 Adjustable iteration count for high-precision zooms
- 🖼️ High-resolution image export
- 🧠 Hardware-optimized rendering using Numba  
- 🧩 Modular structure for easy customization and extension  

---

## 💻 Installation

### 1. Prerequisites

- Python **3.9+**

### 2. Clone the Repository

```bash
git clone https://github.com/yamen1995/fractal-explorer.git
cd fractal-explorer
```

### 3. Set Up Virtual Environment

```bash
python -m venv venv
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Build the Application

```bash
python setup.py
```

### 6. Run the Executable

Find the app at:

```
dist/FractalExplorer.exe
```

---

## 📖 User Guide

### 🖱 Basic Controls

- **Zoom In/Out:** Use the mouse wheel at the cursor location  
- **Zoom with Selection:** Left-click and drag to zoom into a specific region  
- **Pan:** Click and drag with the left mouse button  

---

### ⚙️ Fractal Settings

- **Iterations:**  
  Controls fractal detail level  
  - Use higher values for deep zooms  
  - Recommended max: **5000**

- **Fractal Set:**  
  Choose your fractal type:  
  - `Mandelbrot` – the classic infinite set
  - `Julia` – shaped by complex constants
  - `Burning Ship` – sharp, flame-like structures
  - `Tricorn` – symmetrical and mirrored patterns
  - `Celtic Mandelbrot` – variation with absolute value on real component
  - `Buffalo` – variation with absolute values on both components
  - `Lyapunov` – based on logistic map sequences
  - `Mandelbar` – similar to Tricorn, using conjugate Z
  - `Perpendicular Burning Ship` – variation with swapped real/imaginary components before power
  - `Perpendicular Buffalo` – variation with swapped real/imaginary components before power

---

### 🎞️ Fractal Evolution

- **Animate Parameters:** Bring fractals to life by animating key variables:
  - `Julia C Real/Imag`: Watch Julia sets transform as their constant changes.
  - `Exponent Real/Imag`: See how changing the power alters fractal shapes.
  - `Iterations`: Observe the detail emerge or recede as iterations sweep through a range.
- **Controls:**
  - **Animate Dropdown:** Select the parameter you want to change over time.
  - **Start/End Values:** Define the range for the selected parameter.
  - **Steps:** Set the number of frames in your animation.
  - **Speed (FPS):** Control the playback speed of the animation.
  - **Start/Stop Animation:** Begin or halt the evolution process.

---

### 🎨 Color Settings

- **Colormap:**  
  Choose from beautiful palettes:
  - Plasma (default)
  - Viridis
  - Inferno
  - Magma
  - Jet
  - Fire, Ocean, Forest, Ice, Candy (and more!)

- **Julia c:**  
  - Input your own constant (real + imaginary)
  - Or use a preset like:
    - `-0.7 + 0.27015i` (classic)
    - `-0.4 + 0.6i` (spiral)
    - `0.285 + 0.01i` (swirls)

---

### 🖼️ Rendering & Saving

- **Render Button:** Apply new settings  
- **Save Image:** Save current view as PNG or JPG  
- **Reset View:** Return to default zoom and center  

---

## 💡 Tips & Tricks

- **Zoom Strategy:**  
  Start with low iterations, then increase once you find an interesting region

- **Color Play:**  
  Different palettes highlight different features — try Inferno or Jet for contrast

- **Julia Patterns:**  
  Tiny tweaks to `c` generate wildly different results

- **Performance Tips:**  
  - Close heavy background apps  
  - Reduce iterations during navigation  
  - Use smaller windows for faster feedback

---

## ⚙️ Technical Overview

- Python 3.9+  
- PyQt5 UI with responsive resizing  
- Numba-accelerated pixel rendering  
- Modular layout:  
  - `fractal_math.py` — fractal logic  
  - `ui.py` — interface  
  - `colormaps.py` — color schemes  

---

## 🧩 Extend the App

The code is organized for easy customization:
- Add new fractal types via `fractal_math.py`
- Add more color maps to `colormaps.py`
- Extend the UI via `ui.py`

---
## Update 1.1 (This Branch)
- Branch: `feature/more-fractals-evolution-readme-version`
- **New Fractal Types Added:**
    - Mandelbar
    - Perpendicular Burning Ship
    - Perpendicular Buffalo
- **New Feature: Real-time Fractal Evolution:**
    - Animate parameters like Julia `c` (real/imaginary parts), exponent (real/imaginary parts), and iterations.
    - UI controls for selecting parameter, start/end values, number of steps, and animation speed (FPS).
- **Application Version Bumped:** to v1.1.
- **README Updated:** Reflecting all new features and fractal types.

## Previous Update (Update 1)
- Branch: `feature/fractal-blending-and-exponents`
1. New feature: Color map blending (linear, nonlinear, segment)
2. New feature: Fractal blending (mask, alternating: checker, vertical, horizontal)
3. New feature: Real and complex exponent support
4. Fix: Removed parallel processing in favor of callback-based rendering

## 🏁 Credits

Developed by **Yamen Tahseen**  
Built with:  
🧠 Python • 🎨 PyQt5 • 🔢 NumPy • ⚡ Numba

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo  
2. Create a new feature branch  
3. Submit a pull request

Please follow [PEP8](https://peps.python.org/pep-0008/) and include appropriate comments/tests.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for full terms.

---

> 🧠 Discover the infinite complexity of mathematics — each zoom reveals brand-new, never-before-seen structures.  
> Happy fractaling!
