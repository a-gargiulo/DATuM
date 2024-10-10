import platform
import matplotlib.pyplot as plt

system = platform.system()

# Buttons
if system == "Darwin":
    from tkmacosx import Button
else:
    from tkinter import Button

# Fonts
if system == "Darwin":
    default_font = ("Avenir", 12)
elif system == "Windows":
    default_font = ("Franklin Gothic Book", 12)

colors = {
    "base": "#262626",
    "f1_header": "#2a262f",
    "f1_content": "#413d46",
    "f2_header": "#1e1e1e",
    "f2_content": "#373737",
    "accent": "#bb86fc",
}


plt.rcParams.update({
    "font.size": 14,
    "axes.linewidth": 2,
    "lines.linewidth": 2,
    "xtick.direction": "in",
    "xtick.major.width": 2,
    "ytick.direction": "in",
    "ytick.major.width": 2
})
