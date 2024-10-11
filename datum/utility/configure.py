import platform
import matplotlib.pyplot as plt


system = platform.system()

if system == "Darwin":
    default_font = ("Avenir", 12)
elif system == "Windows":
    default_font = ("Segoe UI", 12)

colors = {
    "base": "#262626",
    "s1_header": "#2a262f",
    "s1_content": "#413d46",
    "s2_header": "#1e1e1e",
    "s2_content": "#373737",
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
