"""This module contains global configuration parameters."""

import platform

import matplotlib.pyplot as plt

system = platform.system()

STYLES = {
    "pad": {"small": 5, "medium": 10, "large": 20},
    "color": {
        "base": "#262626",
        "s1_header": "#2a262f",
        "s1_content": "#413d46",
        "s2_header": "#1e1e1e",
        "s2_content": "#373737",
        "accent": "#bb86fc",
    },
    "font": "Avenir" if system == "Darwin" else "Segoe UI",
    "font_size": {"regular": 12 if system == "Darwin" else 10},
}

RC_PARAMS = {
    "font.size": 14,
    "axes.linewidth": 2,
    "lines.linewidth": 2,
    "xtick.direction": "in",
    "xtick.major.width": 2,
    "ytick.direction": "in",
    "ytick.major.width": 2,
}


def apply_rc_params():
    """Apply the rcParams from RC_PARAMS dictionary."""
    plt.rcParams.update(RC_PARAMS)
