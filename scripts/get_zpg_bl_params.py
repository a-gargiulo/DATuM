"""Calculate integral parameters of a zero-pressure-gradient boundary layer.

This script computes integral boundary layer quantities from streamwise
velocity profiles obtained via stereo PIV measurements in regions of zero
pressure gradient (ZPG).
"""

import pickle as pkl
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Cursor




def load_pkl(pkl_file: str) -> dict:
    """
    Load data from a pickle (.pkl) file.

    :param pkl_file: Path to the pickle file.
    :return: Contents of the pickle file.
    """
    with open(pkl_file, "rb") as f:
        data = pkl.load(f)
    return data


if len(sys.argv) < 3:
    print(
        "[ERROR] Missing inputs: please provide the path to the PIV profile "
        "`.pkl` file and the velocity threshold as a command-line argument."
    )
    sys.exit(1)

data = load_pkl(sys.argv[1])

BL_THRESHOLD = float(sys.argv[2])

for pp in data.keys():
    pr = data[pp]["exp"]

    u = pr["mean_velocity"]["U_SS_MODELED"]
    # y = pr["coordinates"]["Y"] - pr["properties"]["Y_CORRECTION"]
    y = pr["coordinates"]["Y_SS_MODELED"]

    fig, ax = plt.subplots()
    plt.plot(u, y)
    ax.set_xlabel("U [m/s]")
    ax.set_ylabel("y [m]")
    cursor = Cursor(ax, useblit=True, color="red", linewidth=1)

    def calculate_bl_params(event):
        if event.inaxes == ax:
            Ue = event.xdata
            idx = np.where(u >= BL_THRESHOLD * Ue)[0][0]
            delta = y[idx]
            delta_star = np.trapz(
                1 - u[~np.isnan(u)] / Ue,
                y[~np.isnan(u)]
            )
            theta = np.trapz(
                (1 - u[~np.isnan(u)] / Ue) * (u[~np.isnan(u)] / Ue),
                y[~np.isnan(u)],
            )
            print(
                "Boundary layer edge velocity, Ue [m/s]: ".ljust(60) +
                f"{Ue:.2f}"
            )
            print(
                "Boundary layer thickness, delta [m]: ".ljust(60) +
                f"{delta:.4f}")
            print(
                "Boundary layer displacement thickness, delta* [m]: ".ljust(60) +
                f"{delta_star:.4f}"
            )
            print(
                "Boundary layer momentum thickness, theta [m] : ".ljust(60) +
                f"{theta:.4f}"
            )
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", calculate_bl_params)
    plt.show()
