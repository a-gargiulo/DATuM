import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import pdb

def load_pkl(pkl_file: str) -> dict:
    """Load data from a .pkl file.

    :param pkl_file: File in .pkl format.

    :return: Pickle file content.
    :rtype: dict
    """
    with open(pkl_file, "rb") as f:
        data = pkl.load(f)
    return data


data = load_pkl("../../outputs/plane1_650k/plane1_pr.pkl")

for pp in data.keys():
    pr = data[pp]["exp"]
    u = pr["mean_velocity"]["U"]
    y = pr["coordinates"]["Y"] - pr["properties"]["Y_CORRECTION"]

    fig, ax = plt.subplots()
    plt.plot(u, y)
    ax.set_xlabel("u")
    ax.set_ylabel("y")
    cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

    def on_click(event):
        if event.inaxes == ax:
            Ue = event.xdata
            idx = np.where(u >= 0.99 * Ue)[0][0]
            delta = y[idx]
            delta_star = np.trapz(1- u[~np.isnan(u)]/Ue, y[~np.isnan(u)])
            theta = np.trapz((1- u[~np.isnan(u)]/Ue)*(u[~np.isnan(u)]/Ue), y[~np.isnan(u)])
            print(Ue)
            print(delta)
            print(delta_star)
            print(theta)
            plt.close(fig)  # Close the plot after click

    # Connect click event
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()



