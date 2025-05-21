"""Testing module for 1D stereo PIV profile data."""
import numpy as np
import pickle as pkl
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, cm

from datum.core.my_types import Coordinates
from datum.core.beverli import Beverli


RC_PARAMS = {
    "font.size": 18,
    "font.family": "Avenir",
    "axes.linewidth": 2,
    "axes.labelpad": 10,
    "lines.linewidth": 1,
    "xtick.direction": "in",
    "xtick.major.width": 2,
    "xtick.major.size": 4,
    "xtick.minor.size": 3,
    "xtick.major.pad": 10,
    "ytick.direction": "in",
    "ytick.major.width": 2,
    "ytick.major.size": 4,
    "ytick.minor.size": 3,
    "ytick.major.pad": 10,
}
plt.rcParams.update(RC_PARAMS)


def plot_contour(
    coordinates: Coordinates,
    quantity: NDArray[np.float64],
    hill_orientation: float,
    x3: float,
    **kwargs
) -> None:
    """
    Generate a contour plot of the 2D stereo PIV data.

    :param coordinates: Coordinates of the PIV plane.
    :param quantity: Values of the PIV quantity to be plotted.
    :param hill_orientation: BeVERLI hill geometry orientation.
    :param x3: PIV plane x3 coordinate.
    """
    geometry = Beverli(hill_orientation, True)
    cmap = plt.get_cmap(kwargs.get("cmap", "jet"))
    cmin = kwargs.get("cmin", 0)
    cmax = kwargs.get("cmax", 25)
    cnum = kwargs.get("cnum", 100)
    xlim = kwargs.get("xlim", [-1.85, -1.80])
    ylim = kwargs.get("ylim", [-0.02, 0.1])
    xmajor_locator = kwargs.get("xmajor_locator", None)
    ymajor_locator = kwargs.get("ymajor_locator", None)
    cnum_ticks = kwargs.get("cnum_ticks", 6)
    cbar_label = kwargs.get("cbar_label", r"$\overline{U}_1$ (m/s)")

    bounds = np.linspace(cmin, cmax, cnum)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.pcolormesh(
        coordinates["X"], coordinates["Y"], quantity, norm=norm, cmap=cmap
    )

    x1b, x2b = geometry.calculate_x1_x2(x3)
    ax.plot(x1b, x2b, linestyle="--", color="k")
    ax.tick_params(axis="x", pad=10)
    ax.tick_params(axis="y", pad=10)
    ax.set_xlabel(r"$x_1$ (m)", labelpad=10)
    ax.set_ylabel(r"$x_2$ (m)", labelpad=10)
    ax.xaxis.set_tick_params(which="major", size=8, width=2, direction="in")
    ax.xaxis.set_tick_params(which="minor", size=5, width=1.5, direction="in")
    ax.yaxis.set_tick_params(which="major", size=8, width=2, direction="in")
    ax.yaxis.set_tick_params(which="minor", size=5, width=1.5, direction="in")
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    if xmajor_locator:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xmajor_locator))
    if ymajor_locator:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(ymajor_locator))

    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap), orientation="vertical",
        cax=ax.inset_axes((1.05, 0, 0.075, 1)),
    )
    cbar.set_ticks(np.linspace(cmin, cmax, cnum_ticks).tolist())
    cbar.set_label(cbar_label, labelpad=10)
    cbar.ax.tick_params(width=2)
    cbar.ax.minorticks_off()

    fig.savefig("contour.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    PIV_FILE = "./outputs/plane3/plane3_pp.pkl"
    with open(PIV_FILE, "rb") as f:
        piv = pkl.load(f)

    plot_contour(
        piv["coordinates"],
        piv["mean_velocity"]["U"],
        hill_orientation=45,
        x3=0,
        cmap="jet",
        cmin=0,
        cmax=25,
        cnum=100,
        cnum_ticks=6,
        xlim=[-0.5, -0.4],
        ylim=[0.0, 0.125],
        xmajor_locator=0.05,
        ymajor_locator=0.05,
    )
