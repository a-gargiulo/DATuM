"""Plotting code."""
import numpy as np
import platform
import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker
from matplotlib.widgets import Cursor
from datum.core.my_types import NestedDict

from datum.core.beverli import Beverli
from typing import cast, Dict, Sequence, List, Tuple


def plot_contour(
    coordinates: Dict[str, np.ndarray],
    quantity: np.ndarray,
    properties: NestedDict,
    geometry: Beverli,
    outname: str,
) -> None:
    """Generate a contour plot for a specific quantity of the BeVERLI stereo PIV data.

    :param coordinates: Dictionary of NumPy ndarrays of shape (m, n) representing the
        Cartesian coordinates of the PIV quantity in the x:sub:`1`- and the
        x:sub:`2`-direction, where m and n represent the number of data points in the
        x:sub:`1`- and the x:sub:`2`-direction, respectively.
    :param quantity: NumPy ndarray of shape (m, n) representing the
        PIV quantity, where m and n represent the number of data points in the
        x:sub:`1`- and the x:sub:`2`-direction, respectively.
    :param properties: Nested dictionary containing the desired plot properties.
    :param outname: String representing the output figure's file name.
    """
    if platform.system() == "Windows":
        plt.rcParams["font.family"] = "Franklin Gothic Book"
    else:
        plt.rcParams["font.family"] = "Avenir"
    plt.rcParams["font.size"] = "18"
    plt.rcParams["lines.linewidth"] = "2"
    plt.rcParams["axes.linewidth"] = "2"

    cmap = plt.get_cmap(cast(str, properties["colormap"]))
    bounds = np.linspace(
        cast(int, properties["contour_range"]["start"]),
        cast(int, properties["contour_range"]["end"]),
        cast(int, properties["contour_range"]["num_of_contours"]),
    )
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(3, 3))
    axs = fig.add_axes((0, 0, 1, 1))
    axs.pcolor(coordinates["X"], coordinates["Y"], quantity, norm=norm, cmap=cmap)
    x1b, x2b = geometry.calculate_x1_x2(cast(float, properties["zpos"]))
    axs.plot(x1b, x2b, linestyle="--", color="k")
    axs.tick_params(axis="x", pad=10)
    axs.tick_params(axis="y", pad=10)

    axs.set_xlabel(r"$x_1$ (m)", labelpad=5)
    axs.set_ylabel(r"$x_2$ (m)", labelpad=10)

    axs.xaxis.set_tick_params(which="major", size=8, width=2, direction="in")
    axs.xaxis.set_tick_params(which="minor", size=5, width=1.5, direction="in")
    axs.yaxis.set_tick_params(which="major", size=8, width=2, direction="in")
    axs.yaxis.set_tick_params(which="minor", size=5, width=1.5, direction="in")

    axs.set_xlim(cast(int, cast(list, properties["xlim"])[0]), cast(int, cast(list, properties["xlim"])[1]))
    axs.set_ylim(cast(int, cast(list, properties["ylim"])[0]), cast(int, cast(list, properties["ylim"])[1]))

    axs.xaxis.set_major_locator(
        ticker.MultipleLocator(cast(float, properties["xmajor_locator"]))
    )
    if properties["ymajor_locator"]:
        axs.yaxis.set_major_locator(
            ticker.MultipleLocator(cast(float, properties["ymajor_locator"]))
        )

    # divider = make_axes_locatable(axs)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap), orientation="vertical",
        cax=axs.inset_axes((1.15, 0, 0.075, 1))
    )
    cbar.set_ticks(
        cast(Sequence[float], np.linspace(
            cast(int, cast(dict, properties["cbar_range"])["start"]),
            cast(int, cast(dict, properties["cbar_range"])["end"]),
            cast(int, cast(dict, properties["cbar_range"])["num_of_ticks"]),
        ))
    )
    cbar.set_label(cast(str, properties["cbar_label"]), labelpad=10)
    cbar.ax.tick_params(width=2)
    cbar.ax.minorticks_off()

    fig.savefig(
        outname,
        format="png",
        dpi=300,
        backend="agg",
        transparent=False,
        bbox_inches="tight",
    )


def point_selector(
    number_of_points: int,
    coordinates: Dict[str, np.ndarray],
    quantity: np.ndarray,
    properties: NestedDict,
    geometry: Beverli,
) -> List[Tuple[float, float]]:
    """
    Generate a contour plot and selector to select the spatial locations for the profile extraction.

    :param number_of_points: The number of spatial points (profiles).
    :param coordinates: The coordinates of the PIV plane.
    :param quantity: The values of the PIV quantity to be plotted.
    :param properties: The fluid, flow, and reference properties.
    :param geoemtry: An object containing the BeVERLI hill geometry.

    :return: A list of tuples containing the selected locations.
    :rtype: List[Tuple[float, float]]
    """
    cmap = plt.get_cmap(cast(str, properties["colormap"]))
    bounds = np.linspace(
        cast(int, properties["contour_range"]["start"]),
        cast(int, properties["contour_range"]["end"]),
        cast(int, properties["contour_range"]["num_of_contours"]),
    )
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    axs.pcolormesh(coordinates["X"], coordinates["Y"], quantity, norm=norm, cmap=cmap)
    x1b, x2b = geometry.calculate_x1_x2(cast(float, properties["zpos"]))
    axs.plot(x1b, x2b, linestyle="--", color="k")
    axs.tick_params(axis="x", pad=10)
    axs.tick_params(axis="y", pad=10)

    axs.set_xlabel(r"$x_1$ (m)", labelpad=5)
    axs.set_ylabel(r"$x_2$ (m)", labelpad=10)

    axs.xaxis.set_tick_params(which="major", size=8, width=2, direction="in")
    axs.xaxis.set_tick_params(which="minor", size=5, width=1.5, direction="in")
    axs.yaxis.set_tick_params(which="major", size=8, width=2, direction="in")
    axs.yaxis.set_tick_params(which="minor", size=5, width=1.5, direction="in")

    axs.set_xlim(cast(int, cast(list, properties["xlim"])[0]), cast(int, cast(list, properties["xlim"])[1]))
    axs.set_ylim(cast(int, cast(list, properties["ylim"])[0]), cast(int, cast(list, properties["ylim"])[1]))
    axs.xaxis.set_major_locator(
        ticker.MultipleLocator(cast(float, properties["xmajor_locator"]))
    )
    if properties["ymajor_locator"]:
        axs.yaxis.set_major_locator(
            ticker.MultipleLocator(cast(float, properties["ymajor_locator"]))
        )

    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap), orientation="vertical",
        cax=axs.inset_axes((1.15, 0, 0.075, 1))
    )

    cbar.set_ticks(
        cast(Sequence[float], np.linspace(
            cast(int, cast(dict, properties["cbar_range"])["start"]),
            cast(int, cast(dict, properties["cbar_range"])["end"]),
            cast(int, cast(dict, properties["cbar_range"])["num_of_ticks"]),
        ))
    )
    cbar.set_label(cast(str, properties["cbar_label"]), labelpad=10)
    cbar.ax.tick_params(width=2)
    cbar.ax.minorticks_off()

    _ = Cursor(axs, useblit=True, color="gray", linewidth=1)
    zoom_ok = False
    print("\nZoom or pan to view, \npress spacebar when ready to click:\n")
    while not zoom_ok:
        zoom_ok = plt.waitforbuttonpress()
    print("Click once to select location matching global reference point...\n\n")
    pts = plt.ginput(n=number_of_points, timeout=0, show_clicks=True)

    plt.close()
    return cast(List[Tuple[float, float]], pts)
