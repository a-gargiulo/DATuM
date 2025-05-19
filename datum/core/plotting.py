"""Plotting code."""
import numpy as np
import platform
import matplotlib.pyplot as plt
from matplotlib import cm, colors, ticker
from matplotlib.widgets import Cursor
from datum.core.my_types import NestedDict

from datum.core.beverli import Beverli
from typing import Any, cast, Dict, Sequence, List, Tuple, Optional
from datum.core.my_types import Coordinates
from tkinter import messagebox
from datum.utility.logging import logger


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


def points_selector(
    num_pts: int,
    coordinates: Coordinates,
    quantity: np.ndarray,
    settings: NestedDict,
    geometry: Beverli,
) -> Tuple[Tuple[float, float], ...]:
    """
    Generate a contour plot and selector to select the spatial locations for the profile extraction.

    :param number_of_points: The number of spatial points (profiles).
    :param coordinates: The coordinates of the PIV plane.
    :param quantity: The values of the PIV quantity to be plotted.
    :param properties: The fluid, flow, and reference properties.
    :param geoemtry: An object containing the BeVERLI hill geometry.

    :return: A list of tuples containing the selected locations, or 'None' in case of an error
    :rtype: Optional[List[Tuple[float, float]]]
    """
    # Use casts, since this function is only used internally
    cmap = plt.get_cmap(cast(str, settings["colormap"]))
    cstart = cast(int, settings["contour_range"]["start"])
    cend = cast(int, settings["contour_range"]["end"])
    cnum = cast(int, settings["contour_range"]["num_of_contours"])
    zpos = cast(float, settings["zpos"])
    xlim = cast(List[float], settings["xlim"])
    ylim = cast(List[float], settings["ylim"])
    xmajor_locator = cast(float, settings["xmajor_locator"])
    ymajor_locator = None
    if settings["ymajor_locator"]:
        ymajor_locator = cast(float, settings["ymajor_locator"])
    cbar_start = cast(int, settings["cbar_range"]["start"])
    cbar_end = cast(int, settings["cbar_range"]["end"])
    cbar_nticks = cast(int, settings["cbar_range"]["num_of_ticks"])
    cbar_label = cast(str, settings["cbar_label"])

    bounds = np.linspace(cstart, cend, cnum)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    axs.pcolormesh(
        coordinates["X"], coordinates["Y"], quantity, norm=norm, cmap=cmap
    )
    x1b, x2b = geometry.calculate_x1_x2(zpos)
    axs.plot(x1b, x2b, linestyle="--", color="k")

    axs.tick_params(axis="x", pad=10)
    axs.tick_params(axis="y", pad=10)
    axs.set_xlabel(r"$x_1$ (m)", labelpad=5)
    axs.set_ylabel(r"$x_2$ (m)", labelpad=10)
    axs.xaxis.set_tick_params(which="major", size=8, width=2, direction="in")
    axs.xaxis.set_tick_params(which="minor", size=5, width=1.5, direction="in")
    axs.yaxis.set_tick_params(which="major", size=8, width=2, direction="in")
    axs.yaxis.set_tick_params(which="minor", size=5, width=1.5, direction="in")
    axs.set_xlim(xlim[0], xlim[1])
    axs.set_ylim(ylim[0], ylim[1])
    axs.xaxis.set_major_locator(ticker.MultipleLocator(xmajor_locator))
    if ymajor_locator:
        axs.yaxis.set_major_locator(ticker.MultipleLocator(ymajor_locator))

    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap), orientation="vertical",
        cax=axs.inset_axes((1.15, 0, 0.075, 1)),
    )
    cbar.set_ticks(np.linspace(cbar_start, cbar_end, cbar_nticks).tolist())
    cbar.set_label(cbar_label, labelpad=10)
    cbar.ax.tick_params(width=2)
    cbar.ax.minorticks_off()

    _ = Cursor(axs, useblit=True, color="gray", linewidth=1)
    zoom_ok = False
    messagebox.showinfo(
        "INFO",
        "Zoom or pan to view, press spacebar when ready to click."
    )
    while not zoom_ok:
        zoom_ok = plt.waitforbuttonpress()
    messagebox.showinfo(
        "INFO",
        "Click once to select the profile locations.")
    pts = plt.ginput(n=num_pts, timeout=0, show_clicks=True)
    if pts is None:
        raise RuntimeError("Failed to select profile points.")

    plt.close()
    return tuple([(float(a), float(b)) for a, b in pts])


def profile_reconstructor(
    wall_model: List[np.ndarray],
    data: List[np.ndarray],
    add_points: bool,
    number_of_added_points: Optional[int] = None
) -> Tuple[List[Tuple[float, float]], int, int]:
    """Compare an experimental hill-normal profile to the law of the wall.

    Select a lower and upper threshold capturing the range of usable data.Add optional near-wall reconstruction points.

    :param wall_model: The law of the wall data.
    :param data: The experimental hill-normal profile data.
    :param add_points: Boolean indicating whether to add near-wall reconstruction points.
    :param number_of_added_points: Number of reconstruction points.

    :return: A tuple containing the reconstruction points and the lower and upper threshold index of the profile. If the
        return value is 'None' instead, an error has occured.
    :rtype: Optional[Tuple[Optional[List[Tuple[float, float]]], float, float]]
    """
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))

    axs.semilogx(wall_model[0], wall_model[1], color="red")
    axs.semilogx(data[0], data[1], linestyle="none", color="blue", marker="o")

    _ = Cursor(axs, useblit=True, color="gray", linewidth=1)
    zoom_ok = False
    messagebox.showinfo(
        "INFO",
        "Zoom or pan to view, press spacebar when ready to click."
    )
    while not zoom_ok:
        zoom_ok = plt.waitforbuttonpress()

    # Threshold selection
    messagebox.showinfo(
        "INFO",
        "Click twice to select the lower and upper threshold."
    )
    pts1 = plt.ginput(n=2, timeout=0, show_clicks=True)

    # Reconstruction
    pts2 = None
    if add_points:
        if number_of_added_points is None:
            raise RuntimeError("If 'add_points' is True, you must provide 'number_of_added_points'.")
        messagebox.showinfo(
            "INFO",
            f"Click {number_of_added_points} times to select additional profile points."
        )
        pts2 = plt.ginput(n=number_of_added_points, timeout=0, show_clicks=True)
        if pts2 is None:
            raise RuntimeError("No reconstruction points selected.")
        pts2 = [(float(a), float(b)) for a, b in pts2]

    plt.close()

    lower_cutoff_index = int(np.where(data[0] >= pts1[0][0])[0][0])
    upper_cutoff_index = int(np.where(data[0] >= pts1[1][0])[0][0])

    return cast(List[Tuple[float, float]], pts2), lower_cutoff_index, upper_cutoff_index


def check_wall_model(wall_model: np.ndarray, data: np.ndarray):
    """Plot an experimental hill-normal profile against the law of the wall.

    :param wall_model: Law of the wall data.
    :param data: Experimental profile data.
    """
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    axs.semilogx(data[0], data[1], linestyle="none", color="blue", marker="o")
    axs.semilogx(wall_model[0], wall_model[1], color="red")
    plt.show()
