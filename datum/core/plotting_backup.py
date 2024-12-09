"""This module provides functions to carry out generic and BeVERLI Hill geometry
specific visualization tasks, respectively."""
import os
import sys
from typing import Dict, List, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import trimesh
from matplotlib import patches
from matplotlib.widgets import Cursor
from . import utility
from .beverli import Beverli
from .my_types import NestedDict, PoseMeasurement
from .parser import InputFile
import platform


def get_figure_folder(depth: str) -> str:
    """Obtain the path to the folder, where a figure for a specific PIV plane should be
    stored. The main `Figure` folder is generated in the working directory.

    :param depth: A string representing, whether to store the figure at the PIV `plane`
        or `case` level.
    :return: A string containing the path of the figure folder relative to the working
        directory.
    """
    input_data = InputFile().data

    plane_number = input_data["piv_data"]["plane_number"]
    plane_type = input_data["piv_data"]["plane_type"]
    reynolds_number = input_data["general"]["reynolds_number"]

    try:
        if depth.lower() == "plane":
            fig_folder = f"./figures/plane{plane_number}/"
        elif depth.lower() == "case":
            fig_folder = (
                f"./figures/"
                f"plane{plane_number}/"
                f"{int(reynolds_number * 1e-3)}k_{plane_type.upper()}/"
            )
        else:
            raise ValueError("Invalid parameter. Select between `plane` or `case`.")
    except ValueError as err:
        print(f"ERROR: {err}")
        sys.exit(1)

    os.makedirs(fig_folder, exist_ok=True)

    return fig_folder


def plot_global_pose(
    x_1_prof: np.ndarray, x_2_prof: np.ndarray, secant: List[float]
) -> None:
    """Visualize the result of the computation of the global pose used for coordinate
    transformation.

    :param x_1_prof: The x:sub:`1` coordinate of the hill profile.
    :param x_2_prof: The x:sub:`2` coordinate of the hill profile.
    :param secant: The characteristic secant parameters.
    """
    if platform.system() == "Windows":
        plt.rcParams["font.family"] = "Franklin Gothic Book"
    else:
        plt.rcParams["font.family"] = "Avenir"
    plt.rcParams["font.size"] = "18"
    plt.rcParams["lines.linewidth"] = "0.5"
    plt.rcParams["axes.linewidth"] = "2"

    # Plotting
    fig = plt.figure(figsize=(5, 3))
    axs = fig.add_axes([0, 0, 1, 1])
    axs.plot(x_1_prof, x_2_prof, color="blue")
    axs.plot(
        [secant[2], secant[0]],
        [secant[3], secant[1]],
        color="red",
        marker="o",
        markersize=1,
        linestyle="dashed",
    )
    axs.scatter(secant[4], secant[5], s=2, marker="o", c="black", zorder=500)
    rect = patches.Rectangle(
        (secant[0], secant[1]),
        np.sqrt((secant[0] - secant[2]) ** 2 + (secant[1] - secant[3]) ** 2),
        np.sqrt((secant[0] - secant[2]) ** 2 + (secant[1] - secant[3]) ** 2),
        angle=secant[6],
        color="red",
        alpha=0.5,
    )
    axs.add_patch(rect)

    # Appearance
    axs.set_xlabel(r"$x_1$ [m]", labelpad=10)
    axs.set_ylabel(r"$x_2$ [m]", labelpad=10)
    axs.xaxis.set_tick_params(which="major", size=8, width=2, direction="in")
    axs.xaxis.set_tick_params(which="minor", size=5, width=1.5, direction="in")
    axs.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.25))
    axs.yaxis.set_tick_params(which="major", size=8, width=2, direction="in")
    axs.yaxis.set_tick_params(which="minor", size=5, width=1.5, direction="in")
    if secant[4] < 0:
        axs.axis([-0.55, 0, -0.025, 0.186944 + 0.15])
        axs.set_aspect("equal", adjustable="box")
    else:
        axs.axis([0, 0.55, -0.025, 0.186944 + 0.15])
        axs.set_aspect("equal", adjustable="box")

    # Save figure
    fig.savefig(
        os.path.join(get_figure_folder("plane"), "piv_orientation.png"),
        format="png",
        dpi=300,
        backend="agg",
        transparent=False,
        bbox_inches="tight",
    )
    print(f"--> Figure `piv_orientation.png` was created.\n")


def local_reference_selector(
    x_1: np.ndarray, x_2: np.ndarray, vals: np.ndarray, pose: PoseMeasurement
) -> List[Tuple[float, float]]:
    """Visualize the calibration plate image with a manual picker to select the local
    pose used for coordinate transformation.

    :param x_1: The x:sub:`1` coordinates of the calibration plate image.
    :param x_2: The x:sub:`2` coordinates of the calibration plate image.
    :param vals: The intensity values of the calibration plate image.
    :param pose: A nested dictionary containing pose measurement parameters.
    :return: A tuple containing the coordinates of the manually picked points.
    """
    input_data = InputFile().data

    print("Opening reference image...")
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.close()

    if utility.search_nested_dict(pose, "reference_image"):
        plt.figure()
        ref_image_path = utility.find_file(
            input_data["system"]["piv_plane_data_folder"],
            pose["notes"]["reference_image"],
        )
        ref_image = plt.imread(ref_image_path)
        plt.imshow(ref_image)
        print(
            "\nPlease refer to the global parameter locations in the reference image. "
            "Close the image when ready and select the corresponding points on the "
            "perspective-corrected calibration image.\n"
        )
        plt.show()

    print("Opening calibration plate image...")
      
    cmap = plt.get_cmap("gray")
    _, axs = plt.subplots(1, 1, figsize=(8, 6))
    axs.pcolor(x_1, x_2, vals, cmap=cmap, vmin=0, vmax=4000)
    _ = Cursor(axs, useblit=True, color="gray", linewidth=1)
    zoom_ok = False
    print("\nZoom or pan to view, \npress spacebar when ready to click:\n")
    while not zoom_ok:
        zoom_ok = plt.waitforbuttonpress()
    print("Click once to select location matching global reference point...\n\n")
    pts = plt.ginput(n=1, timeout=0, show_clicks=True)

    plt.close()

    return pts

def profile_reconstructor(wall_model: List[np.ndarray], data: List[np.ndarray], add_points: bool, number_of_added_points: int = None):
    fig, axs = plt.subplots(1,1, figsize=(8, 6))
    axs.semilogx(wall_model[0], wall_model[1], color="red")
    axs.semilogx(data[0], data[1], linestyle="none", color="blue", marker="o")
    _ = Cursor(axs, useblit=True, color="gray", linewidth=1)
    zoom_ok = False
    print("\nZoom or pan to view, \npress spacebar when ready to click:\n")
    while not zoom_ok:
        zoom_ok = plt.waitforbuttonpress()
    print("Click twice to select the lower and upper threshold.\n\n")
    pts1 = plt.ginput(n=2, timeout=0, show_clicks=True)
    pts2 = None
    if add_points:
        print(f"Click {number_of_added_points} times to select additional profile points.")
        pts2 = plt.ginput(n=number_of_added_points, timeout=0, show_clicks=True)

    plt.close()

    lower_cutoff_index = np.where(data[0] >= pts1[0][0])[0][0]
    upper_cutoff_index = np.where(data[0] >= pts1[1][0])[0][0]

    return pts2, lower_cutoff_index, upper_cutoff_index


def check_wall_model(wall_model, data):
    fig, axs = plt.subplots(1,1, figsize=(8, 6))
    axs.semilogx(data[0], data[1], linestyle="none", color="blue", marker="o")
    axs.semilogx(wall_model[0], wall_model[1], color="red")
    plt.show()

def point_selector(number_of_points, coordinates, quantity, properties):
    cmap = plt.get_cmap(properties["colormap"])
    bounds = np.linspace(
        properties["contour_range"]["start"],
        properties["contour_range"]["end"],
        properties["contour_range"]["num_of_contours"],
    )
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    axs.pcolor(coordinates["X"], coordinates["Y"], quantity, norm=norm, cmap=cmap)
    x1b, x2b = Beverli().compute_x1_x2_profile(properties["zpos"])
    axs.plot(x1b, x2b, linestyle="--", color="k")
    axs.tick_params(axis="x", pad=10)
    axs.tick_params(axis="y", pad=10)

    axs.set_xlabel(r"$x_1$ (m)", labelpad=5)
    axs.set_ylabel(r"$x_2$ (m)", labelpad=10)

    axs.xaxis.set_tick_params(which="major", size=8, width=2, direction="in")
    axs.xaxis.set_tick_params(which="minor", size=5, width=1.5, direction="in")
    axs.yaxis.set_tick_params(which="major", size=8, width=2, direction="in")
    axs.yaxis.set_tick_params(which="minor", size=5, width=1.5, direction="in")

    axs.set_xlim(properties["xlim"][0], properties["xlim"][1])
    axs.set_ylim(properties["ylim"][0], properties["ylim"][1])

    axs.xaxis.set_major_locator(
        mpl.ticker.MultipleLocator(properties["xmajor_locator"])
    )
    if properties["ymajor_locator"]:
        axs.yaxis.set_major_locator(
            mpl.ticker.MultipleLocator(properties["ymajor_locator"])
        )

    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation="vertical",
        cax=axs.inset_axes([1.15, 0, 0.075, 1])
    )
    cbar.set_ticks(
        np.linspace(
            properties["cbar_range"]["start"],
            properties["cbar_range"]["end"],
            properties["cbar_range"]["num_of_ticks"],
        )
    )
    cbar.set_label(properties["cbar_label"], labelpad=10)
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
    return pts

def plot_contour(
    coordinates: Dict[str, np.ndarray],
    quantity: np.ndarray,
    properties: NestedDict,
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

    cmap = plt.get_cmap(properties["colormap"])
    bounds = np.linspace(
        properties["contour_range"]["start"],
        properties["contour_range"]["end"],
        properties["contour_range"]["num_of_contours"],
    )
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(3, 3))
    axs = fig.add_axes([0, 0, 1, 1])
    axs.pcolor(coordinates["X"], coordinates["Y"], quantity, norm=norm, cmap=cmap)
    x1b, x2b = Beverli().compute_x1_x2_profile(properties["zpos"])
    axs.plot(x1b, x2b, linestyle="--", color="k")
    axs.tick_params(axis="x", pad=10)
    axs.tick_params(axis="y", pad=10)

    axs.set_xlabel(r"$x_1$ (m)", labelpad=5)
    axs.set_ylabel(r"$x_2$ (m)", labelpad=10)

    axs.xaxis.set_tick_params(which="major", size=8, width=2, direction="in")
    axs.xaxis.set_tick_params(which="minor", size=5, width=1.5, direction="in")
    axs.yaxis.set_tick_params(which="major", size=8, width=2, direction="in")
    axs.yaxis.set_tick_params(which="minor", size=5, width=1.5, direction="in")

    axs.set_xlim(properties["xlim"][0], properties["xlim"][1])
    axs.set_ylim(properties["ylim"][0], properties["ylim"][1])

    axs.xaxis.set_major_locator(
        mpl.ticker.MultipleLocator(properties["xmajor_locator"])
    )
    if properties["ymajor_locator"]:
        axs.yaxis.set_major_locator(
            mpl.ticker.MultipleLocator(properties["ymajor_locator"])
        )

    # divider = make_axes_locatable(axs)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation="vertical",
        cax=axs.inset_axes([1.15, 0, 0.075, 1])
    )
    cbar.set_ticks(
        np.linspace(
            properties["cbar_range"]["start"],
            properties["cbar_range"]["end"],
            properties["cbar_range"]["num_of_ticks"],
        )
    )
    cbar.set_label(properties["cbar_label"], labelpad=10)
    cbar.ax.tick_params(width=2)
    cbar.ax.minorticks_off()

    fig.savefig(
        os.path.join(get_figure_folder("case"), outname + ".png"),
        format="png",
        dpi=300,
        backend="agg",
        transparent=False,
        bbox_inches="tight",
    )


def plot_3d_hill(mesh: Union[trimesh.Trimesh, Dict[str, np.ndarray]]) -> go.Figure:
    """Visualize the three-dimensional BeVERLI Hill geometry.

    :param mesh: The BeVERLI Hill geometry.
    :return: A Plotly figure object.
    """
    input_data = InputFile().data
    mesh_properties = {
        "cmin": 0,
        "cmax": 0.186944,
        "colorbar": {
            "borderwidth": 2,
            "dtick": 0.05,
            "len": 0.75,
            "outlinewidth": 2,
            "tickfont": {"size": 22},
            "ticklabelposition": "outside right",
            "ticklen": 10,
            "ticks": "inside",
            "tickwidth": 2,
            "title": {
                "font": {"size": 28},
                "text": "Height [m]",
            },
            "x": 0.85,
            "xanchor": "center",
            "xref": "container",
            "xpad": 30,
            "ypad": 30,
        },
        "colorscale": "Plasma",
        "name": "Hill Surface",
        "opacity": 1,
    }

    plane_properties = {
        "colorscale": [[0, "#000000"], [1, "#000000"]],
        "name": "Surface",
        "opacity": 0.1,
        "showscale": False,
    }

    figure_properties = {
        "margin": {"b": 0, "l": 0, "r": 0, "t": 0},
        "scene": {
            "aspectmode": "manual",
            "aspectratio": {"x": 1, "y": 1, "z": 0.25},
            "xaxis": {
                "dtick": 0.5,
                "range": [-1, 1],
                "tickwidth": 4,
                "ticks": "outside",
            },
            "xaxis_title": "Z [m]",
            "yaxis": {
                "dtick": 0.5,
                "range": [-1, 1],
                "tickwidth": 4,
                "ticks": "outside",
            },
            "yaxis_title": "X [m]",
            "zaxis": {
                "range": [-0.25, 0.25],
                "tickmode": "array",
                "tickwidth": 4,
                "ticks": "outside",
                "tickvals": np.array([0, 0.1, 0.2]),
            },
            "zaxis_title": "Y [m]",
        },
        "title": {
            "font": {"family": "Arial", "size": 48},
            "text": "The BeVERLI Hill",
            "x": 0.5,
            "xanchor": "center",
            "xref": "paper",
            "y": 0.9,
            "yanchor": "middle",
            "yref": "paper",
        },
    }

    properties = {
        "mesh": mesh_properties,
        "plane": plane_properties,
        "figure": figure_properties,
    }

    if input_data["hill_geometry"]["type"] == "CAD":
        fig = _plot_3d_cad_hill(mesh, properties)
    else:
        fig = _plot_3d_analytic_hill(mesh, properties)

    return fig


def plot_x1_x2_profile_3d(x_3_m: float, fig: go.Figure) -> go.Figure:
    """Visualize the cross-sectional x:sub:`1`-x:sub:`2` hill profile at a desired
    spanwise location over the three-dimensional BeVERLI Hill.

    :param x_3_m: The desired spanwise location.
    :param fig: The Plotly figure containing the three-dimensional BeVERLI Hill
        geometry.
    :return: A Plotly figure object.
    """
    x_1, x_2 = Beverli().compute_x1_x2_profile(x_3_m)
    profile_trace = go.Scatter3d(
        x=x_3_m * np.ones((len(x_1),)),
        y=x_1,
        z=x_2,
        line={"color": "black", "width": 4},
        mode="lines",
        name="Profile",
        showlegend=False,
    )
    fig.add_trace(profile_trace)

    return fig


def _plot_3d_cad_hill(
    mesh: Union[trimesh.Trimesh, Dict[str, np.ndarray]], properties: NestedDict
) -> go.Figure:
    """Plot the three-dimensional BeVERLI Hill CAD geometry using Plotly.

    :param mesh: The BeVERLI Hill geometry.
    :param properties: A dictionary containing the figure's properties.
    :return: A Plotly figure object.
    """
    input_data = InputFile().data
    # Create mesh trace
    mesh_trace = go.Mesh3d(
        x=mesh.vertices[:, 2],
        y=mesh.vertices[:, 0],
        z=mesh.vertices[:, 1],
        i=mesh.faces[:, 2],
        j=mesh.faces[:, 0],
        k=mesh.faces[:, 1],
        intensity=mesh.vertices[:, 1],
        **properties["mesh"],
    )

    # Perimeter trace
    angle_deg = input_data["general"]["hill_orientation"]
    x_1_perimeter, x_3_perimeter = Beverli().compute_perimeter(angle_deg)
    perimeter_trace = go.Scatter3d(
        x=x_3_perimeter,
        y=x_1_perimeter,
        z=np.zeros((len(x_1_perimeter),)),
        line={"color": "white", "width": 4},
        mode="lines",
        name="Perimeter",
        showlegend=False,
    )

    # Create zero plane trace
    x_range = np.linspace(-1, 1, 50)
    y_range = np.linspace(-1, 1, 50)
    x_plane, y_plane = np.meshgrid(x_range, y_range)
    z_plane = np.zeros((x_plane.size[0], x_plane.size[1]))
    plane_trace = go.Surface(x=x_plane, y=y_plane, z=z_plane, **properties["plane"])

    # Create and display figure
    fig = go.Figure(
        data=[mesh_trace, plane_trace, perimeter_trace],
        layout=go.Layout(**properties["figure"]),
    )

    return fig


def _plot_3d_analytic_hill(
    mesh: Union[trimesh.Trimesh, Dict[str, np.ndarray]], properties: NestedDict
) -> go.Figure:
    """Plot the three-dimensional BeVERLI Hill analytic geometry using Plotly.

    :param mesh: The BeVERLI Hill geometry.
    :param properties: A dictionary containing the figure's properties.
    :return: A Plotly figure object.
    """
    input_data = InputFile().data
    surface_trace = go.Surface(
        x=mesh["Z"], y=mesh["X"], z=mesh["Y"], **properties["mesh"]
    )

    # Create zero plane trace
    x_range = np.linspace(-1, 1, 50)
    y_range = np.linspace(-1, 1, 50)
    x_plane, y_plane = np.meshgrid(x_range, y_range)
    z_plane = np.zeros((x_plane.size[0], x_plane.size[1]))
    plane_trace = go.Surface(x=x_plane, y=y_plane, z=z_plane, **properties["plane"])

    # Perimeter trace
    angle_deg = input_data["general"]["hill_orientation"]
    x_1_perimeter, x_3_perimeter = Beverli().compute_perimeter(angle_deg)
    perimeter_trace = go.Scatter3d(
        x=x_3_perimeter,
        y=x_1_perimeter,
        z=np.zeros((len(x_1_perimeter),)),
        line={"color": "white", "width": 4},
        mode="lines",
        name="Perimeter",
        showlegend=False,
    )

    fig = go.Figure(
        data=[surface_trace, plane_trace, perimeter_trace], layout=properties["figure"]
    )

    return fig
