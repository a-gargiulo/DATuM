"""Plot 45deg BeVERLI Hill profiles."""
import os
import pickle
import sys
from typing import Dict, List, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

InputDict = Dict[str, Union[str, List[int], float]]
# ExperimentalProfile = Dict[str, Dict[str]]


def initialize_plotting_parameters() -> None:
    """Initialize plotting parameters."""
    plt.rcParams["font.family"] = "Avenir"
    plt.rcParams["font.size"] = "18"
    plt.rcParams["lines.linewidth"] = "1.5"
    plt.rcParams["axes.linewidth"] = "2"


def obtain_file_path(current_station: int, input_data: InputDict) -> str:
    """Obtain the system path to the file containing the profile data for a specific
    station.

    :param current_station: An integer representing the current profile station.
    :param input_data: A dictionary containing all input parameters.
    :return: A string representing the system path.
    """
    file_name = (
        f"plane{current_station}_"
        f"{int(input_data['reynolds_number'] * 1e-3)}k_"
        f"{input_data['piv_sampling']}_"
        f"{input_data['coordinate_system']}_profiles.pkl"
    )

    return os.path.join(
        input_data["data_root_folder"],
        f"plane{current_station}",
        "preprocessed",
        file_name,
    )


def load_profile_data(file_path: str) -> Dict:
    """Load experimental profile data.

    :param file_path: A string representing the system path to the profile data.
    :return: Experimental profile data.
    """
    try:
        with open(file_path, "rb") as file:
            profiles = pickle.load(file)
            num_profiles = len(list(profiles.keys()))
            return profiles[f"profile_{num_profiles}"]["exp"]
    except FileNotFoundError:
        print("No profile data found!")
        sys.exit(1)


def extract_profile_data_to_plot(
    profile: Dict, input_data: InputDict
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the profile data of specific flow quantity to plot

    :param profile:
    :param input_data:
    :return:
    """
    if input_data["quantity"] == "reynolds_stress":
        factor = 2
    else:
        factor = 1

    if input_data["coordinate_system"] == "shear":
        x_data = (
            profile["coordinates"]["Y_SS"] - profile["properties"]["Y_SS_CORRECTION"]
        ) / profile["properties"]["integral_parameters"]["vinuesa"]["DELTA"]
        y_data = (
            profile[input_data["quantity"]][input_data["component"]]
            / profile["properties"]["integral_parameters"]["vinuesa"]["U_E"]**factor
        )
    else:
        x_data = profile["coordinates"]["Y"] - profile["coordinates"]["Y"][0]
        y_data = (
            profile[input_data["quantity"]][input_data["component"]]
            / profile["properties"]["U_INF"]**factor
        )

    return x_data, y_data


def plot_profiles(input_data: InputDict) -> None:
    """Bala bala

    :param input_data: Input data
    """
    initialize_plotting_parameters()

    fig = plt.figure(figsize=(3, 3))
    axs = fig.add_axes([0, 0, 1, 1])

    for station in input_data["stations"]:
        file_path = obtain_file_path(station, input_data)

        profile = load_profile_data(file_path)

        x_data, y_data = extract_profile_data_to_plot(profile, input_data)

        axs.semilogx(x_data, y_data)

        if input_data["quantity"] == "reynolds_stress":
            factor = 2
        else:
            factor = 1

        if input_data["coordinate_system"] == "shear":
            normalization = profile["properties"]["integral_parameters"]["vinuesa"]["U_E"]**factor
        else:
            normalization = profile["properties"]["U_INF"]**factor

        axs.fill_between(
            x_data,
            y_data
            - profile["uncertainty"][f"d{input_data['component']}"]
            / normalization,
            y_data
            + profile["uncertainty"][f"d{input_data['component']}"]
            / normalization,
            alpha=0.25,
        )

    axs.xaxis.set_tick_params(
        which="major", size=5, width=1.5, direction="in", bottom=True, top=True
    )
    axs.xaxis.set_tick_params(
        which="minor", size=3, width=1, direction="in", bottom=True, top=True
    )
    axs.yaxis.set_tick_params(
        which="major", size=5, width=1.5, direction="in", left=True, right=True
    )
    axs.yaxis.set_tick_params(
        which="minor", size=3, width=1, direction="in", left=True, right=True
    )
    axs.set_xlabel(input_data["xlabel"], labelpad=5)
    axs.set_ylabel(input_data["ylabel"], labelpad=10)
    axs.tick_params(axis="x", pad=10)
    axs.tick_params(axis="y", pad=10)

    axs.set_xlim(input_data["xrange"])
    axs.set_ylim(input_data["yrange"])

    axs.grid(True, "both", alpha=0.5, linestyle="dotted")
    axs.yaxis.set_major_locator(mpl.ticker.MultipleLocator(input_data["y_major_locator"]))

    fig.savefig(f"figures/profiles/{input_data['output_file']}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    inputs = {
        "data_root_folder": r"/Users/galdo/Desktop/direct-analysis-turbulence-models/data/piv/experiment",
        "stations": [3, 4, 6, 7],
        "reynolds_number": 250e3,
        "piv_sampling": "FS",
        "coordinate_system": "tunnel",
        "quantity": "mean_velocity",
        "component": "U",
        "xlabel": r"$x_2-x_w$ (m)",
        "ylabel": r"$\overline{U}_1/U_\infty$",
        "xrange": [1e-3, 1e-1],
        "yrange": [0.25, 1.25],
        "y_major_locator": 0.1,
        "output_file": "W.png"
    }

    plot_profiles(inputs)
