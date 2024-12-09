"""This module contains a library of functions for the computation of integral boundary
layer parameters from profile data."""
import pdb
import re
from typing import Callable, Dict, List, Union

import numpy as np
from scipy.interpolate import interp1d

from . import plotting, utility
from .my_types import ProfileDictSingle
from .parser import InputFile
from .spalding import spalding_profile


def get_centerline_pressure() -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
    """Obtain pressure data interpolants along the centerline of the port wall and the
    BeVERLI hill.

    :return: A nested dictionary containing the pressure interpolants and additional
        property data.
    """
    # pylint: disable=too-many-locals
    # Load pressure data
    input_data = InputFile().data
    port_wall_pressure_file = utility.find_file(
        input_data["system"]["pressure_data_root_folder"],
        input_data["pressure_data"]["port_wall"],
    )
    hill_pressure_file = utility.find_file(
        input_data["system"]["pressure_data_root_folder"],
        input_data["pressure_data"]["hill"],
    )
    readme_file = utility.find_file(
        input_data["system"]["pressure_data_root_folder"],
        input_data["pressure_data"]["readme"],
    )
    properties_file = utility.construct_file_path(
        input_data["system"]["piv_plane_data_folder"],
        [],
        input_data["general"]["fluid_and_flow_properties"],
    )
    pressure_data_port_wall = np.loadtxt(port_wall_pressure_file, skiprows=1)
    pressure_data_hill = np.loadtxt(hill_pressure_file, skiprows=1)

    properties = utility.load_json(properties_file)
    rho = properties["fluid"]["density"]

    # Obtain centerline pressure and properties
    centerline_pressure = {"interpolants": {}, "properties": {}, "coordinates": {}}
    with open(readme_file, "r", encoding="utf-8") as file:
        content = file.read()

        # Define regular expressions to match and extract the values
        patterns = {
            "P_INF": r"P_inf\s*=\s*(\d+)\s*Pa",
            "P_0": r"P_0,inf\s*=\s*(\d+)\s*Pa",
            "U_INF": r"U_inf\s*=\s*([\d.]+)\s*m/s\s*\+/\-\s*([\d.]+)\s*m/s",
        }

        for var, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                centerline_pressure["properties"][var] = float(match.group(1))

    # Join port wall and hill data along the centerline
    centerline_pressure_x_1 = pressure_data_port_wall[
        pressure_data_port_wall[:, 3] == 0, 1
    ]
    centerline_pressure_x_1 = np.append(
        centerline_pressure_x_1, pressure_data_hill[pressure_data_hill[:, 3] == 0, 1]
    )

    centerline_cp_inf = pressure_data_port_wall[pressure_data_port_wall[:, 3] == 0, 4]
    centerline_cp_inf = np.append(
        centerline_cp_inf, pressure_data_hill[pressure_data_hill[:, 3] == 0, 4]
    )

    # Sort data
    sorting_indices = np.argsort(centerline_pressure_x_1)
    centerline_pressure_x_1 = centerline_pressure_x_1[sorting_indices]
    centerline_cp_inf = centerline_cp_inf[sorting_indices]
    centerline_static_pressure = (
        centerline_cp_inf
        * (0.5 * rho * centerline_pressure["properties"]["U_INF"] ** 2)
        + centerline_pressure["properties"]["P_INF"]
    )

    # Get interpolants
    centerline_pressure["interpolants"]["P_THIRD_ORDER"] = interp1d(
        centerline_pressure_x_1, centerline_static_pressure, kind=3, fill_value=np.nan
    )
    centerline_pressure["interpolants"]["CP_THIRD_ORDER"] = interp1d(
        centerline_pressure_x_1, centerline_cp_inf, kind=3, fill_value=np.nan
    )
    centerline_pressure["interpolants"]["CP_FIRST_ORDER"] = interp1d(
        centerline_pressure_x_1, centerline_cp_inf, kind=1, fill_value=np.nan
    )

    return centerline_pressure


def calculate_boundary_layer_integral_parameters(profile: ProfileDictSingle):
    """Calculate the boundary layer thickness of a profile in local wall shear stress
    coordinates using one out of three methods of your choice.

    :param profile: A nested dictionary containing the profile data.
    :return: A dictionary containing the boundary layer thickness computed using three
        different methods.
    """
    pressure_data = get_centerline_pressure()
    exp_velocity_interpolant = interp1d(
        profile["coordinates"]["Y_SS"][~np.isnan(profile["mean_velocity"]["U_SS"])] - profile["properties"]["Y_SS_CORRECTION"],
        profile["mean_velocity"]["U_SS"][~np.isnan(profile["mean_velocity"]["U_SS"])],
        kind="linear",
        fill_value="extrapolate",
    )
    # Reconstruct velocity profile
    reconstruct_profile(profile)

    # Show model profile
    plotting.check_wall_model(
        wall_model=[
            profile["coordinates"]["Y_SS_MODELED"]
            * profile["properties"]["U_TAU"]
            / profile["properties"]["NU"],
            profile["mean_velocity"]["U_SS_MODELED"] / profile["properties"]["U_TAU"],
        ],
        data=[
            (profile["coordinates"]["Y_SS"] - profile["properties"]["Y_SS_CORRECTION"])
            * profile["properties"]["U_TAU"]
            / profile["properties"]["NU"],
            profile["mean_velocity"]["U_SS"] / profile["properties"]["U_TAU"],
        ],
    )

    integral_parameters = {"griffin": {}, "vinuesa": {}}
    # Local Reconstruction Method, Griffin et al. (2021)
    delta_threshold = 0.99
    u_griffin = (2 / profile["properties"]["RHO"]) * (
        pressure_data["properties"]["P_0"]
        - pressure_data["interpolants"]["P_THIRD_ORDER"](profile["coordinates"]["X"][0])
    ) - profile["mean_velocity"]["V_SS"] ** 2

    velocity_ratio_interpolant = interp1d(
        profile["mean_velocity"]["U_SS"][~np.isnan(profile["mean_velocity"]["U_SS"])] ** 2 / u_griffin[~np.isnan(profile["mean_velocity"]["U_SS"])],
        profile["coordinates"]["Y_SS"][~np.isnan(profile["mean_velocity"]["U_SS"])] - profile["properties"]["Y_SS_CORRECTION"],
        kind="linear",
        fill_value="extrapolate",
    )
    integral_parameters["griffin"]["DELTA"] = velocity_ratio_interpolant(
        delta_threshold**2
    )
    print(integral_parameters["griffin"]["DELTA"])
    integral_parameters["griffin"]["U_E"] = exp_velocity_interpolant(
        integral_parameters["griffin"]["DELTA"]
    )

    # Diagnostic Plot Method from Vinuesa et al. (2016)
    # Estimate delta
    delta_new = 0.05
    delta_old = 0
    u_e = None
    while np.abs(delta_new - delta_old) > 1e-7:
        delta_old = delta_new
        u_e = exp_velocity_interpolant(delta_old)
        delta_star = np.trapz(
            (
                1
                - profile["mean_velocity"]["U_SS_MODELED"][
                    profile["coordinates"]["Y_SS_MODELED"] <= delta_old
                ]
                / u_e
            ),
            x=profile["coordinates"]["Y_SS_MODELED"][
                profile["coordinates"]["Y_SS_MODELED"] <= delta_old
            ],
        )
        theta = np.trapz(
            (
                profile["mean_velocity"]["U_SS_MODELED"][
                    profile["coordinates"]["Y_SS_MODELED"] <= delta_old
                ]
                / u_e
            )
            * (
                1
                - profile["mean_velocity"]["U_SS_MODELED"][
                    profile["coordinates"]["Y_SS_MODELED"] <= delta_old
                ]
                / u_e
            ),
            x=profile["coordinates"]["Y_SS_MODELED"][
                profile["coordinates"]["Y_SS_MODELED"] <= delta_old
            ],
        )
        shape_parameter = delta_star / theta
        f_int = interp1d(
            np.sqrt(profile["reynolds_stress"]["UU_SS"][~np.isnan(profile["reynolds_stress"]["UU_SS"])])
            / (u_e * np.sqrt(shape_parameter)),
            profile["coordinates"]["Y_SS"][~np.isnan(profile["reynolds_stress"]["UU_SS"])] - profile["properties"]["Y_SS_CORRECTION"],
            kind="linear",
            fill_value="extrapolate",
        )
        delta_new = f_int(0.02)
    integral_parameters["vinuesa"]["DELTA"] = delta_new
    print(integral_parameters["vinuesa"]["DELTA"])
    integral_parameters["vinuesa"]["U_E"] = u_e

    # Parameters
    for method in ["vinuesa", "griffin"]:
        integral_parameters[method]["DELTA_STAR"] = np.trapz(
            (
                1
                - profile["mean_velocity"]["U_SS_MODELED"][
                    profile["coordinates"]["Y_SS_MODELED"]
                    <= integral_parameters[method]["DELTA"]
                ]
                / integral_parameters[method]["U_E"]
            ),
            x=profile["coordinates"]["Y_SS_MODELED"][
                profile["coordinates"]["Y_SS_MODELED"]
                <= integral_parameters[method]["DELTA"]
            ],
        )
        integral_parameters[method]["THETA"] = np.trapz(
            (
                profile["mean_velocity"]["U_SS_MODELED"][
                    profile["coordinates"]["Y_SS_MODELED"]
                    <= integral_parameters[method]["DELTA"]
                ]
                / integral_parameters[method]["U_E"]
            )
            * (
                1
                - profile["mean_velocity"]["U_SS_MODELED"][
                    profile["coordinates"]["Y_SS_MODELED"]
                    <= integral_parameters[method]["DELTA"]
                ]
                / integral_parameters[method]["U_E"]
            ),
            x=profile["coordinates"]["Y_SS_MODELED"][
                profile["coordinates"]["Y_SS_MODELED"]
                <= integral_parameters[method]["DELTA"]
            ],
        )

    profile["properties"]["integral_parameters"] = integral_parameters


def reconstruct_profile(profile: ProfileDictSingle) -> None:
    input_data = InputFile().data
    exp_velocity_interpolant = interp1d(
        profile["coordinates"]["Y_SS"][~np.isnan(profile["mean_velocity"]["U_SS"])] - profile["properties"]["Y_SS_CORRECTION"],
        profile["mean_velocity"]["U_SS"][~np.isnan(profile["mean_velocity"]["U_SS"])],
        kind="linear",
        fill_value="extrapolate",
    )

    u_1_plus_spalding = np.linspace(0, 30, 10000)
    x_2_plus_spalding = spalding_profile(u_1_plus_spalding)

    x_2_ss_wall_model = (
        x_2_plus_spalding * profile["properties"]["NU"] / profile["properties"]["U_TAU"]
    )
    u_1_ss_wall_model = u_1_plus_spalding * profile["properties"]["U_TAU"]

    (
        additional_pts,
        cutoff_index_lower,
        cutoff_index_upper,
    ) = plotting.profile_reconstructor(
        wall_model=[x_2_plus_spalding, u_1_plus_spalding],
        data=[
            (profile["coordinates"]["Y_SS"] - profile["properties"]["Y_SS_CORRECTION"])
            * profile["properties"]["U_TAU"]
            / profile["properties"]["NU"],
            profile["mean_velocity"]["U_SS"] / profile["properties"]["U_TAU"],
        ],
        add_points=input_data["profiles"]["add_reconstruction_points"],
        number_of_added_points=input_data["profiles"][
            "number_of_reconstruction_points"
        ],
    )

    # Append additional points
    if additional_pts:
        additional_x_2_ss = (
            np.array([i for i, j in additional_pts])
            * profile["properties"]["NU"]
            / profile["properties"]["U_TAU"]
        )

        sort_indices = np.argsort(additional_x_2_ss)
        additional_x_2_ss = additional_x_2_ss[sort_indices]

        additional_u_1_ss = (
                np.array([j for i, j in additional_pts]) * profile["properties"]["U_TAU"]
        )

        additional_u_1_ss = additional_u_1_ss[sort_indices]

        # Spalding near-wall section
        lower_cutoff_condition = (
                x_2_ss_wall_model
                < additional_x_2_ss[0]
        )
        u_1_ss_wall_model = u_1_ss_wall_model[lower_cutoff_condition]
        x_2_ss_wall_model = x_2_ss_wall_model[lower_cutoff_condition]

        x_2_ss_wall_model = np.append(x_2_ss_wall_model, additional_x_2_ss)
        u_1_ss_wall_model = np.append(u_1_ss_wall_model, additional_u_1_ss)
    else:
        # Spalding near-wall section
        lower_cutoff_condition = (
            x_2_ss_wall_model
            < profile["coordinates"]["Y_SS"][cutoff_index_lower]
            - profile["properties"]["Y_SS_CORRECTION"]
        )
        u_1_ss_wall_model = u_1_ss_wall_model[lower_cutoff_condition]
        x_2_ss_wall_model = x_2_ss_wall_model[lower_cutoff_condition]

    # Append experimental data for outer section
    x_2_ss_wall_model = np.append(
        x_2_ss_wall_model,
        np.linspace(
            profile["coordinates"]["Y_SS"][cutoff_index_lower]
            - profile["properties"]["Y_SS_CORRECTION"],
            profile["coordinates"]["Y_SS"][cutoff_index_upper]
            - profile["properties"]["Y_SS_CORRECTION"],
            1000,
        ),
    )
    u_1_ss_wall_model = np.append(
        u_1_ss_wall_model,
        exp_velocity_interpolant(
            np.linspace(
                profile["coordinates"]["Y_SS"][cutoff_index_lower]
                - profile["properties"]["Y_SS_CORRECTION"],
                profile["coordinates"]["Y_SS"][cutoff_index_upper]
                - profile["properties"]["Y_SS_CORRECTION"],
                1000,
            )
        ),
    )

    profile["mean_velocity"]["U_SS_MODELED"] = u_1_ss_wall_model
    profile["coordinates"]["Y_SS_MODELED"] = x_2_ss_wall_model
