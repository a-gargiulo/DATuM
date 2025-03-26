"""This module defines functions for the computation of integral boundary layer parameters from profile data."""
import re
from typing import cast, Dict, Union

import numpy as np
from scipy.interpolate import interp1d

from . import plotting
from .my_types import SingleProfile, Properties, UserInputs, Interp1DCallable
from .spalding import spalding_profile


def _get_centerline_pressure(
    inputs: UserInputs,
    properties: Properties
) -> Dict[str, Dict[str, Union[float, Interp1DCallable]]]:
    # Load pressure data
    port_wall_pressure_file = str(inputs["port_wall_pressure_file"])
    hill_pressure_file = str(inputs["hill_pressure_file"])
    readme_file = str(inputs["readme_pressure_file"])

    pressure_data_port_wall = np.loadtxt(port_wall_pressure_file, skiprows=1)
    pressure_data_hill = np.loadtxt(hill_pressure_file, skiprows=1)

    rho = float(properties["fluid"]["density"])

    # Obtain centerline pressure and properties
    centerline_pressure = {"interpolants": {}, "properties": {}}
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
    centerline_pressure_x_1 = pressure_data_port_wall[pressure_data_port_wall[:, 3] == 0, 1]
    centerline_pressure_x_1 = np.append(centerline_pressure_x_1, pressure_data_hill[pressure_data_hill[:, 3] == 0, 1])

    centerline_cp_inf = pressure_data_port_wall[pressure_data_port_wall[:, 3] == 0, 4]
    centerline_cp_inf = np.append(centerline_cp_inf, pressure_data_hill[pressure_data_hill[:, 3] == 0, 4])

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
        centerline_pressure_x_1, centerline_static_pressure, kind='cubic', fill_value=np.nan
    )
    centerline_pressure["interpolants"]["CP_THIRD_ORDER"] = interp1d(
        centerline_pressure_x_1, centerline_cp_inf, kind='cubic', fill_value=np.nan
    )
    centerline_pressure["interpolants"]["CP_FIRST_ORDER"] = interp1d(
        centerline_pressure_x_1, centerline_cp_inf, kind='linear', fill_value=np.nan
    )

    return centerline_pressure


def _reconstruct_profile(profile: SingleProfile, inputs: UserInputs) -> bool:
    exp_velocity_interpolant = interp1d(
        (
            cast(dict, profile["coordinates"])["Y_SS"][~np.isnan(cast(dict, profile["mean_velocity"])["U_SS"])] -
            cast(dict, profile["properties"])["Y_SS_CORRECTION"]
        ),
        cast(dict, profile["mean_velocity"])["U_SS"][~np.isnan(cast(dict, profile["mean_velocity"])["U_SS"])],
        kind="linear",
        fill_value="extrapolate",
    )

    u_1_plus_spalding = np.linspace(0, 30, 10000)
    x_2_plus_spalding = cast(np.ndarray, spalding_profile(u_1_plus_spalding))

    x_2_ss_wall_model = (
        x_2_plus_spalding * cast(float, profile["properties"]["NU"]) / cast(float, profile["properties"]["U_TAU"])
    )
    u_1_ss_wall_model = u_1_plus_spalding * cast(float, profile["properties"]["U_TAU"])

    result = plotting.profile_reconstructor(
        wall_model=[x_2_plus_spalding, u_1_plus_spalding],
        data=[
            cast(
                np.ndarray,
                (profile["coordinates"]["Y_SS"] - profile["properties"]["Y_SS_CORRECTION"])
                * profile["properties"]["U_TAU"]
                / profile["properties"]["NU"]
            ),
            cast(
                np.ndarray,
                profile["mean_velocity"]["U_SS"] / profile["properties"]["U_TAU"]
            ),
        ],
        add_points=bool(inputs["add_reconstruction_points"]),
        number_of_added_points=(
            int(inputs["num_of_reconstruction_points"]) if inputs["num_of_reconstruction_points"] is not None else None
        ),
    )
    if result is None:
        return False
    additional_pts, cutoff_index_lower, cutoff_index_upper = result

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
            < cast(dict, profile["coordinates"])["Y_SS"][cutoff_index_lower]
            - profile["properties"]["Y_SS_CORRECTION"]
        )
        u_1_ss_wall_model = u_1_ss_wall_model[lower_cutoff_condition]
        x_2_ss_wall_model = x_2_ss_wall_model[lower_cutoff_condition]

    # Append experimental data for outer section
    x_2_ss_wall_model = np.append(
        x_2_ss_wall_model,
        np.linspace(
            cast(dict, profile["coordinates"])["Y_SS"][cutoff_index_lower]
            - profile["properties"]["Y_SS_CORRECTION"],
            cast(dict, profile["coordinates"])["Y_SS"][cutoff_index_upper]
            - profile["properties"]["Y_SS_CORRECTION"],
            1000,
        ),
    )
    u_1_ss_wall_model = np.append(
        u_1_ss_wall_model,
        exp_velocity_interpolant(
            np.linspace(
                cast(dict, profile["coordinates"])["Y_SS"][cutoff_index_lower]
                - profile["properties"]["Y_SS_CORRECTION"],
                cast(dict, profile["coordinates"])["Y_SS"][cutoff_index_upper]
                - profile["properties"]["Y_SS_CORRECTION"],
                1000,
            )
        ),
    )

    profile["mean_velocity"]["U_SS_MODELED"] = u_1_ss_wall_model
    profile["coordinates"]["Y_SS_MODELED"] = x_2_ss_wall_model

    return True


def calculate_boundary_layer_integral_parameters(profile: SingleProfile, inputs: UserInputs, properties: Properties) -> bool:
    """
    Calculate the boundary layer integral parameters for a single experimental hill-normal profile.

    The algorithm uses the profile data expressed in local wall shear stress coordinates (SS system).

    The integral boundary layer parameters are calculated using two methods: the Griffin method and the Vinuesa method.
    Once calculated, the parameters are automatically appended to the profile data.

    :param profile: The hill-normal profile data.
    :param inputs: The user's inputs.
    :param properties: The fluid, flow, and reference properties.
    """
    pressure_data = _get_centerline_pressure(inputs, properties)
    exp_velocity_interpolant = interp1d(
        (
            cast(dict, profile["coordinates"])["Y_SS"][~np.isnan(cast(dict, profile["mean_velocity"])["U_SS"])] -
            cast(dict, profile["properties"])["Y_SS_CORRECTION"]
        ),
        cast(dict, profile["mean_velocity"])["U_SS"][~np.isnan(cast(dict, profile["mean_velocity"])["U_SS"])],
        kind="linear",
        fill_value="extrapolate",
    )

    # Reconstruct velocity profile
    _reconstruct_profile(profile, inputs)

    # Show model profile
    plotting.check_wall_model(
        wall_model=cast(
            np.ndarray,
            [
                profile["coordinates"]["Y_SS_MODELED"]
                * profile["properties"]["U_TAU"]
                / profile["properties"]["NU"],
                profile["mean_velocity"]["U_SS_MODELED"] / profile["properties"]["U_TAU"],
            ]
        ),
        data=cast(
            np.ndarray,
            [
                (profile["coordinates"]["Y_SS"] - profile["properties"]["Y_SS_CORRECTION"])
                * profile["properties"]["U_TAU"]
                / profile["properties"]["NU"],
                profile["mean_velocity"]["U_SS"] / profile["properties"]["U_TAU"],
            ]
        ),
    )

    integral_parameters = {"griffin": {}, "vinuesa": {}}
    # Local Reconstruction Method, Griffin et al. (2021)
    delta_threshold = 0.99
    u_griffin = cast(
        np.ndarray,
        (2 / cast(float, profile["properties"]["RHO"])) *
        (
            cast(float, pressure_data["properties"]["P_0"]) -
            cast(Interp1DCallable, pressure_data["interpolants"]["P_THIRD_ORDER"])(
                cast(list, profile["coordinates"]["X"])[0]
            )
        ) -
        profile["mean_velocity"]["V_SS"] ** 2
    )

    velocity_ratio_interpolant = interp1d(
        cast(
            np.ndarray,
            profile["mean_velocity"]["U_SS"]
        )[~np.isnan(cast(np.ndarray, profile["mean_velocity"]["U_SS"]))] ** 2 /
        u_griffin[~np.isnan(cast(np.ndarray, profile["mean_velocity"]["U_SS"]))],
        cast(
            np.ndarray,
            profile["coordinates"]["Y_SS"]
        )[~np.isnan(profile["mean_velocity"]["U_SS"])] - profile["properties"]["Y_SS_CORRECTION"],
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
                - cast(dict, profile["mean_velocity"])["U_SS_MODELED"][
                    profile["coordinates"]["Y_SS_MODELED"] <= delta_old
                ]
                / u_e
            ),
            x=cast(dict, profile["coordinates"])["Y_SS_MODELED"][
                profile["coordinates"]["Y_SS_MODELED"] <= delta_old
            ],
        )
        theta = np.trapz(
            (
                cast(dict, profile["mean_velocity"])["U_SS_MODELED"][
                    profile["coordinates"]["Y_SS_MODELED"] <= delta_old
                ]
                / u_e
            )
            * (
                1
                - cast(dict, profile["mean_velocity"])["U_SS_MODELED"][
                    profile["coordinates"]["Y_SS_MODELED"] <= delta_old
                ]
                / u_e
            ),
            x=cast(dict, profile["coordinates"])["Y_SS_MODELED"][
                profile["coordinates"]["Y_SS_MODELED"] <= delta_old
            ],
        )
        shape_parameter = delta_star / theta
        f_int = interp1d(
            np.sqrt(cast(dict, profile["reynolds_stress"])["UU_SS"][~np.isnan(profile["reynolds_stress"]["UU_SS"])])
            / (u_e * np.sqrt(shape_parameter)),
            cast(dict, profile["coordinates"])["Y_SS"][
                ~np.isnan(profile["reynolds_stress"]["UU_SS"])
            ] - profile["properties"]["Y_SS_CORRECTION"],
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
                - cast(dict, profile["mean_velocity"])["U_SS_MODELED"][
                    profile["coordinates"]["Y_SS_MODELED"]
                    <= integral_parameters[method]["DELTA"]
                ]
                / integral_parameters[method]["U_E"]
            ),
            x=cast(dict, profile["coordinates"])["Y_SS_MODELED"][
                profile["coordinates"]["Y_SS_MODELED"]
                <= integral_parameters[method]["DELTA"]
            ],
        )
        integral_parameters[method]["THETA"] = np.trapz(
            (
                cast(dict, profile["mean_velocity"])["U_SS_MODELED"][
                    profile["coordinates"]["Y_SS_MODELED"]
                    <= integral_parameters[method]["DELTA"]
                ]
                / integral_parameters[method]["U_E"]
            )
            * (
                1
                - cast(dict, profile["mean_velocity"])["U_SS_MODELED"][
                    profile["coordinates"]["Y_SS_MODELED"]
                    <= integral_parameters[method]["DELTA"]
                ]
                / integral_parameters[method]["U_E"]
            ),
            x=cast(dict, profile["coordinates"])["Y_SS_MODELED"][
                profile["coordinates"]["Y_SS_MODELED"]
                <= integral_parameters[method]["DELTA"]
            ],
        )

    cast(dict, profile["properties"])["integral_parameters"] = integral_parameters
    return True
