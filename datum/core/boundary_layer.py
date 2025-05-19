"""This module defines functions for the computation of integral boundary layer parameters from profile data."""
import re
from typing import cast, Dict, Union

import numpy as np
from scipy.interpolate import interp1d

from . import plotting
from .my_types import ProfileData, Properties, PRInputs, CenterlinePressure, FloatOrArray, Interp1DCallable
from .spalding import spalding_profile


def _get_centerline_pressure(
    ui: PRInputs, props: Properties
) -> CenterlinePressure:
    # Load pressure data
    port_wall_pressure_file = cast(str, ui["port_wall_pressure"])
    hill_pressure_file = cast(str, ui["hill_pressure"])
    readme_file = cast(str, ui["pressure_readme"])

    pressure_data_port_wall = np.loadtxt(port_wall_pressure_file, skiprows=1)
    pressure_data_hill = np.loadtxt(hill_pressure_file, skiprows=1)

    rho = props["fluid"]["density"]

    # Obtain centerline pressure and properties
    def empty_interp(x: FloatOrArray) -> FloatOrArray:
        return x

    centerline_pressure: CenterlinePressure = {
        "interpolants": {
            "P_THIRD_ORDER": empty_interp,
            "CP_THIRD_ORDER": empty_interp,
            "CP_FIRST_ORDER": empty_interp,

        },
        "properties": {
            "P_INF": 0.0,
            "U_INF": 0.0,
            "P_0": 0.0,
        }
    }

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


def _reconstruct_profile(profile: ProfileData, ui: PRInputs, exp_vel: Interp1DCallable) -> None:
    add_reconstruction = cast(bool, ui["add_reconstruction_points"])
    num_recr_pts = cast(int, ui["number_of_reconstruction_points"]) if add_reconstruction else None
    u_tau = cast(float, profile["properties"]["U_TAU"])
    nu = profile["properties"]["NU"]
    y_ss = cast(np.ndarray, profile["coordinates"]["Y_SS"])
    y_0_ss = cast(np.ndarray, profile["properties"]["Y_SS_CORRECTION"])
    u_ss = cast(np.ndarray, profile["mean_velocity"]["U_SS"])

    u_1_plus_spalding = np.linspace(0, 30, 10000)
    x_2_plus_spalding = spalding_profile(u_1_plus_spalding)

    x_2_ss_wall_model = (x_2_plus_spalding * nu / u_tau)
    u_1_ss_wall_model = u_1_plus_spalding * u_tau

    result = plotting.profile_reconstructor(
        wall_model=[x_2_plus_spalding, u_1_plus_spalding],
        data=[(y_ss - y_0_ss) * u_tau / nu, u_ss / u_tau],
        add_points=add_reconstruction,
        number_of_added_points=num_recr_pts,
    )
    additional_pts, cutoff_index_lower, cutoff_index_upper = result

    # Append additional points
    if add_reconstruction:
        additional_x_2_ss = (
            np.array([i for i, j in additional_pts]) * nu / u_tau
        )

        sort_indices = np.argsort(additional_x_2_ss)
        additional_x_2_ss = additional_x_2_ss[sort_indices]

        additional_u_1_ss = (
                np.array([j for i, j in additional_pts]) * u_tau
        )

        additional_u_1_ss = additional_u_1_ss[sort_indices]

        # Spalding near-wall section
        lower_cutoff_condition = (x_2_ss_wall_model < additional_x_2_ss[0])
        u_1_ss_wall_model = u_1_ss_wall_model[lower_cutoff_condition]
        x_2_ss_wall_model = x_2_ss_wall_model[lower_cutoff_condition]

        x_2_ss_wall_model = np.append(x_2_ss_wall_model, additional_x_2_ss)
        u_1_ss_wall_model = np.append(u_1_ss_wall_model, additional_u_1_ss)
    else:
        # Spalding near-wall section
        lower_cutoff_condition = (
            x_2_ss_wall_model < y_ss[cutoff_index_lower] - y_0_ss
        )
        u_1_ss_wall_model = u_1_ss_wall_model[lower_cutoff_condition]
        x_2_ss_wall_model = x_2_ss_wall_model[lower_cutoff_condition]

    # Append experimental data for outer section
    x_2_ss_wall_model = np.append(
        x_2_ss_wall_model,
        np.linspace(
            y_ss[cutoff_index_lower] - y_0_ss,
            y_ss[cutoff_index_upper] - y_0_ss,
            1000
        ),
    )
    u_1_ss_wall_model = np.append(
        u_1_ss_wall_model,
        exp_vel(
            np.linspace(
                y_ss[cutoff_index_lower] - y_0_ss,
                y_ss[cutoff_index_upper] - y_0_ss,
                1000,
            )
        ),
    )

    profile["mean_velocity"]["U_SS_MODELED"] = u_1_ss_wall_model
    profile["coordinates"]["Y_SS_MODELED"] = x_2_ss_wall_model


def calculate_boundary_layer_integral_parameters(profile: ProfileData, ui: PRInputs, props: Properties) -> bool:
    """
    Calculate the boundary layer integral parameters for a single experimental hill-normal profile.

    The algorithm uses the profile data expressed in local wall shear stress coordinates (SS system).

    The integral boundary layer parameters are calculated using two methods: the Griffin method and the Vinuesa method.
    Once calculated, the parameters are automatically appended to the profile data.

    :param profile: The hill-normal profile data.
    :param inputs: The user's inputs.
    :param properties: The fluid, flow, and reference properties.
    """
    pressure_data = _get_centerline_pressure(ui, props)

    u_ss = cast(np.ndarray, profile["mean_velocity"]["U_SS"])
    v_ss = cast(np.ndarray, profile["mean_velocity"]["V_SS"])
    y_ss = cast(np.ndarray, profile["coordinates"]["Y_SS"])
    y_0_ss = cast(np.ndarray, profile["properties"]["Y_SS_CORRECTION"])
    u_tau = cast(float, profile["properties"]["U_TAU"])
    nu = profile["properties"]["NU"]
    rho = profile["properties"]["RHO"]
    p_0 = pressure_data["properties"]["P_0"]
    p3 = pressure_data["interpolants"]["P_THIRD_ORDER"]

    exp_velocity_interpolant = interp1d(
        y_ss[~np.isnan(u_ss)] - y_0_ss,
        u_ss[~np.isnan(u_ss)],
        kind="linear",
        fill_value="extrapolate",  # type: ignore[arg-type]
    )

    # Reconstruct velocity profile
    _reconstruct_profile(profile, ui, exp_velocity_interpolant)

    # Show model profile
    y_ss_mod = cast(np.ndarray, profile["coordinates"]["Y_SS_MODELED"])
    u_ss_mod = cast(np.ndarray, profile["mean_velocity"]["U_SS_MODELED"])
    plotting.check_wall_model(
        wall_model=np.array([y_ss_mod * u_tau / nu, u_ss_mod / u_tau]),
        data=np.array([(y_ss - y_0_ss) * u_tau / nu, u_ss / u_tau]),
    )

    integral_parameters = {"griffin": {}, "vinuesa": {}}
    # Local Reconstruction Method, Griffin et al. (2021)
    delta_threshold = 0.99
    u_griffin = (2 / rho) * (p_0 - p3(profile["coordinates"]["X"][0])) - v_ss ** 2

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
