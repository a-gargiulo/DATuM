import sys
import io
import json
import pickle as pkl
from collections.abc import Mapping
from typing import Any, Literal, Optional

import numpy as np
from numpy.typing import NDArray


# Constant
H = 0.186944

# Inputs
profiles_file: str = "./outputs/plane3/plane3_pr.pkl"
properties_file: str = "./outputs/plane3/fluid_and_flow_properties.json"
transform_file: str = "./outputs/plane3/plane3_tp.json"
hill_orientation: float = 45.0
reynolds_number: float = 250000.0
description: str = (
    "Maximum pressure region along the centerline of the BeVERLI Hill"
)
coordinate_system: str = "Shear"
# profile_numbers: tuple[int, ...] = (1, 2, 3)
profile_numbers: tuple[int, ...] = (1,)


def load_pkl(pkl_file: str) -> dict:
    """Load data from file in .pkl format.

    :param pkl_file: File in .pkl format.

    :return: Pickle file content.
    :rtype: dict
    """
    with open(pkl_file, "rb") as f:
        data = pkl.load(f)
    return data


def get_bl_parameters(
    pr_pkl: dict,
    profile_numbers: tuple[int, ...],
    system: Literal["Shear", "Tunnel"]
) -> list[dict]:
    """Extract boundary layer parameters from profile data in .pkl format.

    Only available in for 'Shear' coordinate system.

    The parameters include integral boundary parameters and spalding fit
    parameters.

    :param pr_pkl: Profile data in .pkl format.
    :param profile_numbers: Numbers of the profiles to extract.
    :param system: Coordinate system type.

    :return: Boundary layer parameters.
    :rtype: dict
    """
    if system != "Shear":
        return []

    bl_parameters = []
    for pn in profile_numbers:
        properties = pr_pkl[f"p{pn}"]["exp"]["properties"]
        parameters = properties["BL_PARAMS"]

        bl = {"griffin": {}, "vinuesa": {}, "spalding": {}}

        for method in ("griffin", "vinuesa"):
            bl[method]["u_e"] = parameters[method.upper()]["U_E"]
            bl[method]["delta"] = parameters[method.upper()]["DELTA"]
            bl[method]["delta_star"] = parameters[method.upper()]["DELTA_STAR"]
            bl[method]["theta"] = parameters[method.upper()]["THETA"]

        bl["griffin"]["threshold"] = parameters["GRIFFIN"]["THRESHOLD"]

        bl["spalding"]["X_0"] = properties["X_CORRECTION"]
        bl["spalding"]["Y_0"] = properties["Y_CORRECTION"]
        bl["spalding"]["u_tau"] = properties["U_TAU"]

        bl_parameters.append(bl)

    return bl_parameters


def load_json(file: str) -> dict:
    """Load data from a file in .json format.

    :param file: File in .json format.

    :return: Properties data.
    :rtype: dict
    """
    with open(file, "r") as f:
        data = json.load(f)
    return data


def get_information() -> dict:
    return {
        "DESCRIPTION": "Highest pressure point on the BeVERLI Hill at Re_H = 250,000 and 45 deg hill orientation",
        "TITLE": "Plane1_x=neg1p8288m_Re=250k",
        "VARIABLE_NAMES": [
            "X (m)",
            "Y (m)",
            "Z (m)",
            "u/u_ref",
            "v/u_ref",
            "w/u_ref",
            "TKE/(u_ref)^2",
            "<rho u''u''>/(rho*u_ref^2)",
            "<rho v''v''>/(rho*u_ref^2)",
            "<rho w''w''>/(rho*u_ref^2)",
            "<rho u''v''>/(rho*u_ref^2)",
            "<rho v''w''>/(rho*u_ref^2)",
            "<rho u''w''>/(rho*u_ref^2)",
            "u_tau/u_ref",
            "nu_wall/(u_ref*H)",
            "u/uref_UQ",
            "v/uref_UQ",
            "w/uref_UQ",
            "<rho u''u''>/(rho*u_ref^2)_UQ",
            "<rho v''v''>/(rho*u_ref^2)_UQ",
            "<rho w''w''>/(rho*u_ref^2)_UQ",
            "<rho u''v''>/(rho*u_ref^2)_UQ",
            "<rho v''w''>/(rho*u_ref^2)_UQ",
            "<rho u''w''>/(rho*u_ref^2)_UQ",
            "TKE/(u_ref)^2_UQ"
        ],
        "ZONE": "Plane1_x=neg1p8288m_Re=250k",
        "AUXDATA": {
            "Type": "PIV Profile",
            "HillOrientation": "45",
            "ReynoldsNumber": "250000",
            "NumberOfPoints": "500",
            "ProfileNumber": "1",
            "NaN": "-999.9",
        }
    }


def get_varmap(system: Literal["Shear", "Tunnel"]) -> dict:
    """Get a mapping of dictionary keys simplifying access to the .pkl data.

    :param system: Coordinate system type.

    :raises ValueError: If the coordinate system selction is invalid.

    :return: Map of keys.
    :rtype: dict
    """
    if system == "Shear":
        postfix = "_SS"
    elif system == "Tunnel":
        postfix = ""
    else:
        raise ValueError("Coordinate system must be 'Shear' or 'Tunnel'.")

    mp = {
        "x": ("coordinates", "X"),
        "y": ("coordinates", "Y"),
        "u": ("mean_velocity", f"U{postfix}"),
        "v": ("mean_velocity", f"V{postfix}"),
        "w": ("mean_velocity", f"W{postfix}"),
        "uu": ("reynolds_stress", f"UU{postfix}"),
        "vv": ("reynolds_stress", f"VV{postfix}"),
        "ww": ("reynolds_stress", f"WW{postfix}"),
        "uv": ("reynolds_stress", f"UV{postfix}"),
        "uw": ("reynolds_stress", f"UW{postfix}"),
        "vw": ("reynolds_stress", f"VW{postfix}"),
        "du": ("uncertainty", f"dU{postfix}"),
        "dv": ("uncertainty", f"dV{postfix}"),
        "dw": ("uncertainty", f"dW{postfix}"),
        "duu": ("uncertainty", f"dUU{postfix}"),
        "dvv": ("uncertainty", f"dVV{postfix}"),
        "dww": ("uncertainty", f"dWW{postfix}"),
        "duv": ("uncertainty", f"dUV{postfix}"),
        "duw": ("uncertainty", f"dUW{postfix}"),
        "dvw": ("uncertainty", f"dVW{postfix}"),
        "u_ref": ("properties", "U_REF"),
        "u_tau": ("properties", "U_TAU"),
        "rho": ("properties", "RHO"),
        "nu": ("properties", "NU"),
    }

    if system == "Shear":
        mp["x0"] = ("properties", "X_CORRECTION")
        mp["y0"] = ("properties", "Y_CORRECTION")

    return mp


def get_nested_value(d: dict, keys: tuple[Any, ...]) -> Any:
    """Obtain a specific value in a nested dictionary, following a series of keys.

    :param d: Dictionary to search.
    :param keys: Target keys.

    :return: Value at the final key.
    :rtype: Any
    """
    for key in keys:
        d = d[key]
    return d


def extract_profile_quantities(
    pr_pkl: dict,
    profile_numbers: tuple[int, ...],
    system: Literal["Shear", "Tunnel"],
    info: dict,
) -> list[NDArray[np.float64]]:
    """Extract quantities from the profile data.

    :param pr_pkl: Profile data from file in .pkl format.
    :param profile_numbers: Profile numbers to extract.
    :param system: Target coordinate system.
    :param info: Dataset information

    :return: Profile quantities.
    :rtype: list[NDArray[np.float64]]
    """
    pq: list[NDArray[np.float64]] = []

    varmap = get_varmap(system)

    for pn in profile_numbers:
        pr = pr_pkl[f"p{pn}"]["exp"]

        vrs = {}
        for quantity, keys in varmap.items():
            vrs[quantity] = get_nested_value(pr, keys)

        vrs["tke"] = 0.5 * (vrs["uu"] + vrs["vv"] + vrs["ww"])

        NaN = float(info["AUXDATA"]["NaN"])


        data = np.array([
            vrs["x"] - vrs["x0"],
            vrs["y"] - vrs["y0"],
            np.zeros_like(vrs["x"]),
            np.nan_to_num(vrs["u"] / vrs["u_ref"], nan=NaN),
            np.nan_to_num(vrs["v"] / vrs["u_ref"], nan=NaN),
            np.nan_to_num(vrs["w"] / vrs["u_ref"], nan=NaN),
            np.nan_to_num(vrs["tke"] / vrs["u_ref"]**2, nan=NaN),
            np.nan_to_num(vrs["uu"] / vrs["u_ref"]**2, nan=NaN),
            np.nan_to_num(vrs["vv"] / vrs["u_ref"]**2, nan=NaN),
            np.nan_to_num(vrs["ww"] / vrs["u_ref"]**2, nan=NaN),
            np.nan_to_num(vrs["uv"] / vrs["u_ref"]**2, nan=NaN),
            np.nan_to_num(vrs["vw"] / vrs["u_ref"]**2, nan=NaN),
            np.nan_to_num(vrs["uw"] / vrs["u_ref"]**2, nan=NaN),
            np.ones_like(vrs["x"]) * vrs["u_tau"] / vrs["u_ref"],
            np.ones_like(vrs["x"]) * vrs["nu"] / H / vrs["u_ref"],
            np.nan_to_num(vrs["du"] / abs(vrs["u_ref"]), nan=NaN),
            np.nan_to_num(vrs["dv"] / abs(vrs["u_ref"]), nan=NaN),
            np.nan_to_num(vrs["dw"] / abs(vrs["u_ref"]), nan=NaN),
            np.nan_to_num(vrs["duu"] / vrs["u_ref"]**2, nan=NaN),
            np.nan_to_num(vrs["dvv"] / vrs["u_ref"]**2, nan=NaN),
            np.nan_to_num(vrs["dww"] / vrs["u_ref"]**2, nan=NaN),
            np.nan_to_num(vrs["duv"] / vrs["u_ref"]**2, nan=NaN),
            np.nan_to_num(vrs["dvw"] / vrs["u_ref"]**2, nan=NaN),
            np.nan_to_num(vrs["duw"] / vrs["u_ref"]**2, nan=NaN),
            np.nan_to_num(0.5 * np.sqrt((vrs["duu"] + vrs["dvv"] + vrs["dww"]) / 1.96**2) / vrs["u_ref"]**2, nan=NaN),
        ])
        pq.append(data)

    return pq


def write_banner(
    f: io.IOBase,
    profiles: list[NDArray[np.float64]],
    profile_numbers: tuple[int, ...],
    properties: dict,
    bl: list[dict],
    info: dict
) -> None:
    """Write the file's information banner.

    :param f: File handle.
    :param profiles: Profile data.
    :param profile_numbers: Profile identification numbers.
    :param properties: Profile properties.
    :param bl: Boundary layer parameters.
    :param info: Additional information.

    :return: Info text.
    :rtype: list[str]
    """
    def tab(n: Optional[int] = None) -> str:
        if n:
            return n * 4 * " "
        else:
            return 4 * " "

    tp = load_json(transform_file)

    f.write("#" + 120 * "+" + "\n")
    f.write("#\n")
    f.write(f"#{tab()}Last modified: May 22, 2025)\n")
    f.write(f"#{tab()}\n")
    f.write("#" + tab() + "+" + 79 * "-" + "+" + "\n")
    f.write(f"#{tab()}| NASA-VT Benchmark Validation Experiment for RANS/LES Investigations (BeVERLI) |\n")
    f.write("#" + tab() + "+" + 79 * "-" + "+" + "\n")
    f.write("#\n")
    f.write("#" + tab() + 32 * "~" + "\n")
    f.write(f"#{tab()}One-dimensional (1D) PIV profile\n")
    f.write("#" + tab() + 32 * "~" + "\n")
    f.write("#\n")
    source = (
        f"X = {tp['translation']['x_1_glob_ref_m']:.4f} m, "
        f"Y = {tp['translation']['x_2_glob_ref_m']:.4f} m, "
        f"Z = {tp['translation']['x_3_glob_ref_m']:.4f} m"
    )
    f.write(f"#{tab()}SOURCE:".ljust(33) + f"Stereo PIV plane at {source}\n")
    f.write(f"#{tab(8)}{description}\n")
    f.write("#\n")
    # f.write(f"#{tab()}COORDINATE coordinate_system:\n")
    # f.write(f"#{tab(2)}* Type:".ljust(32) + "Cartesian, (X, Y, Z)\n")
    # f.write(f"#{tab(2)}* Units:".ljust(32) + "Meters, m\n")
    # f.write(f"#{tab(2)}* Origin:".ljust(32) + "Interior center of the BeVERLI Hill on the tunnel port wall\n")
    # f.write(f"#{tab(2)}* X-axis:".ljust(32) + "Positive in the dowstream direction\n")
    # f.write(f"#{tab(2)}* Y-axis:".ljust(32) + "Normal to the tunnel port wall and positive inside of the tunnel\n")
    # f.write(f"#{tab(2)}* Z-axis:".ljust(32) + "Spanwise direction, completing the coordinate system in the right-handed sense\n")
    # f.write("#\n")
    # f.write(f"#{tab()}LOCATION:".ljust(32) + f"X = {profile[0, 0]:.4f} m, Y = {profile[1, 0]:.4f} m, Z = {profile[2, 0]:.4f} m\n")
    # f.write("#\n")
    # f.write(f"#{tab()}ORIENTATION:".ljust(32) + "Normal to the tunnel port wall (X-Y-plane)\n")
    # f.write("#\n")
    # f.write(f"#{tab()}BOUNDARY & REFERENCE CONDITIONS:\n")
    # f.write(f"#{tab(2)}* Density, rho (kg/m^3):".ljust(60) + f"{properties['fluid']['density']}\n")
    # f.write(f"#{tab(2)}* Dynamic (molecular) viscosity, mu (Pa*s):".ljust(60) + f"{properties['fluid']['dynamic_viscosity']:.4e}\n")
    # f.write(f"#{tab(2)}* Free-stream velocity, u_inf (m/s):".ljust(60) + f"{properties['flow']['U_inf']:.2f}\n")
    # f.write(f"#{tab(2)}* Free-stream pressure, p_inf (Pa):".ljust(60) + f"{properties['flow']['p_inf']:.1f}\n")
    # f.write(f"#{tab(2)}* Ambient pressure, p_amb (Pa):".ljust(60) + f"{properties['flow']['p_atm']:.1f}\n")
    # f.write(f"#{tab(2)}* Stagnation pressure, p_0 (Pa):".ljust(60) + f"{properties['flow']['p_0']:.1f}\n")
    # f.write(f"#{tab(2)}* Stagnation temperature, T_0 (K):".ljust(60) + f"{properties['flow']['T_0']:.1f}\n")
    # f.write(f"#{tab(2)}* Reference velocity, u_ref (m/s):".ljust(60) + f"{properties['reference']['U_ref']:.2f}\n")
    # f.write(f"#{tab(2)}* Reference pressure, p_ref, (m/s):".ljust(60) + f"{properties['reference']['p_ref']:.1f}\n")
    # f.write(f"#{tab(2)}* Reference temperature, T_ref (K):".ljust(60) + f"{properties['reference']['T_ref']:.1f}\n")
    # f.write(f"#{tab(2)}* Reference Mach number, M (dimensionless):".ljust(60) + f"{properties['reference']['M_ref']:.2f}\n")
    # f.write(f"#{tab(2)}* Reference density, rho_ref (kg/m^3):".ljust(60) + f"{properties['reference']['density_ref']:.3f}\n")
    # f.write(f"#{tab(2)}* Reference dynamic viscoisty, mu_ref (Pa*s):".ljust(60) + f"{properties['reference']['dynamic_viscosity_ref']:.4e}\n")
    # f.write("#\n")
    # f.write(f"#{tab()}SPALDING FIT:\n")
    # f.write(f"#{tab(2)}* Friction velocity, u_tau (m/s):".ljust(68) + f"{bl['spalding']['u_tau']:.2f}\n")
    # f.write(f"#{tab(2)}* Profile distance to wall correction in X, X_0 (m):".ljust(68) + f"{bl['spalding']['X_0']:.4e}\n")
    # f.write(f"#{tab(2)}* Profile distance to wall correction in Y, Y_0 (m):".ljust(68) + f"{bl['spalding']['X_0']:.4e}\n")
    # f.write("#\n")
    # f.write(f"#{tab()}BOUNDARY LAYER PARAMETERS:\n")
    # f.write(f"#{tab(2)}* Griffin:\n")
    # f.write(f"#{tab(3)}** Edge velocity, U_e (m/s):".ljust(60) + f"{bl['griffin']['u_e']:.2f}\n")
    # f.write(f"#{tab(3)}** Thickness, delta_{bl['griffin']['threshold']} (m):".ljust(60) + f"{bl['griffin']['delta']:.4f}\n")
    # f.write(f"#{tab(3)}** Displacement thickness, delta* (m):".ljust(60) + f"{bl['griffin']['delta_star']:.4f}\n")
    # f.write(f"#{tab(3)}** Momentum thickness, theta (m):".ljust(60) + f"{bl['griffin']['theta']:.4f}\n")
    # f.write("#\n")
    # f.write(f"#{tab(2)}* Vinuesa:\n")
    # f.write(f"#{tab(3)}** Edge velocity, U_e (m/s):".ljust(60) + f"{bl['vinuesa']['u_e']:.2f}\n")
    # f.write(f"#{tab(3)}** Thickness, delta_2.0% (m):".ljust(60) + f"{bl['vinuesa']['delta']:.4f}\n")
    # f.write(f"#{tab(3)}** Displacement thickness, delta* (m):".ljust(60) + f"{bl['vinuesa']['delta_star']:.4f}\n")
    # f.write(f"#{tab(3)}** Momentum thickness, theta (m):".ljust(60) + f"{bl['vinuesa']['theta']:.4f}\n")
    # f.write("#\n")
    # f.write(f"#{tab()}NOMENCLATURE:\n")
    # f.write(f"#{tab(2)}* X = streamwise location in tunnel in meters (X = 0 m is the center of the hill, positive downstream)\n")
    # f.write(f"#{tab(2)}* Y = vertical location in tunnel in meters (Y = 0 m is inside the hill on the tunnel port wall, positive into tunnel)\n")
    # f.write(f"#{tab(2)}* Z = spanwise location in tunnel in meters (Z = 0 m is the center of the hill and in the spanwise direction)\n")
    # f.write(f"#{tab(2)}* u/u_ref = normalized X velocity (dimensionless)\n")
    # f.write(f"#{tab(2)}* v/u_ref = normalized Y velocity (dimensionless)\n")
    # f.write(f"#{tab(2)}* w/u_ref = normalized Z velocity (dimensionless)\n")
    # f.write(f"#{tab(2)}* TKE/(u_ref)^2 = normalized turbulent kinetic energy (dimensionless)\n")
    # f.write(f"#{tab(2)}* omega/(u_ref/H) = normalized turbulent frequency (dimensionless)\n")
    # f.write(f"#{tab(2)}* <rho u''u''>/(rho*u_ref^2) = normalized Reynolds normal stress component (dimensionless)\n")
    # f.write(f"#{tab(2)}* <rho v''v''>/(rho*u_ref^2) = normalized Reynolds normal stress component (dimensionless)\n")
    # f.write(f"#{tab(2)}* <rho w''w''>/(rho*u_ref^2) = normalized Reynolds normal stress component (dimensionless)\n")
    # f.write(f"#{tab(2)}* <rho u''v''>/(rho*u_ref^2) = normalized Reynolds shear stress component (dimensionless)\n")
    # f.write(f"#{tab(2)}* <rho v''w''>/(rho*u_ref^2) = normalized Reynolds shear stress component (dimensionless)\n")
    # f.write(f"#{tab(2)}* <rho u''w''>/(rho*u_ref^2) = normalized Reynolds shear stress component (dimensionless)\n")
    # f.write(f"#{tab(2)}* u_tau/u_ref = normalized wall friction velocity (dimensionless)\n")
    # f.write(f"#{tab(2)}* nu_wall/(u_ref*H) = normalized laminar kinematic viscosity (nu) at the wall (dimensionless)\n")
    # f.write("#\n")
    # f.write("#\n")
    # f.write(f"#{tab()}+" + 36 * "-" + "+\n")
    # f.write(f"#{tab()}| Additional (Important) Information |\n")
    # f.write(f"#{tab()}+" + 36 * "-" + "+\n")
    # f.write("#\n")
    # f.write(f"#{tab()}HILL SURFACE NORMAL VS. TUNNEL PORT WALL NORMAL PROFILES:\n")
    # f.write(f"#{tab(2)}* PIV profiles are extracted either in a direction normal to the tunnel port wall or locally normal to the\n")
    # f.write(f"#{tab(2)}  surface of the BeVERLI Hill, as indicated under 'ORIENTATION'. For hill surface normal profiles, additional\n")
    # f.write(f"#{tab(2)}  parameters, including Spalding fit and boundary layer parameters are reported above. Details on the\n")
    # f.write(f"#{tab(2)}  calculation of these parameters are described below.\n")
    # f.write("#\n")
    # f.write(f"#{tab()}SPALDING FIT PARAMETERS:\n")
    # f.write(f"#{tab(2)}* Where appropriate, i.e., for profiles that or normal to their local surface, the data was fitted to the\n")
    # f.write(f"#{tab(2)}  Spalding [1] composite profile to determine corrections to the profile's distance from the surface, X_0 and\n")
    # f.write(f"#{tab(2)}  Y_0, and the friction velocity, u_tau. In such cases, these parameters are reported above.\n")
    # f.write("#\n")
    # f.write(f"#{tab(2)}  ** [1] Spalding, D. B. (1961). A single formula for the law of the wall. Journal of Applied mechanics, 28(3),\n")
    # f.write(f"#{tab(2)}         455-458.\n")
    # f.write("#\n")
    # f.write(f"#{tab()}BOUNDARY LAYER PARAMETERS:\n")
    # f.write(f"#{tab(2)}* Where available, boundary layer parameters computed using two methods, Griffin et al. [2] and Vinuesa et al. [3],\n")
    # f.write(f"#{tab(2)}  are provided above. The boundary layer thickness, delta, is reported based on either\n")
    # f.write(f"#{tab(2)}  99% or 95% of the edge velocity, U_e, depending on which yielded a more robust estimate, or\n")
    # f.write(f"#{tab(2)}  based on 2.0% of the turbulence intensity.\n")
    # f.write("#\n")
    # f.write(f"#{tab(2)}  ** [2] Vinuesa, R., Bobke, A., Örlü, R., & Schlatter, P. (2016). On determining characteristic length scales\n")
    # f.write(f"#{tab(2)}         in pressure-gradient turbulent boundary layers. Physics of fluids, 28(5).\n")
    # f.write(f"#{tab(2)}  ** [3] Griffin, K. P., Fu, L., & Moin, P. (2021). General method for determining the boundary layer thickness\n")
    # f.write(f"#{tab(2)}         in nonequilibrium flows. Physical Review Fluids, 6(2), 024608.\n")
    # f.write("#\n")
    # f.write(f"#{tab()}UNCERTAINTY QUANTIFICATION (UQ):\n")
    # f.write(f"#{tab(2)}* The reported uncertainties represent 95% confidence intervals and account for both epistemic and aleatory\n")
    # f.write(f"#{tab(2)}  sources of uncertainty. The epistemic component arises from the rotation angles used to transform the raw PIV\n")
    # f.write(f"#{tab(2)}  data from the local measurement coordinate system to the present Cartesian coordinate system. The aleatory\n")
    # f.write(f"#{tab(2)}  component reflects random sampling variability inherent to the measurements. In the dataset, the uncertainties\n")
    # f.write(f"#{tab(2)}  are labeled by prefixing quantities with the letter 'd'.\n")
    # f.write(f"#{tab()}\n")
    # f.write(f"#{tab()}UNAVAILABLE VALUES:\n")
    # f.write(f"#{tab(2)}* Fields with a value of -999.9 represent unavailable data points\n")
    # f.write("#\n")
    # f.write(f"#{tab()}AUXDATA:\n")
    # f.write(f"#{tab(2)}* HillOrientation:".ljust(36) +  "BeVERLI Hill orientation in degrees\n")
    # f.write(f"#{tab(2)}* ReynoldsNumber:".ljust(36) +  "Reynolds number (dimensionless)\n")
    # f.write(f"#{tab(2)}* NumberOfPoints:".ljust(36) +  "Number of profile points\n")
    # f.write(f"#{tab(2)}* ProfileNumber:".ljust(36) +  "Profile index\n")
    # f.write("#\n")


def prof2tec(
    profiles: list[NDArray[np.float64]],
    profile_numbers: tuple[int, ...],
    properties: dict,
    bl: list[dict],
    info: dict
) -> None:
    """Write profiles to Tecplot format.

    :param profiles: Profile data.
    :param profile_numbers: Identification numbers of profiles.
    :param properties: Data properties.
    :param bl: Boundary layer and Spalding parameters.
    :param info: Additional information.

    """
    with open(info["TITLE"] + ".dat", "w", encoding="utf-8") as f:
        write_banner(f, profiles, profile_numbers, properties, bl, info)
        # f.write("\n")
        # f.write(f'TITLE = "{info["TITLE"]}"\n')
        # f.write(f'''VARIABLES = {' '.join([f'"{item}"' for item in info["VARIABLE_NAMES"]])}\n''')
        # f.write("\n")
        # f.write("#-----------------------------------------------------------------------------------------\n")
        # f.write("#   EACH ZONE REFERS TO A DIFFERENT REYNOLDS NUMBER (and possibly grid level)\n")
        # f.write("#-----------------------------------------------------------------------------------------\n")
        # f.write("\n")
        # f.write(f'ZONE T ="{info["ZONE"]}"\n')
        # for key, val in info["AUXDATA"].items():
        #     f.write(f'AUXDATA {key.ljust(16)} = "{val}"\n')
        #     if key == "RelIterConvLevel":
        #         f.write("#  NOTE: the AUXDATA variables below should be the same for every zone\n")
        # f.write("\n")
        # np.savetxt(f, profile.T, fmt="%14.9f")


if __name__ == "__main__":
    # try:
    pr_pkl = load_pkl(profiles_file)
    bl = get_bl_parameters(pr_pkl, profile_numbers, coordinate_system)
    properties = load_json(properties_file)
    info = get_information()
    pr_data = extract_profile_quantities(pr_pkl, profile_numbers, coordinate_system, info)
    prof2tec(pr_data, profile_numbers, properties, bl, info)
    # except Exception as e:
    #     print(f"[ERROR]: {e}")
    #     sys.exit(-1)
