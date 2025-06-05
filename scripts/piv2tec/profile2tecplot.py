import sys
import io
import json
import pickle as pkl
from collections.abc import Mapping
from typing import Any, Literal, Optional

import numpy as np
from numpy.typing import NDArray


# Constant
INFO_SECTION_TEMPLATE = "info_section.in"
H = 0.186944

# Inputs
path_profiles = "../../outputs/plane3/plane3_pr.pkl"
path_properties = "../../outputs/plane3/fluid_and_flow_properties.json"
path_transformation = "../../outputs/plane3/plane3_tp.json"

outfile_name = "Plane3_Re=250k"

hill_orientation = 45.0
reynolds_number = 250000.0
location = (
    "Maximum pressure region along the centerline of the BeVERLI Hill"
)
profiles_orientation: str = "Shear"
profiles_IDs: tuple[int, ...] = (1,)
nan_val = -999.9

piv_rate = 12.5
piv_samp = 10000

# "ZONE": "Plane1_x=neg1p8288m_Re=250k",
# "DESCRIPTION": "Highest pressure point on the BeVERLI Hill at Re_H = 250,000 and 45 deg hill orientation",
# "TITLE": "Plane1_x=neg1p8288m_Re=250k",

def load_pkl(pkl_file: str) -> dict:
    """Load data from a .pkl file.

    :param pkl_file: File in .pkl format.

    :return: Pickle file content.
    :rtype: dict
    """
    with open(pkl_file, "rb") as f:
        data = pkl.load(f)
    return data


def load_json(json_file: str) -> dict:
    """Load data from a .json file.

    :param json_file: JSON file.

    :return: JSON file content.
    :rtype: dict
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def get_bl_parameters(
    pr_pkl: dict,
    pr_IDs: tuple[int, ...],
    orientation: Literal["Shear", "Tunnel"]
) -> list[dict]:
    """Extract boundary layer parameters from .pkl profile data.

    Only available for 'Shear' coordinate system.

    The parameters include integral boundary layer and spalding fit parameters.

    :param pr_pkl: Pickle profile data.
    :param pr_ids: Profile IDs.
    :param cs: Coordinate system type.

    :return: Boundary layer parameters.
    :rtype: dict
    """
    if orientation != "Shear":
        return []

    bl_parameters = []
    for pr_ID in pr_IDs:
        properties = pr_pkl[f"p{pr_ID}"]["exp"]["properties"]
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


def get_varnames() -> list[str]:
    varnames = [
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
    ]

    if profiles_orientation == "Tunnel":
        values_to_remove = {"u_tau/u_ref"}
        varnames = [x for x in varnames if x not in values_to_remove]

    return varnames


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
        "rho": ("properties", "RHO"),
        "nu": ("properties", "NU"),
    }

    if system == "Shear":
        mp["u_tau"] = ("properties", "U_TAU")
        mp["x0"] = ("properties", "X_CORRECTION")
        mp["y0"] = ("properties", "Y_CORRECTION")

    return mp


def get_nested_value(d: dict, keys: tuple[Any, ...]) -> Any:
    """Obtain a value in a nested dictionary, following a series of keys.

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
    pr_IDs: tuple[int, ...],
    pr_orientation: Literal["Shear", "Tunnel"],
) -> list[NDArray[np.float64]]:
    """Extract quantities from the profile data.

    :param pr_pkl: Profile data from file in .pkl format.
    :param pr_IDs: Profile numbers to extract.
    :param pr_orientation: Target coordinate system.
    :param info: Dataset information

    :return: Profile quantities.
    :rtype: list[NDArray[np.float64]]
    """
    pq: list[NDArray[np.float64]] = []

    varmap = get_varmap(pr_orientation)

    for pn in profiles_IDs:
        pr = pr_pkl[f"p{pn}"]["exp"]

        vrs = {}
        for quantity, keys in varmap.items():
            vrs[quantity] = get_nested_value(pr, keys)

        vrs["tke"] = 0.5 * (vrs["uu"] + vrs["vv"] + vrs["ww"])

        NaN = float(nan_val)

        data = np.array([
            vrs["x"] - vrs["x0"] if pr_orientation == "Shear" else vrs["x"],
            vrs["y"] - vrs["y0"] if pr_orientation == "Shear" else vrs["y"],
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
            np.nan_to_num(
                0.5 * np.sqrt(
                    (vrs["duu"] + vrs["dvv"] + vrs["dww"]) / 1.96**2
                ) / vrs["u_ref"]**2, nan=NaN),
        ])
        if pr_orientation == "Shear":
            np.insert(
                data, 13, np.ones_like(vrs["x"]) * vrs["u_tau"] / vrs["u_ref"]
            )
        pq.append(data)

    return pq


def write_banner(f: io.IOBase, properties: dict) -> None:
    """Write the file's information banner.

    :param f: File handle.
    :param properties: Profile properties.

    :return: Info text.
    :rtype: list[str]
    """
    def tab(n: Optional[int] = None) -> str:
        if n:
            return n * 4 * " "
        else:
            return 4 * " "

    tp = load_json(path_transformation)

    if profiles_orientation == "Shear":
        orientation = "Locally normal to the surface of the BeVERLI Hill or tunnel port wall."
    elif profiles_orientation == "Tunnel":
        orientation = "Normal to the surface of the tunnel port wall."
    else:
        raise ValueError("Coordinate system must be 'Shear' or 'Tunnel'.")

    replacements = {
        "NumOfProfiles": (len(profiles_IDs), "{:d}"),
        "H": (H, "{:.6f}"),
        "Xsrc": (tp["translation"]["x_1_glob_ref_m"], "{:.4f}"),
        "Ysrc": (tp["translation"]["x_2_glob_ref_m"], "{:.4f}"),
        "Zsrc": (tp["translation"]["x_3_glob_ref_m"], "{:.4f}"),
        "SourceDescription": (location, "{}"),
        "Orientation": (orientation, "{}"),
        "phi": (hill_orientation, "{:.1f}"),
        "ReH": (reynolds_number, "{}"),
        "uref": (properties["reference"]["U_ref"], "{:.2f}"),
        "pref": (properties["reference"]["p_ref"], "{:.1f}"),
        "Tref": (properties["reference"]["T_ref"], "{:.1f}"),
        "Mref": (properties["reference"]["M_ref"], "{:.2f}"),
        "rhoref": (properties["reference"]["density_ref"], "{:.3f}"),
        "muref": (properties["reference"]["dynamic_viscosity_ref"], "{:.5e}"),
        "p0": (properties["flow"]["p_0"], "{:.1f}"),
        "T0": (properties["flow"]["T_0"], "{:.1f}"),
        "uinf": (properties["flow"]["U_inf"], "{:.2f}"),
        "pinf": (properties["flow"]["p_inf"], "{:.1f}"),
        "pamb": (properties["flow"]["p_atm"], "{:.1f}"),
        "rho": (properties["fluid"]["density"], "{:.3f}"),
        "mu": (properties["fluid"]["dynamic_viscosity"], "{:.5e}"),
        "PivSamplingRate": (piv_rate, "{}"),
        "PivNumOfSamples": (piv_samp, "{:d}"),
    }

    with open(INFO_SECTION_TEMPLATE, "r") as t:
        info = t.read()

    for key, (value, fmt) in replacements.items():
        formatted = fmt.format(value)
        info = info.replace(f"<{key}>", formatted)

    f.write(info)


def prof2tec(
    pr_data: list[NDArray[np.float64]],
    pr_IDs: tuple[int, ...],
    properties: dict,
    bl: list[dict],
) -> None:
    """Write profiles to Tecplot format.

    :param pr_data: Profile data.
    :param pr_IDs: Identification numbers of profiles.
    :param properties: Data properties.
    :param bl: Boundary layer and Spalding parameters.
    """
    with open(outfile_name + ".dat", "w", encoding="utf-8") as f:
        write_banner(f, properties)
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
    profiles_pkl = load_pkl(path_profiles)
    bl_parameters = get_bl_parameters(
        profiles_pkl, profiles_IDs, profiles_orientation
    )
    properties = load_json(path_properties)
    profiles_data = extract_profile_quantities(
        profiles_pkl, profiles_IDs, profiles_orientation
    )
    prof2tec(profiles_data, profiles_IDs, properties, bl_parameters)

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
