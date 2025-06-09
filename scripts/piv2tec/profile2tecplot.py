"""Write PIV profile data to Tecplot format."""

import io
import json
import pickle as pkl
import sys
import traceback
from typing import Any, Literal, Optional

import numpy as np
from colorama import Fore, Style, init
from numpy.typing import NDArray

init()  # colorama

# Inputs
H = 0.186944

# "ZONE": "Plane1_x=neg1p8288m_Re=250k",
# "TITLE": "Plane1_x=neg1p8288m_Re=250k",


def get_flag_value(flag: str, default: Any) -> Any:
    """Return the command-line argument value that follows a given flag.

    :param flag: Command-line flag
    :param default: Default value if the flag is not used.

    :return: Flag value or a default
    :rtype: Any
    """
    if flag in sys.argv:
        flag_idx = sys.argv.index(flag)
        if flag_idx + 1 < len(sys.argv):
            return sys.argv[flag_idx + 1]
        else:
            raise RuntimeError(f"{flag} flag provided but no value provided.")
    return default


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
    orientation: Literal["Shear", "Tunnel"],
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


def get_varnames(orientation: str) -> list[str]:
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
        "TKE/(u_ref)^2_UQ",
    ]

    if orientation == "Tunnel":
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
    nan_val: float,
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

    for pn in pr_IDs:
        pr = pr_pkl[f"p{pn}"]["exp"]

        vrs = {}
        for quantity, keys in varmap.items():
            vrs[quantity] = get_nested_value(pr, keys)

        vrs["tke"] = 0.5 * (vrs["uu"] + vrs["vv"] + vrs["ww"])

        NaN = nan_val

        data = np.array(
            [
                (
                    vrs["x"] - vrs["x0"]
                    if pr_orientation == "Shear"
                    else vrs["x"]
                ),
                (
                    vrs["y"] - vrs["y0"]
                    if pr_orientation == "Shear"
                    else vrs["y"]
                ),
                np.zeros_like(vrs["x"]),
                np.nan_to_num(vrs["u"] / vrs["u_ref"], nan=NaN),
                np.nan_to_num(vrs["v"] / vrs["u_ref"], nan=NaN),
                np.nan_to_num(vrs["w"] / vrs["u_ref"], nan=NaN),
                np.nan_to_num(vrs["tke"] / vrs["u_ref"] ** 2, nan=NaN),
                np.nan_to_num(vrs["uu"] / vrs["u_ref"] ** 2, nan=NaN),
                np.nan_to_num(vrs["vv"] / vrs["u_ref"] ** 2, nan=NaN),
                np.nan_to_num(vrs["ww"] / vrs["u_ref"] ** 2, nan=NaN),
                np.nan_to_num(vrs["uv"] / vrs["u_ref"] ** 2, nan=NaN),
                np.nan_to_num(vrs["vw"] / vrs["u_ref"] ** 2, nan=NaN),
                np.nan_to_num(vrs["uw"] / vrs["u_ref"] ** 2, nan=NaN),
                np.ones_like(vrs["x"]) * vrs["nu"] / H / vrs["u_ref"],
                np.nan_to_num(vrs["du"] / abs(vrs["u_ref"]), nan=NaN),
                np.nan_to_num(vrs["dv"] / abs(vrs["u_ref"]), nan=NaN),
                np.nan_to_num(vrs["dw"] / abs(vrs["u_ref"]), nan=NaN),
                np.nan_to_num(vrs["duu"] / vrs["u_ref"] ** 2, nan=NaN),
                np.nan_to_num(vrs["dvv"] / vrs["u_ref"] ** 2, nan=NaN),
                np.nan_to_num(vrs["dww"] / vrs["u_ref"] ** 2, nan=NaN),
                np.nan_to_num(vrs["duv"] / vrs["u_ref"] ** 2, nan=NaN),
                np.nan_to_num(vrs["dvw"] / vrs["u_ref"] ** 2, nan=NaN),
                np.nan_to_num(vrs["duw"] / vrs["u_ref"] ** 2, nan=NaN),
                np.nan_to_num(
                    0.5
                    * np.sqrt((vrs["duu"] + vrs["dvv"] + vrs["dww"]) / 1.96**2)
                    / vrs["u_ref"] ** 2,
                    nan=NaN,
                ),
            ]
        )
        if pr_orientation == "Shear":
            data = np.insert(
                data,
                13,
                np.ones_like(vrs["x"]) * vrs["u_tau"] / vrs["u_ref"],
                axis=0,
            )
        pq.append(data)

    return pq


def write_info(f: io.IOBase, properties: dict, config: dict) -> None:
    """Write the file's information banner.

    :param f: File handle.
    :param properties: Profile properties.
    :param config: Input parameters

    :return: Info text.
    :rtype: list[str]
    """

    def tab(n: Optional[int] = None) -> str:
        if n:
            return n * 4 * " "
        else:
            return 4 * " "

    tp = load_json(config["transformation_path"])

    if config["profiles_orientation"] == "Shear":
        orientation = "Locally normal to the surface of the BeVERLI Hill or tunnel port wall."
    elif config["profiles_orientation"] == "Tunnel":
        orientation = "Normal to the surface of the tunnel port wall."
    else:
        raise ValueError("Coordinate system must be 'Shear' or 'Tunnel'.")

    replacements = {
        "NumOfProfiles": (len(config["profiles_IDs"]), "{:d}"),
        "H": (H, "{:.6f}"),
        "Xsrc": (tp["translation"]["x_1_glob_ref_m"], "{:.4f}"),
        "Ysrc": (tp["translation"]["x_2_glob_ref_m"], "{:.4f}"),
        "Zsrc": (tp["translation"]["x_3_glob_ref_m"], "{:.4f}"),
        "SourceDescription": (config["src_description"], "{}"),
        "Orientation": (orientation, "{}"),
        "phi": (config["hill_orientation"], "{:.1f}"),
        "ReH": (config["reynolds_number"], "{}"),
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
        "PivSamplingRate": (config["piv_rate"], "{}"),
        "PivNumOfSamples": (config["piv_samples"], "{:d}"),
    }

    with open(config["info_template"], "r") as t:
        info = t.read()

    for key, (value, fmt) in replacements.items():
        formatted = fmt.format(value)
        info = info.replace(f"<{key}>", formatted)

    f.write(info)


def prof2tec(
    pr_data: list[NDArray[np.float64]],
    properties: dict,
    bl: list[dict],
    config: dict,
) -> None:
    """Write profiles to Tecplot format.

    :param pr_data: Profile data.
    :param properties: Data properties.
    :param bl: Boundary layer and Spalding parameters.
    :param config: Input parameters
    """
    varnames = get_varnames(config["profiles_orientation"])

    with open(config["output_file"], "w", encoding="utf-8") as f:
        write_info(f, properties, config)
        f.write("\n")
        f.write(f'TITLE = "{config["title"]}"\n')
        f.write(
            f"""VARIABLES = {' '.join([f'"{item}"' for item in varnames])}\n"""
        )
        f.write("\n")

        for pp in range(len(config["profiles_IDs"])):
            indices = []
            for ii in range(pr_data[pp].shape[0]):
                row = pr_data[pp][ii, :]
                mask = row == config["nan_val"]
                rev_mask = mask[::-1]
                first_false = np.where(~rev_mask)[0][0]
                if first_false.size > 0:
                    last_valid_idx = len(row) - 1 - first_false
                    indices.append(last_valid_idx)
                else:
                    indices.append(pr_data[pp].shape[1])
            LX = min(indices)
            xw = f"{pr_data[pp][0, 0] + bl[pp]['spalding']['X_0']:.4e}"
            yw = f"{pr_data[pp][1, 0] + bl[pp]['spalding']['Y_0']:.4e}"
            zw = f"{pr_data[pp][2, 0]:.4e}"
            f.write(
                f"ZONE T".ljust(29)
                + f'= "Profile_{config["profiles_IDs"][pp]}_X={xw}_Y={yw}_Z={zw}"\n'
            )
            f.write(
                f'AUXDATA {"number_of_points".ljust(20)} = "{pr_data[pp][:, 0:LX+1].shape[1]}"\n'
            )
            f.write(
                f"AUXDATA {'profile_number'.ljust(20)} = \"{config['profiles_IDs'][pp]}\"\n"
            )
            f.write(
                f"AUXDATA {'X_0'.ljust(20)} = \"{bl[pp]['spalding']['X_0']:.4e}\"\n"
            )
            f.write(
                f"AUXDATA {'Y_0'.ljust(20)} = \"{bl[pp]['spalding']['Y_0']:.4e}\"\n"
            )
            f.write(
                f"AUXDATA {'U_e_griffin'.ljust(20)} = \"{bl[pp]['griffin']['u_e']:.2f}\"\n"
            )
            threshold = bl[pp]["griffin"]["threshold"]
            f.write(
                f"AUXDATA {f'delta{int(threshold*100)}_griffin'.ljust(20)} = \"{bl[pp]['griffin']['delta']:.3f}\"\n"
            )
            f.write(
                f"AUXDATA {f'delta_star_griffin'.ljust(20)} = \"{bl[pp]['griffin']['delta_star']:.3f}\"\n"
            )
            f.write(
                f"AUXDATA {f'theta_griffin'.ljust(20)} = \"{bl[pp]['griffin']['theta']:.3f}\"\n"
            )
            f.write(
                f"AUXDATA {'U_e_vinuesa'.ljust(20)} = \"{bl[pp]['vinuesa']['u_e']:.2f}\"\n"
            )
            f.write(
                f"AUXDATA {f'delta02_vinuesa'.ljust(20)} = \"{bl[pp]['vinuesa']['delta']:.3f}\"\n"
            )
            f.write(
                f"AUXDATA {f'delta_star_vinuesa'.ljust(20)} = \"{bl[pp]['vinuesa']['delta_star']:.3f}\"\n"
            )
            f.write(
                f"AUXDATA {f'theta_vinuesa'.ljust(20)} = \"{bl[pp]['vinuesa']['theta']:.3f}\"\n"
            )
            f.write("\n")
            np.savetxt(f, pr_data[pp][:, 0 : LX + 1].T, fmt="%14.9f")
            f.write("\n")


def validate_config(config) -> None:
    types = {
        "info_template": str,
        "profiles_path": str,
        "properties_path": str,
        "transformation_path": str,
        "output_file": str,
        "title": str,
        "zone_names": tuple,
        "hill_orientation": float,
        "reynolds_number": float,
        "src_description": str,
        "profiles_orientation": str,
        "profiles_IDs": tuple,
        "nan_val": float,
        "piv_rate": float,
        "piv_samples": int,
    }

    for key in config.keys():
        if not isinstance(config[key], types[key]):
            raise ValueError(
                f"Expected {types[key]} type for '{key}' in 'config', got {type(config[key])} instead."
            )

    for name in config["zone_names"]:
        if not isinstance(name, str):
            raise ValueError(
                f"Expected str type for zone name in 'config['zone_names']', got {type(name)} instead."
            )

    for ID in config["profiles_IDs"]:
        if not isinstance(ID, int):
            raise ValueError(
                f"Expected int type for id in 'config['profiles_IDs']', got {type(ID)} instead."
            )

    if len(config["zone_names"]) != len(config["profiles_IDs"]):
        raise ValueError(
            "Length of 'zone_names' and 'profiles_IDs' in 'config' must match."
        )


if __name__ == "__main__":
    config = {
        "info_template": "../../datum/resources/piv2tec/info_section.in",
        "profiles_path": "../../outputs/plane1_650k/plane1_pr.pkl",
        "properties_path": "../../outputs/plane1_650k/fluid_and_flow_properties.json",
        "transformation_path": "../../outputs/plane1_250k/plane1_tp.json",
        "output_file": get_flag_value("-o", "Plane1_Profiles.dat"),
        "title": "Inflow_Profiles_FastPiv_ReH=650k_Phi=45",
        "zone_names": ("Profile 1", "Profile 2", "Profile 3"),
        "hill_orientation": 45.0,
        "reynolds_number": 650000.0,
        "src_description": "Tunnel inflow profiles",
        "profiles_orientation": "Shear",
        "profiles_IDs": (1, 2, 3),
        "nan_val": -999.9,
        "piv_rate": 12.5,
        "piv_samples": 10000,
    }

    try:
        validate_config(config)
        profiles_pkl = load_pkl(config["profiles_path"])
        bl_parameters = get_bl_parameters(
            profiles_pkl,
            config["profiles_IDs"],
            config["profiles_orientation"],
        )
        properties = load_json(config["properties_path"])
        profiles_data = extract_profile_quantities(
            profiles_pkl,
            config["profiles_IDs"],
            config["profiles_orientation"],
            config["nan_val"],
        )
        prof2tec(profiles_data, properties, bl_parameters, config)
    except Exception as e:
        print(
            "["
            + Style.BRIGHT
            + Fore.RED
            + "ERROR"
            + Style.RESET_ALL
            + f"]: {e}"
        )
        traceback.print_exc()
        sys.exit(1)
