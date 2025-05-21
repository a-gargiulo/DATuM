"""Write a 1D PIV profile to Tecplot format."""
import numpy as np
from numpy.typing import NDArray
from typing import Literal, Any
from collections.abc import Mapping
import pickle as pkl

H = 0.186944
PROFILE_FILE = "../../outputs/plane3/plane3_pr.pkl"
SYSTEM = "Shear"
INFO = {
    "TITLE": "Plane3_x=neg0p4372m_Re=250k",
    "VARIABLES": [
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
    "ZONE": "Plane3_x=neg0p4372m_Re=250k",
    "AUXDATA": {
        "ReynoldsNumber": "250k",
        "GridLevel": "Level2",
        "GridCells": "500",
        "RelIterConvLevel": "1E-8",
        "ID": "X",
        "Name": "Chris Roy",
        "SolverName": "Experiment",
        "BasicAlgorithm": "Cell Center Finite Volume Method",
        "TurbulenceModel": "PIV",
        "FlowEqnOrder": "2",
        "TurbEqnOrder": "2",
        "Geometry": "As-Built",
        "Profile": "BLProfile#1",
        "Miscellaneous": "-999.9",
    }
}


def get_map(system: Literal["Shear", "Tunnel"]) -> dict:
    """Get variable map.

    :param system: Coordinate system type.
    :raises ValueError: If the coordinate system selction is invalid.
    :return: Map dictionary.
    :rtype: dict
    """
    if system == "Shear":
        postfix = "_SS"
    elif system == "Tunnel":
        postfix = ""
    else:
        raise ValueError("System can only be 'Shear' or 'Tunnel'.")

    mapp = {
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
        mapp["y0_gx"] = ("properties", "X_CORRECTION")
        mapp["y0_gy"] = ("properties", "Y_CORRECTION")
        mapp["y0"] = ("properties", "Y_SS_CORRECTION")

    return mapp


def get_banner() -> list[str]:
    """Retrieve info banner.

    :return: Info text.
    :rtype: list[str]
    """
    with open("banner.txt", "r", encoding="utf-8") as f:
        banner = f.readlines()
    return banner


def get_nested(d: Mapping, keys: tuple[Any, ...]) -> Any:
    """Get value in nested dictionary.

    :param d: Dictionary.
    :param keys: Target keys.

    :return: Value at the final key.
    :rtype: Any
    """
    for key in keys:
        d = d[key]
    return d


def get_profile_data(file_name: str, system: Literal["Shear", "Tunnel"]) -> NDArray[np.float64]:
    """Retrieve profile data to write.

    :param file_name: Profile data file.
    :param system: Target coordinate system ('Shear', 'Tunnel').

    :return: Writable profile data.
    :rtype: NDArray[np.float64]
    """
    with open(file_name, "rb") as f:
        profile_data = pkl.load(f)
    pr = profile_data["p1"]["exp"]

    pmap = get_map(system)

    vrs = {}
    for quantity, keys in pmap.items():
        vrs[quantity] = get_nested(pr, keys)

    vrs["tke"] = 0.5 * (vrs["uu"] + vrs["vv"] + vrs["ww"])

    NaN = float(INFO["AUXDATA"]["Miscellaneous"])

    writable_pr = np.array([
        vrs["x"] - vrs["y0_gx"],
        vrs["y"] - vrs["y0_gy"],
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

    return writable_pr


def prof2tec(profile: NDArray[np.float64], info: dict) -> None:
    """Write a profile to Tecplot format."""
    banner = get_banner()
    with open(info["TITLE"] + ".dat", "w", encoding="utf-8") as f:
        for line in banner:
            f.write(line)
        f.write("\n")
        f.write(f'TITLE = "{info["TITLE"]}"\n')
        f.write(f'''VARIABLES = {' '.join([f'"{item}"' for item in info["VARIABLES"]])}\n''')
        f.write("\n")
        f.write("#-----------------------------------------------------------------------------------------\n")
        f.write("#   EACH ZONE REFERS TO A DIFFERENT REYNOLDS NUMBER (and possibly grid level)\n")
        f.write("#-----------------------------------------------------------------------------------------\n")
        f.write("\n")
        f.write(f'ZONE T ="{info["ZONE"]}"\n')
        f.write("#  NOTE: the AUXDATA variables below should change for every zone (i.e., every Reynolds number or grid level)\n")
        for key, val in info["AUXDATA"].items():
            f.write(f'AUXDATA {key.ljust(16)} = "{val}"\n')
            if key == "RelIterConvLevel":
                f.write("#  NOTE: the AUXDATA variables below should be the same for every zone\n")
        f.write("\n")
        np.savetxt(f, profile.T, fmt="%14.9f")


if __name__ == "__main__":
    pr_data = get_profile_data(PROFILE_FILE, SYSTEM)
    prof2tec(pr_data, INFO)
