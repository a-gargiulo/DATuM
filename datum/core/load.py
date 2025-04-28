"""Load raw PIV data."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Dict, Set, cast

import numpy as np

from ..core.my_types import (
    Coordinates,
    InstVelFrame,
    MeanVelocity,
    PivData,
    PPInputs,
    ReynoldsStress,
    TurbulenceScales,
)
from ..utility.apputils import safe_loadmat

if TYPE_CHECKING:
    from .piv import Piv

KEYS_MAP = MappingProxyType(
    {
        "X": ("coordinates", "X"),
        "Y": ("coordinates", "Y"),
        "U": ("velocities", "U"),
        "V": ("velocities", "V"),
        "W": ("velocities", "W"),
        "UU": ("reynolds_stress", "UU"),
        "VV": ("reynolds_stress", "VV"),
        "WW": ("reynolds_stress", "WW"),
        "UV": ("reynolds_stress", "UV"),
        "UW": ("reynolds_stress", "UW"),
        "VW": ("reynolds_stress", "VW"),
        "epsVals": ("turbulence_dissipation", "EPSILON"),
    }
)

FLIP_KEYS = frozenset({"W", "UW", "VW"})


def flip(key: str, value: np.ndarray, flip_u_3: bool) -> np.ndarray:
    """Flip 'value' if 'key' is in FLIP_KEYS and flip_u_3 is True.

    :param key: Data component identifier.
    :param value: Data component.
    :param flip_u_3: True if the data should be flipped. False otherwise.

    :return: Flipped or unchanged data dictionary.
    """
    return -value if key in FLIP_KEYS and flip_u_3 else value


def validate_keys(required: Set[str], data: Dict[str, np.ndarray]):
    """Ensure all required keys are in the data dictionary.

    :param required: Components required in the data dictionary.
    :param data: Data dictionary.
    """
    missing = required - data.keys()
    if missing:
        raise RuntimeError(
            f"Missing expected components: {', '.join(missing)}"
        )


def load_dataset(
    path: str, required: Set[str], flip_u_3: bool
) -> Dict[str, np.ndarray]:
    """Load dataset, validate content, and flip components if necessary.

    :param path: System path to raw data.
    :param required: Id's of required data components.
    :param flip_u_3: True if the data should be flipped. False otherwise.
    """
    raw_data = safe_loadmat(path)
    validate_keys(required, raw_data)
    return {
        KEYS_MAP[key][1]: flip(key, val, flip_u_3)
        for key, val in raw_data.items()
        if key in required
    }


def get_keys_for_group(group: str) -> Set[str]:
    """Get raw field names that belong to a group."""
    return {raw_key for raw_key, (grp, _) in KEYS_MAP.items() if grp == group}


def load_raw_data(piv: "Piv", ui: PPInputs):
    """Load the `raw` BeVERLI Hill stereo PIV data (.mat format)."""
    flip_u_3 = ui["flip_u3"]
    paths = ui["piv_data_paths"]
    load_set = ui["load_set"]

    piv_data: PivData = {}

    piv_data["coordinates"] = cast(
        Coordinates,
        load_dataset(
            paths["mean_velocity"], get_keys_for_group("coordinates"), False
        ),
    )
    piv_data["mean_velocity"] = cast(
        MeanVelocity,
        load_dataset(
            paths["mean_velocity"], get_keys_for_group("velocities"), flip_u_3
        ),
    )

    if load_set["reynolds_stress"]:
        piv_data["reynolds_stress"] = cast(
            ReynoldsStress,
            load_dataset(
                paths["reynolds_stress"],
                get_keys_for_group("reynolds_stress"),
                flip_u_3,
            ),
        )
        piv_data["turbulence_scales"] = {
            "TKE": 0.5
            * (
                piv_data["reynolds_stress"]["UU"]
                + piv_data["reynolds_stress"]["VV"]
                + piv_data["reynolds_stress"]["WW"]
            )
        }

    if load_set["instantaneous_velocity_frame"]:
        piv_data["instantaneous_velocity_frame"] = cast(
            InstVelFrame,
            load_dataset(
                paths["instantaneous_velocity_frame"],
                get_keys_for_group("velocities"),
                flip_u_3,
            ),
        )

    if load_set["turbulence_dissipation"]:
        epsilon = load_dataset(
            paths["turbulence_dissipation"],
            get_keys_for_group("turbulence_dissipation"),
            False,
        )
        if "turbulence_scales" not in piv_data:
            piv_data["turbulence_scales"] = cast(TurbulenceScales, epsilon)

        for raw_key, (group, internal_name) in KEYS_MAP.items():
            if group == "turbulence_dissipation":
                piv_data["turbulence_scales"][internal_name] = epsilon[raw_key]

    piv.data = piv_data
