"""Load raw PIV data."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Dict, Set

import numpy as np

from ...utility.apputils import safe_loadmat
from ..my_types import PivData, PPInputs

if TYPE_CHECKING:
    from ..piv import Piv

# raw_name : (group, internal_name)
RAW_DATA_MAP = MappingProxyType(
    {
        "X": ("coordinates", "X"),
        "Y": ("coordinates", "Y"),
        "U": (("mean_velocity", "velocity_snapshot"), "U"),
        "V": (("mean_velocity", "velocity_snapshot"), "V"),
        "W": (("mean_velocity", "velocity_snapshot"), "W"),
        "UU": ("reynolds_stress", "UU"),
        "VV": ("reynolds_stress", "VV"),
        "WW": ("reynolds_stress", "WW"),
        "UV": ("reynolds_stress", "UV"),
        "UW": ("reynolds_stress", "UW"),
        "VW": ("reynolds_stress", "VW"),
        "epsVals": ("turbulence_scales", "EPSILON"),
    }
)

FLIP_KEYS = frozenset({"W", "UW", "VW"})


def make_empty_piv_data() -> PivData:
    """Create an empty piv data dictionary."""
    return {
        "coordinates": {
            "X": np.empty((0,)),
            "Y": np.empty((0,)),
            "Z": None,
        },
        "mean_velocity": {
            "U": np.empty((0,)),
            "V": np.empty((0,)),
            "W": np.empty((0,)),
        },
        "reynolds_stress": None,
        "velocity_snapshot": None,
        "turbulence_scales": {
            "TKE": None,
            "EPSILON": None,
            "NUT": None,
        },
        "mean_velocity_gradient": None,
        "strain_tensor": None,
        "rotation_tensor": None,
        "normalized_rotation_tensor": None,
    }


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


def get_keys_for_group(group: str) -> Set[str]:
    """Get raw field names that belong to a group."""
    return {
        raw_key
        for raw_key, (grp, _) in RAW_DATA_MAP.items()
        if (isinstance(grp, tuple) and group in grp) or (grp == group)
    }


def load_dataset(piv_data: PivData, group: str, path: str, flip_u_3: bool):
    """Load dataset, validate content, and flip components if necessary.

    :param path: System path to raw data.
    :param required: Id's of required data components.
    :param flip_u_3: True if the data should be flipped. False otherwise.
    """
    raw_data = safe_loadmat(path)
    required_keys = get_keys_for_group(group)
    validate_keys(required_keys, raw_data)

    if piv_data[group] is None:
        piv_data[group] = {}

    for raw_key in required_keys:
        _, internal_key = RAW_DATA_MAP[raw_key]
        value = flip(raw_key, raw_data[raw_key], flip_u_3)
        piv_data[group][internal_key] = value


def load_raw_data(piv: "Piv", ui: PPInputs):
    """Load the `raw` BeVERLI Hill stereo PIV data (.mat format)."""
    flip_u_3 = ui["flip_u3"]
    paths = ui["piv_data_paths"]
    load_set = ui["load_set"]

    piv_data: PivData = make_empty_piv_data()

    load_dataset(piv_data, "coordinates", paths["mean_velocity"], False)
    load_dataset(piv_data, "mean_velocity", paths["mean_velocity"], flip_u_3)

    if load_set["reynolds_stress"]:
        load_dataset(
            piv_data, "reynolds_stress", paths["reynolds_stress"], flip_u_3
        )
        if piv_data["reynolds_stress"] and piv_data["turbulence_scales"]:
            piv_data["turbulence_scales"]["TKE"] = 0.5 * (
                piv_data["reynolds_stress"]["UU"]
                + piv_data["reynolds_stress"]["VV"]
                + piv_data["reynolds_stress"]["WW"]
            )

    if load_set["velocity_snapshot"]:
        load_dataset(
            piv_data, "velocity_snapshot", paths["velocity_snapshot"], flip_u_3
        )

    if load_set["turbulence_dissipation"]:
        load_dataset(
            piv_data,
            "turbulence_scales",
            paths["turbulence_dissipation"],
            False,
        )

    piv.data = piv_data
