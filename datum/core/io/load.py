"""Load raw PIV data."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Dict, Set

import numpy as np

from datum.core.my_types import PivData, PPInputs
from datum.utility.apputils import safe_loadmat
from datum.utility.logging import logger

if TYPE_CHECKING:
    from ..piv import Piv

# Map structure
# raw_data_name : (data_group, internal_data_name)
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
    """Create empty BeVERLI Hill stereo PIV flow data.

    :return: Flow data.
    :rtype: PivData
    """
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


def flip(key: str, dataset: np.ndarray, flip_u3: bool) -> np.ndarray:
    """Flip a dataset if the key is in FLIP_KEYS and flip_u3 is True.

    :param key: Dataset identifier.
    :param dataset: Dataset.
    :param flip_u3: True if the user instructs to flip the data, False
        otherwise.
    :return: Flipped/Unflipped dataset.
    """
    return -dataset if key in FLIP_KEYS and flip_u3 else dataset


def validate_keys(
    required_keys: Set[str], raw_data: Dict[str, np.ndarray]
) -> None:
    """Ensure that the raw data comprises all required datasets.

    :param required_keys: Identifiers for the required datasets.
    :param raw_data: Raw data.
    :raises RuntimeError: If the raw data is missing a required dataset.
    """
    missing = required_keys - raw_data.keys()
    if missing:
        raise RuntimeError(f"Missing datasets: {', '.join(missing)}")


def get_keys_for_group(group: str) -> Set[str]:
    """Get raw data field names that belong to a specified data group.

    :param group: Group identifier.
    :return: Raw data field names for the specified data group.
    :rtype: Set[str]
    """
    return {
        raw_key
        for raw_key, (grp, _) in RAW_DATA_MAP.items()
        if (isinstance(grp, tuple) and group in grp) or (grp == group)
    }


def load_dataset(
    piv_data: PivData, group: str, path: str, flip_u3: bool
) -> None:
    """Load dataset, validate content, and flip components if necessary.

    :param path: System path to raw data.
    :param required: Id's of required data components.
    :param flip_u3: True if the data should be flipped. False otherwise.
    :raises RuntimeError: If the data is not loaded successfully at any
        step.
    """
    raw_data = safe_loadmat(path)
    required_keys = get_keys_for_group(group)
    validate_keys(required_keys, raw_data)

    if piv_data[group] is None:
        piv_data[group] = {}

    for raw_key in required_keys:
        _, internal_key = RAW_DATA_MAP[raw_key]
        value = flip(raw_key, raw_data[raw_key], flip_u3)
        piv_data[group][internal_key] = value

    logger.info(f"{group}: {required_keys} loaded successfully.")


def load_raw_data(piv: "Piv", ui: PPInputs) -> None:
    """Load the raw BeVERLI Hill stereo PIV data.

    :param piv: PIV plane data.
    :param ui: User inputs from the preprocessing GUI.
    :raises RuntimeError: If the data is loaded unsuccessfully at any
        step.
    """
    flip_u3 = ui["flip_u3"]
    paths = ui["piv_data_paths"]
    load_set = ui["load_set"]

    piv_data: PivData = make_empty_piv_data()

    # Coordinates
    load_dataset(piv_data, "coordinates", paths["mean_velocity"], False)

    # Mean velocity
    load_dataset(piv_data, "mean_velocity", paths["mean_velocity"], flip_u3)

    # Reynolds stress
    if load_set["reynolds_stress"]:
        load_dataset(
            piv_data, "reynolds_stress", paths["reynolds_stress"], flip_u3
        )
        if piv_data["reynolds_stress"] and piv_data["turbulence_scales"]:
            piv_data["turbulence_scales"]["TKE"] = 0.5 * (
                piv_data["reynolds_stress"]["UU"]
                + piv_data["reynolds_stress"]["VV"]
                + piv_data["reynolds_stress"]["WW"]
            )

    # Velocity snapshot
    if load_set["velocity_snapshot"]:
        load_dataset(
            piv_data, "velocity_snapshot", paths["velocity_snapshot"], flip_u3
        )

    # Turbulence dissipation
    if load_set["turbulence_dissipation"]:
        load_dataset(
            piv_data,
            "turbulence_scales",
            paths["turbulence_dissipation"],
            False,
        )

    piv.data = piv_data
