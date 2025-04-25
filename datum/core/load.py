"""Load raw PIV data."""

from typing import TYPE_CHECKING, cast, Set, Dict

import numpy as np

from ..utility.apputils import safe_loadmat

from ..core.my_types import (
    InstVelFrame,
    MeanVelocity,
    PivData,
    PPInputs,
    ReynoldsStress,
)

if TYPE_CHECKING:
    from .piv import Piv


FLIP_KEYS = {"W", "UW", "VW"}
COORDS_KEYS = {"X", "Y"}
VELOCITY_KEYS = {"U", "V", "W"}
STRESS_KEYS = {"UU", "VV", "WW", "UV", "UW", "VW"}


def load_raw_data(piv: "Piv", ui: PPInputs):
    """Load the `raw` BeVERLI Hill stereo PIV data (.mat format)."""
    flip_u_3 = ui["flip_u3"]

    def flip(k: str, v: np.ndarray):
        if (k in FLIP_KEYS) and flip_u_3:
            return -v
        return v

    def validate_keys(req: Set[str], data: Dict[str, np.ndarray]):
        if not req <= data.keys():
            raise RuntimeError(
                    "Missing expected components from " + ", ".join(req)
            )

    piv_data: PivData = {}

    mean_velocity = safe_loadmat(ui["piv_data_paths"]["mean_velocity"])
    validate_keys(COORDS_KEYS, mean_velocity)
    validate_keys(VELOCITY_KEYS, mean_velocity)

    piv_data["coordinates"] = {
        "X": mean_velocity["X"],
        "Y": mean_velocity["Y"],
    }
    piv_data["mean_velocity"] = cast(
        MeanVelocity,
        {
            key: flip(key, val)
            for key, val in mean_velocity.items()
            if key in VELOCITY_KEYS
        },
    )

    if ui["load_set"]["reynolds_stress"]:
        reynolds_stress = safe_loadmat(ui["piv_data_paths"]["reynolds_stress"])
        validate_keys(STRESS_KEYS, reynolds_stress)
        piv_data["reynolds_stress"] = cast(
            ReynoldsStress,
            {
                key: flip(key, val)
                for key, val in reynolds_stress.items()
                if key in STRESS_KEYS
            },
        )
        piv_data["turbulence_scales"] = {
            "TKE": 0.5
            * (
                reynolds_stress["UU"]
                + reynolds_stress["VV"]
                + reynolds_stress["WW"]
            )
        }

    if ui["load_set"]["instantaneous_velocity_frame"]:
        instantaneous_velocity_frame = safe_loadmat(
            ui["piv_data_paths"]["instantaneous_velocity_frame"]
        )
        piv_data["instantaneous_velocity_frame"] = cast(
            InstVelFrame,
            {
                key: (-val if key == "W" and flip_u_3 else val)
                for key, val in instantaneous_velocity_frame.items()
                if key in {"U", "V", "W"}
            },
        )

    if ui["load_set"]["turbulence_dissipation"]:
        turbulence_dissipation = safe_loadmat(
            ui["piv_data_paths"]["turbulence_dissipation"]
        )
        if "turbulence_scales" not in piv_data:
            raise RuntimeError("Failed to assign 'EPSILON'")
        piv_data["turbulence_scales"]["EPSILON"] = turbulence_dissipation[
            "epsVals"
        ]

    piv.data = piv_data
