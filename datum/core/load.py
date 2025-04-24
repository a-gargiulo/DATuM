"""Load raw PIV data."""

from typing import TYPE_CHECKING

import scipy.io as scio

from ..core.my_types import PPInputs

if TYPE_CHECKING:
    from .piv import Piv


def load_raw_data(piv: "Piv", ui: PPInputs) -> None:
    """Load the `raw` BeVERLI Hill stereo PIV data (.mat format)."""
    mean_velocity = (
        scio.loadmat(ui["piv_data_paths"]["mean_velocity"])
        if ui["load_set"]["mean_velocity"]
        else None
    )
    reynolds_stress = (
        scio.loadmat(ui["piv_data_paths"]["reynolds_stress"])
        if ui["load_set"]["reynolds_stress"]
        else None
    )
    instantaneous_velocity_frame = (
        scio.loadmat(ui["piv_data_paths"]["instantaneous_velocity_frame"])
        if ui["load_set"]["instantaneous_velocity_frame"]
        else None
    )
    turbulence_dissipation = (
        scio.loadmat(ui["piv_data_paths"]["turbulence_dissipation"])
        if ui["load_set"]["turbulence_dissipation"]
        else None
    )

    flip_u_3 = ui["flip_u3"]

    piv_data = {}

    if mean_velocity:
        piv_data["coordinates"] = {
            "X": mean_velocity["X"],
            "Y": mean_velocity["Y"],
        }
        piv_data["mean_velocity"] = {
            key: (-val if key == "W" and flip_u_3 else val)
            for key, val in mean_velocity.items()
            if key in {"U", "V", "W"}
        }

    if reynolds_stress:
        piv_data["reynolds_stress"] = {
            key: (-val if key in {"UW", "VW"} and flip_u_3 else val)
            for key, val in reynolds_stress.items()
            if key in {"UU", "VV", "WW", "UV", "UW", "VW"}
        }
        piv_data["turbulence_scales"] = {
            "TKE": 0.5
            * (
                reynolds_stress["UU"]
                + reynolds_stress["VV"]
                + reynolds_stress["WW"]
            )
        }

    if instantaneous_velocity_frame:
        piv_data["instantaneous_velocity_frame"] = {
            key: (-val if key == "W" and flip_u_3 else val)
            for key, val in instantaneous_velocity_frame.items()
            if key in {"U", "V", "W"}
        }

    if turbulence_dissipation:
        piv_data["turbulence_scales"]["EPSILON"] = turbulence_dissipation[
            "epsVals"
        ]

    piv.data = piv_data
