"""Transformation sub-module for scaling operations."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..piv import Piv


def scale_all(piv: "Piv", scale_factor: float):
    """Scale PIV data plane."""
    piv.data["coordinates"]["X"] *= scale_factor
    piv.data["coordinates"]["Y"] *= scale_factor
