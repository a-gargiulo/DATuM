"""Transformation sub-module for rotation operations."""
from typing import TYPE_CHECKING, Literal
import numpy as np

from ..my_types import Vec3
from .config import DATA_TO_ROTATE

if TYPE_CHECKING:
    from ..piv import Piv

Axis = Literal["x", "y", "z"]


def get_rotation_matrix(phi_deg: float, axis: Axis) -> np.ndarray:
    """Get an Euler rotation matrix about a specified axis.

    The matrix represents a body's Euler rotation about the specified axis of
    the body's Cartesian coordinate system.

    :param phi_deg: Rotation angle.
    :param axis: Identifier for Cartesian coordinate system axis.
    :return: Rotation matrix as :py:type:`np.ndarray` of shape (3, 3).
    """
    phi_rad = phi_deg * np.pi / 180.0
    if axis == 'z':
        return np.array([
            [np.cos(phi_rad), -np.sin(phi_rad), 0],
            [np.sin(phi_rad), np.cos(phi_rad), 0],
            [0, 0, 1]
        ])
    elif axis == 'y':
        return np.array([
            [np.cos(phi_rad), 0, np.sin(phi_rad)],
            [0, 1, 0],
            [-np.sin(phi_rad), 0, np.cos(phi_rad)]
        ])
    elif axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, np.cos(phi_rad), -np.sin(phi_rad)],
            [0, np.sin(phi_rad), np.cos(phi_rad)]
        ])
    else:
        raise ValueError(f"Invalid axis {axis}. Supported: 'x', 'y', 'z'")


def rotate_data(piv: "Piv"):
    """Rotate all PIV data from local to global coordinates."""
    rotation_matrix = _obtain_rotation_matrix(piv)

    for quantity, components in DATA_TO_ROTATE:
        is_avail = piv.search(quantity)

        if is_avail:
            rotate_flow_quantity(
                piv, quantity, components, rotation_matrix
            )



