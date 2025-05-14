"""Transformation sub-module for interpolation operations."""
from typing import Tuple, TYPE_CHECKING

import numpy as np

from ...utility import mathutils

if TYPE_CHECKING:
    from ..piv import Piv


DAT2INTERP = [
    ("mean_velocity", ["U", "V", "W"]),
    ("reynolds_stress", ["UU", "VV", "WW", "UV", "UW", "VW"]),
    ("turbulence_scales", ["TKE", "EPSILON"]),
    ("velocity_snapshot", ["U", "V", "W"]),
]


def interpolate_all(piv: "Piv", n_grid: int):
    """Interpolate all data.

    :param piv: PIV plane data.
    :param n_grid: Number of interpolation grid points.
    """
    x1_q, x2_q = get_interpolation_grid(piv, n_grid)

    coords = get_flattened_coordinates(piv)
    for data_type, keys in DAT2INTERP:
        for key in keys:
            if piv.data[data_type][key] is not None:
                data = get_flattened_data(piv, data_type, key)
                interp_data = mathutils.interpolate(coords, data, (x1_q, x2_q))
                set_interpolated_data(piv, data_type, key, interp_data)

    piv.data["coordinates"]["X"] = x1_q
    piv.data["coordinates"]["Y"] = x2_q


def get_interpolation_grid(
    piv: "Piv", n_grid: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a structured, dense coordinate mesh for interpolation.

    :param piv: Instance of the :py:class:`datum.piv.Piv` class.

    :return: A tuple containing NumPy ndarrays of shape (m, n) that represent a
        mesh grid, where m and n represent the number of available data points
        in the x:sub:`1`- and x:sub:`2`-direction.
    """
    num_of_pts = n_grid
    x1_range = np.linspace(
        np.min(piv.data["coordinates"]["X"]),
        np.max(piv.data["coordinates"]["X"]),
        num_of_pts,
    )
    x2_range = np.linspace(
        np.min(piv.data["coordinates"]["Y"]),
        np.max(piv.data["coordinates"]["Y"]),
        num_of_pts,
    )
    x1_q, x2_q = [matrix.T for matrix in np.meshgrid(x1_range, x2_range)]

    return x1_q, x2_q


def get_flattened_coordinates(piv: "Piv") -> np.ndarray:
    """Retrieve an array of flattened spatial coordinates.

    :param piv: Instance of the :py:class:`datum.piv.Piv` class.
    :return: NumPy ndarray of shape (m, 2) containing the flattened set of
        spatial coordinates for the desired quantity from the stereo PIV data,
        where m represents the number of total available data points.
    """
    flattened_coordinates = np.column_stack(
        (
            piv.data["coordinates"]["X"].flatten(),
            piv.data["coordinates"]["Y"].flatten(),
        )
    )

    return flattened_coordinates


def get_flattened_data(piv: "Piv", data_type: str, key: str) -> np.ndarray:
    """Retrieve an array of flattened data of a specific vector or tensor.

    :param piv: Instance of the :py:class:`datum.piv.Piv` class.
    :param data_type: A string specifying the desired quantity to interpolate.
    :param key: A string indicating the desired vector or tensor component.
    :return: NumPy ndarrays of shape (n, ) containing the flattened set of
        values for the desired quantity's component from the stereo PIV data,
        where n represents the number of total available data points.
    """
    return piv.data[data_type][key].flatten()


def set_interpolated_data(
    piv: "Piv", data_type: str, key: str, interpolated_data: np.ndarray
):
    """Update the PIV data with the interpolated data.

    :param piv: Instance of the :py:class:`datum.piv.Piv` class.
    :param data_type: A string specifying the particular vector or tensor
        quantity to update.
    :param key: A string specifying the particular component of the vector or
        tensor quantity to update.
    :param interpolated_data: NumPy ndarrays of shape (n, ) representing the
        interpolated component, where n represents the number of total
        available data points.
    """
    piv.data[data_type][key] = interpolated_data
