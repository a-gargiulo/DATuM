"""Custom mathematical functions."""

from typing import Tuple

import numpy as np
import scipy.interpolate as spinterpolate


def calculate_derivative_1d(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Compute the first derivative of x2 with respect to x1.

    The algorithm operates on 1D data and uses a simple second-order
    centered finite difference approximation.

    :param x1: Independent variable. :parma x2: Dependent variable.
    :return: dx2/dx1
    :rtype: numpy.ndarray
    """
    num_of_pts = len(x1)
    x2_prime = np.zeros((num_of_pts,))

    for i in range(num_of_pts):
        if i == 0:
            x2_prime[i] = (x2[i + 1] - x2[i]) / (x1[i + 1] - x1[i])
        elif i == num_of_pts - 1:
            x2_prime[i] = (x2[i] - x2[i - 1]) / (x1[i] - x1[i - 1])
        else:
            x2_prime[i] = (x2[i + 1] - x2[i - 1]) / (x1[i + 1] - x1[i - 1])

    return x2_prime


def compute_derivative_2d(
    x1: np.ndarray, x2: np.ndarray, u_i: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """First derivative of planar data of a vector quantity u:sub:`i`.

    The calculation uses fourth-order centered finite difference approximation.

    :param x1: Independent variable in the first dimension as NumPy ndarray of
        shape (n, ).
    :param x2: Independent variable in the second dimension as NumPy ndarray of
        shape (n, ).
    :param u_i: Dependent variable as NumPy ndarray of shape (n, ).

    :return: Gradient of u:sub:`i` in the x:sub:`1` and x:sub:`2`,
        respectively, in a tuple of NumPy ndarrays of shape (m, n).
    :rtype: Tuple[numpy.ndarray, numpy.ndarray]
    """
    dudx = np.zeros_like(u_i)
    dudy = np.zeros_like(u_i)

    dudx[2:-2, :] = (
        -u_i[4:, :] + 8 * u_i[3:-1, :] - 8 * u_i[1:-3, :] + u_i[0:-4, :]
    ) / (12 * (x1[3:-1, :] - x1[2:-2, :]))
    dudx[0:2, :] = (-3 * u_i[0:2, :] + 4 * u_i[1:3, :] - u_i[2:4, :]) / (
        2 * (x1[1:3, :] - x1[0:2, :])
    )
    dudx[-2:, :] = (3 * u_i[-2:, :] - 4 * u_i[-3:-1, :] + u_i[-4:-2, :]) / (
        2 * (x1[-2:, :] - x1[-3:-1, :])
    )

    dudy[:, 2:-2] = (
        -u_i[:, 4:] + 8 * u_i[:, 3:-1] - 8 * u_i[:, 1:-3] + u_i[:, 0:-4]
    ) / (12 * (x2[:, 3:-1] - x2[:, 2:-2]))
    dudy[:, 0:2] = (-3 * u_i[:, 0:2] + 4 * u_i[:, 1:3] - u_i[:, 2:4]) / (
        2 * (x2[:, 1:3] - x2[:, 0:2])
    )
    dudy[:, -2:] = (3 * u_i[:, -2:] - 4 * u_i[:, -3:-1] + u_i[:, -4:-2]) / (
        2 * (x2[:, -2:] - x2[:, -3:-1])
    )

    return dudx, dudy


def interpolate(
    coordinates: np.ndarray,
    data_to_interpolate: np.ndarray,
    grid_coordinates: Tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Perform linear interpolation of planar data onto a regular grid.

    The interpolation uses Delaunay triangulation.

    :param coordinates: A NumPy ndarray of shape (n, 2) representing the
        flattened spatial coordinates.
    :param data_to_interpolate: A NumPy ndarray of shape (n, )
        representing the flattened data to interpolate.
    :param grid_coordinates: A tuple of NumPy ndarrays of shape (m, n)
        representing a regular mesh grid.
    :raises RuntimeError: If the griddata interpolatio fails.
    :return: A NumPy ndarray of shape (m, n) representing the
        interpolated data.
    :rtype: numpy.ndarray
    """
    try:
        return spinterpolate.griddata(
            coordinates,
            data_to_interpolate,
            grid_coordinates,
            method="linear",
            rescale=True,
        )
    except Exception as e:
        raise RuntimeError(f"Griddata interpolation failed: {e}") from e


def cross(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Cross product of two vectors.

    :param vec1: First vector as NumPy ndarray of shape (3, ).
    :param vec2: Second vector as NumPy ndarray of shape (3, ).
    :return: Cross product as NumPy ndarray of shape (3, ).
    :rtype: numpy.ndarray
    """
    return np.array(
        [
            vec1[1] * vec2[2] - vec1[2] * vec2[1],
            vec1[2] * vec2[0] - vec1[0] * vec2[2],
            vec1[0] * vec2[1] - vec1[1] * vec2[0],
        ]
    )


def spatial_tensor_multiply(
    tensor_1: np.ndarray, tensor_2: np.ndarray
) -> np.ndarray:
    """Product of two planar datasets of a tensor quantity.

    :param tensor_1: A NumPy ndarray of shape (k, l, m, n), where k and
        l represent the dimensions of the tensor, and m and n are the
        number of points in each spatial direction of the 2D data field.
    :param tensor_2: A NumPy ndarray of shape (k, l, m, n), where k and
        l represent the dimensions of the tensor, and m and n are the
        number of points in each spatial direction of the 2D data field.
    :return: The tensor multiplication as a NumPy array of shape (k, l,
        m, n)
    """
    return (
        tensor_1.transpose(2, 3, 0, 1) @ tensor_2.transpose(2, 3, 0, 1)
    ).transpose(2, 3, 0, 1)
