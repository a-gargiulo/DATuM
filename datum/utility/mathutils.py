"""Define useful mathematical operations."""
from typing import Tuple

import numpy as np
import scipy.interpolate as spinterpolate


def calculate_derivative_1d(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Compute the first derivative of x2 with respect to x1.

    The algorithm operates on 1D data and uses a simple second-order centered finite
    difference approximation.
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
    x_1: np.ndarray, x_2: np.ndarray, u_i: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the first derivative of a vector quantity's component u:sub:`i` from a
    2-D dataset using a fourth-order centered finite difference approximation.

    :param x_1: NumPy ndarray of shape (n, ) containing spatial x:sub:`1` coordinates
        of the selected vector component from the 2-D dataset.
    :param x_2: NumPy ndarray of shape (n, ) containing spatial x:sub:`2` coordinates
        of the selected vector component from the 2-D dataset.
    :param u_i: NumPy ndarray of shape (n, ) containing the values of the selected
        vector component from the 2-D dataset.
    :return: A tuple of two NumPy ndarrays of shape (m, n), where m and n represent the
        number of available data points in the x:sub:`1`- and x:sub:`2`-direction. The
        two matrices represent the mean velocity gradient component in the
        x:sub:`1`- and x:sub:`2`-direction, respectively.
    """
    dudx = np.zeros_like(u_i)
    dudy = np.zeros_like(u_i)

    dudx[2:-2, :] = (
        -u_i[4:, :] + 8 * u_i[3:-1, :] - 8 * u_i[1:-3, :] + u_i[0:-4, :]
    ) / (12 * (x_1[3:-1, :] - x_1[2:-2, :]))
    dudx[0:2, :] = (-3 * u_i[0:2, :] + 4 * u_i[1:3, :] - u_i[2:4, :]) / (
        2 * (x_1[1:3, :] - x_1[0:2, :])
    )
    dudx[-2:, :] = (3 * u_i[-2:, :] - 4 * u_i[-3:-1, :] + u_i[-4:-2, :]) / (
        2 * (x_1[-2:, :] - x_1[-3:-1, :])
    )

    dudy[:, 2:-2] = (
        -u_i[:, 4:] + 8 * u_i[:, 3:-1] - 8 * u_i[:, 1:-3] + u_i[:, 0:-4]
    ) / (12 * (x_2[:, 3:-1] - x_2[:, 2:-2]))
    dudy[:, 0:2] = (-3 * u_i[:, 0:2] + 4 * u_i[:, 1:3] - u_i[:, 2:4]) / (
        2 * (x_2[:, 1:3] - x_2[:, 0:2])
    )
    dudy[:, -2:] = (3 * u_i[:, -2:] - 4 * u_i[:, -3:-1] + u_i[:, -4:-2]) / (
        2 * (x_2[:, -2:] - x_2[:, -3:-1])
    )

    return dudx, dudy


def interpolate(
    coordinates: np.ndarray,
    data_to_interpolate: np.ndarray,
    grid_coordinates: (Tuple[np.ndarray, np.ndarray]),
) -> np.ndarray:
    """Perform linear interpolation onto a regular grid using Delaunay triangulation,
    treating the original BeVERLI stereo PIV data as scattered data points.

    :param coordinates: A NumPy ndarray of shape (n, 2) representing the flattened
        spatial coordinates, where n represents the total number of available data
        points.
    :param data_to_interpolate: A NumPy ndarray of shape (n, ) representing the
        flattened quantity, where n represents the total number of available data
        points.
    :param grid_coordinates: A tuple of NumPy ndarrays of shape (m, n) representing a
        regular mesh grid, where m and n represent the number of total available data
        points in the x:sub:`1`- and x:sub:`2`-direction.
    :return: A NumPy ndarray of shape (m, n) representing the interpolated data, where
        m and n represent the number of total available data points in the
        x:sub:`1`- and x:sub:`2`-direction.
    """
    return spinterpolate.griddata(
        coordinates,
        data_to_interpolate,
        grid_coordinates,
        method="linear",
        rescale=True,
    )


def cross(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Cross product."""
    return np.array(
        [
            vec1[1] * vec2[2] - vec1[2] * vec2[1],
            vec1[2] * vec2[0] - vec1[0] * vec2[2],
            vec1[0] * vec2[1] - vec1[1] * vec2[0],
        ]
    )

def spatial_tensor_multiply(tensor_1: np.ndarray, tensor_2: np.ndarray) -> np.ndarray:
    """Multiply two 4D NumPy ndarrays containing a tensor quantity for each point in a
    2D data field.

    :param tensor_1: A NumPy ndarray of shape (k, l, m, n), where k and l represent the
        dimensions of the tensor, and m and n are the number of points in each spatial
        direction of the 2D data field.
    :param tensor_2: A NumPy ndarray of shape (k, l, m, n), where k and l represent the
        dimensions of the tensor, and m and n are the number of points in each spatial
        direction of the 2D data field.
    :return: The tensor multiplication as a NumPy array of shape (k, l, m, n)
    """
    return (tensor_1.transpose(2, 3, 0, 1) @ tensor_2.transpose(2, 3, 0, 1)).transpose(2, 3, 0, 1)
