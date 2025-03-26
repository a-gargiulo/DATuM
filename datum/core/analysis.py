"""This module contains a library of functions for the analysis of turbulent
constitutive relations using the BeVERLI stereo PIV data."""
from typing import Dict, Tuple

import numpy as np

from ..utility import mathutils

def calculate_eddy_viscosities(base_tensors: Dict[str, np.ndarray]) -> np.ndarray:
    """Calculate the eddy viscosity from the experimental data.

    :param base_tensors: A dictionary containing the base tensors relevant to the
        constitutive relations
    :return: The eddy viscosity as a NumPy ndarray of shape (n, n), where n represents
        the available number of data points on the interpolation grid.
    """
    r_mat = base_tensors["R"]
    s_mat = base_tensors["S"]

    tke = 0.5 * np.trace(r_mat, axis1=0, axis2=1)
    a_mat = -r_mat + (2.0 / 3.0) * tke[None, None, :, :] * np.eye(3)[:, :, None, None]

    nut = np.trace(
        mathutils.spatial_tensor_multiply(a_mat, s_mat.transpose(1, 0, 2, 3))
    ) / (2 * np.trace(mathutils.spatial_tensor_multiply(s_mat, s_mat.transpose(1, 0, 2, 3))))

    return nut


def get_base_tensors(piv_obj) -> Dict[str, np.ndarray]:
    """Obtain the base tensors for the analysis of the selected constitutive relations.

    :param piv_obj: An instance of the :py:class:`datum.piv.Piv` class, containing the
        BeVERLI stereo PIV data.
    :return: A dictionary containing the base tensors for the analysis of the selected
        constitutive relations.
    """
    base_tensors = {
        "S": (calculate_mean_strain_and_rotation_tensor(piv_obj))[0],
        "W": (calculate_mean_strain_and_rotation_tensor(piv_obj))[1],
        "O": (calculate_mean_strain_and_rotation_tensor(piv_obj))[2],
        "R": construct_reynolds_stress_tensor(piv_obj),
        "dV": construct_mean_velocity_gradient_tensor(piv_obj),
    }

    return base_tensors


def calculate_mean_strain_and_rotation_tensor(
    piv_obj,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the mean rate-of-strain and mean rotation tensors.

    :param piv_obj: An instance of the :py:class:`datum.piv.Piv` class, containing the
        BeVERLI stereo PIV data.
    :return: A tuple of NumPy arrays of shape (3, 3, n, n), where n corresponds to the
        number of mean velocity gradient component data points, containing the mean
        rate-of-strain, mean rotation, and normalized mean rotation tensors.
    """
    mean_strain = np.zeros(
        (3, 3, *piv_obj.data["mean_velocity_gradient"]["dUdX"].shape)
    )
    mean_rotation = np.zeros(
        (3, 3, *piv_obj.data["mean_velocity_gradient"]["dUdX"].shape)
    )
    mean_rotation_normalized = np.zeros(
        (3, 3, *piv_obj.data["mean_velocity_gradient"]["dUdX"].shape)
    )

    gradient_list = [
        (["dUdX", "dUdX"], [0, 0]),
        (["dVdY", "dVdY"], [1, 1]),
        (["dWdZ", "dWdZ"], [2, 2]),
        (["dUdY", "dVdX"], [0, 1]),
        (["dUdZ", "dWdX"], [0, 2]),
        (["dVdZ", "dWdY"], [1, 2]),
    ]

    for component, position in gradient_list:
        mean_strain[*position] = 0.5 * (
            piv_obj.data["mean_velocity_gradient"][component[0]]
            + piv_obj.data["mean_velocity_gradient"][component[1]]
        )
        mean_rotation[*position] = 0.5 * (
            piv_obj.data["mean_velocity_gradient"][component[0]]
            - piv_obj.data["mean_velocity_gradient"][component[1]]
        )

    # Symmetric/antisymmetric components
    mean_strain[1, 0] = mean_strain[0, 1]
    mean_strain[2, 0] = mean_strain[0, 2]
    mean_strain[2, 1] = mean_strain[1, 2]

    mean_rotation[1, 0] = -mean_rotation[0, 1]
    mean_rotation[2, 0] = -mean_rotation[0, 2]
    mean_rotation[2, 1] = -mean_rotation[1, 2]

    # Normalize mean rotation tensor
    sum_squared_mean_velocity_gradients = np.sum(
        [
            piv_obj.data["mean_velocity_gradient"][f"d{i}d{j}"] ** 2
            for i in ["U", "V", "W"]
            for j in ["X", "Y", "Z"]
        ],
        axis=0,
    )

    mean_rotation_normalized = (
        2
        * mean_rotation
        / np.sqrt(sum_squared_mean_velocity_gradients[None, None, :, :])
    )

    return mean_strain, mean_rotation, mean_rotation_normalized


def construct_reynolds_stress_tensor(piv_obj) -> np.ndarray:
    """Construct the Reynolds stress tensor.

    :param piv_obj: An instance of the :py:class:`datum.piv.Piv` class, containing the
        BeVERLI stereo PIV data.
    :return: A NumPy array of shape (3, 3, n, n), where n corresponds to the
        number of Reynolds stress component data points, representing the Reynolds
        stress tensor.
    """
    reynolds_stress = np.zeros((3, 3, *piv_obj.data["reynolds_stress"]["UU"].shape))

    stress_list = [
        ("UU", [0, 0]),
        ("VV", [1, 1]),
        ("WW", [2, 2]),
        ("UV", [0, 1]),
        ("UW", [0, 2]),
        ("VW", [1, 2]),
    ]

    for component, indices in stress_list:
        reynolds_stress[*indices] = piv_obj.data["reynolds_stress"][component]

    reynolds_stress[1, 0] = reynolds_stress[0, 1]
    reynolds_stress[2, 0] = reynolds_stress[0, 2]
    reynolds_stress[2, 1] = reynolds_stress[1, 2]

    return reynolds_stress


def construct_mean_velocity_gradient_tensor(piv_obj) -> np.ndarray:
    """Construct the mean velocity gradient tensor.

    :param piv_obj: An instance of the :py:class:`datum.piv.Piv` class, containing the
        BeVERLI stereo PIV data.
    :return: A NumPy array of shape (3, 3, n, n), where n corresponds to the number of
        mean velocity gradient component data points, representing the mean velocity
        gradient tensor.
    """
    mean_velocity_gradient = np.zeros(
        (3, 3, *piv_obj.data["mean_velocity_gradient"]["dUdX"].shape)
    )

    gradient_list = [
        ("dUdX", [0, 0]),
        ("dUdY", [0, 1]),
        ("dUdZ", [0, 2]),
        ("dVdX", [1, 0]),
        ("dVdY", [1, 1]),
        ("dVdZ", [1, 2]),
        ("dWdX", [2, 0]),
        ("dWdY", [2, 1]),
        ("dWdZ", [2, 2]),
    ]

    for component, indices in gradient_list:
        mean_velocity_gradient[*indices] = piv_obj.data["mean_velocity_gradient"][
            component
        ]

    return mean_velocity_gradient


def calculate_alignment_tensor(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Calculate the "cosine" angle between two model tensors, i.e., the left-hand-side
    and right-hand-side tensors of the analyzed constitutive relation.

    :param lhs: A NumPy ndarray of shape (3, 3, n, n), representing the left-hand-side
        tensor of the analyzed constitutive relation at each spatial point of the
        interpolation grid, where n is the number of grid points.
    :param rhs: A NumPy ndarray of shape (3, 3, n, n), representing the right-hand-side
        tensor of the analyzed constitutive relation at each spatial point of the
        interpolation grid, where n is the number of grid points.
    :return: A NumPy ndarray of shape (n, n) representing the cosine angle between the
        left-hand-side and the right-hand-side tensors of the analyzed constitutive
        relation at each spatial point of the interpolation grid.
    """
    # Reshaping
    lhs_reshaped = lhs.transpose(2, 3, 0, 1)
    rhs_reshaped = rhs.transpose(2, 3, 0, 1)

    # Calculate and return tensor
    return (
        np.abs(
            np.trace(
                lhs_reshaped @ np.transpose(rhs_reshaped, (0, 1, 3, 2)),
                axis1=2,
                axis2=3,
            )
        )
        / np.sqrt(
            np.trace(
                lhs_reshaped @ np.transpose(lhs_reshaped, (0, 1, 3, 2)),
                axis1=2,
                axis2=3,
            )
        )
        / np.sqrt(
            np.trace(
                rhs_reshaped @ np.transpose(rhs_reshaped, (0, 1, 3, 2)),
                axis1=2,
                axis2=3,
            )
        )
    )


def boussinesq_alignment(base_tensors: Dict[str, np.ndarray]) -> np.ndarray:
    """Perform the alignment analysis for the Boussinesq model.

    :param base_tensors: A dictionary containing the base tensors relevant to the
        constitutive relations
    :return: A NumPy ndarray of shape (n, n), where n represents the available number
        of data points on the interpolation grid.
    """
    r_mat = base_tensors["R"]
    s_mat = base_tensors["S"]

    tke = 0.5 * np.trace(r_mat, axis1=0, axis2=1)

    lhs = -r_mat + (2.0 / 3.0) * tke[None, None, :, :] * np.eye(3)[:, :, None, None]
    rhs = 2 * s_mat

    return calculate_alignment_tensor(lhs, rhs)


def qcr_alignment(base_tensors: Dict[str, np.ndarray], version: int) -> np.ndarray:
    """Perform the alignment analysis for the QCR model.

    :param base_tensors: A dictionary containing the base tensors relevant to the
        constitutive relations
    :param version: An integer representing the specific version of the QCR model to
        test.
    :return: A NumPy ndarray of shape (n, n), where n represents the available number
        of data points on the interpolation grid.
    """
    # Constants
    c_qcr_1 = 0.3
    c_qcr_2 = 2.5

    r_mat = base_tensors["R"]
    s_mat = base_tensors["S"]
    o_mat = base_tensors["O"]

    tke = 0.5 * np.trace(r_mat, axis1=0, axis2=1)

    lhs = np.zeros_like(r_mat)
    rhs = np.zeros_like(r_mat)
    if version == 2000:
        lhs = -r_mat + (2.0 / 3.0) * tke[None, None, :, :] * np.eye(3)[:, :, None, None]
        rhs = 2 * s_mat + 2 * c_qcr_1 * (
            mathutils.spatial_tensor_multiply(s_mat, o_mat)
            - mathutils.spatial_tensor_multiply(o_mat, s_mat)
        )
    elif version == 2013:
        lhs = -r_mat
        rhs = (
            2 * s_mat
            + 2
            * c_qcr_1
            * (
                mathutils.spatial_tensor_multiply(s_mat, o_mat)
                - mathutils.spatial_tensor_multiply(o_mat, s_mat)
            )
            - c_qcr_2
            * np.sqrt(
                2 * np.trace(mathutils.spatial_tensor_multiply(s_mat, s_mat), axis1=0, axis2=1)
            )[None, None, :, :]
            * np.eye(3)[:, :, None, None]
        )

    return calculate_alignment_tensor(lhs, rhs)


def gatski_and_speziale_alignment(
    base_tensors: Dict[str, np.ndarray],
    epsilon: np.ndarray,
    pressure_strain_model: str,
    model_order: int,
) -> np.ndarray:
    """Perform the alignment analysis for the Gatski and Speziale model.

    :param base_tensors: A dictionary containing the base tensors relevant to the
        constitutive relations
    :param epsilon: A NumPy ndarray of shape (m, n) representing the turbulence
        dissipation rate, where m and n are the number of available data points in the
        x:sub:`1` and x:sub:`2` direction, respectively.
    :param pressure_strain_model: A string indicating the pressure rate-of-strain model
        to employ.
    :param model_order: An integer number representing the order of non-linearity for
        the Gatski and Speziale model.
    :return: A NumPy ndarray of shape (n, n), where n represents the available number
        of data points on the interpolation grid.
    """
    r_mat = base_tensors["R"]
    s_mat = base_tensors["S"]
    w_mat = base_tensors["W"]
    dv_mat = base_tensors["dV"]
    tke_production = -np.trace(
        mathutils.spatial_tensor_multiply(r_mat, dv_mat.transpose(1, 0, 2, 3)),
        axis1=0,
        axis2=1,
    )
    tke = 0.5 * np.trace(r_mat, axis1=0, axis2=1)
    b_mat = (
        r_mat - (2.0 / 3.0) * tke[None, None, :, :] * np.eye(3)[:, :, None, None]
    ) / (2 * tke[None, None, :, :])

    production_to_dissipation_ratio = tke_production / epsilon  # n x n
    turbulence_time_scale = tke / epsilon  # n x n

    p_1_b = np.trace(
        mathutils.spatial_tensor_multiply(b_mat, b_mat.transpose(1, 0, 2, 3)),
        axis1=0,
        axis2=1,
    )  # n x n

    c_1 = c_2 = c_3 = c_4 = None
    if pressure_strain_model == "LRR":
        c_1 = 3
        c_2 = 0.8
        c_3 = 1.75
        c_4 = 1.31
    elif pressure_strain_model == "GL":
        c_1 = 3.6
        c_2 = 0.8
        c_3 = 1.2
        c_4 = 1.2
    elif pressure_strain_model == "SSG":
        c_1 = 3.4 + 1.8 * production_to_dissipation_ratio  # n x n
        c_2 = 0.8 - 1.3 * np.sqrt(p_1_b)  # n x n
        c_3 = 1.24
        c_4 = 0.4

    g = (0.5 * c_1 + production_to_dissipation_ratio - 1) ** (-1)  # n x n

    s_star = (
        0.5
        * g[None, None, :, :]
        * turbulence_time_scale[None, None, :, :]
        * (2 - c_3)
        * s_mat
    )

    w_star = (
        0.5
        * g[None, None, :, :]
        * turbulence_time_scale[None, None, :, :]
        * (2 - c_4)
        * w_mat
    )

    b_star = (
        (c_3 - 2) / (c_2[None, None, :, :] - (4 / 3)) * b_mat
        if pressure_strain_model == "SSG"
        else (c_3 - 2) / (c_2 - (4 / 3)) * b_mat
    )

    # Integrity Basis, 3, 3, n x n
    t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10 = construct_pope_integrity_basis(
        s_star, w_star
    )

    # eta, n x n
    eta1 = np.trace(mathutils.spatial_tensor_multiply(s_star, s_star), axis1=0, axis2=1)
    eta2 = np.trace(mathutils.spatial_tensor_multiply(w_star, w_star), axis1=0, axis2=1)
    eta3 = np.trace(
        mathutils.spatial_tensor_multiply(mathutils.spatial_tensor_multiply(s_star, s_star), s_star),
        axis1=0,
        axis2=1,
    )
    eta4 = np.trace(
        mathutils.spatial_tensor_multiply(s_star, mathutils.spatial_tensor_multiply(w_star, w_star)),
        axis1=0,
        axis2=1,
    )
    eta5 = np.trace(
        mathutils.spatial_tensor_multiply(
            mathutils.spatial_tensor_multiply(s_star, s_star),
            mathutils.spatial_tensor_multiply(w_star, w_star),
        ),
        axis1=0,
        axis2=1,
    )

    # D, n x n
    D = (
        3
        - (7 / 2) * eta1
        + eta1**2
        - (15 / 2) * eta2
        - 8 * eta1 * eta2
        + 3 * eta2**2
        - eta3
        + (2 / 3) * eta1 * eta3
        - 2 * eta2 * eta3
        + 21 * eta4
        + 24 * eta5
        + 2 * eta1 * eta4
        - 6 * eta2 * eta4
    )

    # g_1, n x n
    g_1 = -(1 / 2) * (6 - 3 * eta1 - 21 * eta2 - 2 * eta3 + 30 * eta4) / D

    lhs = b_star / g_1[None, None, :, :]
    rhs = t_1 - (
        6
        * (
            3
            * (
                threshold_map(5, model_order) * t_5
                + threshold_map(6, model_order) * t_6
                - threshold_map(7, model_order) * t_7
                - threshold_map(8, model_order) * t_8
                - 2 * threshold_map(9, model_order) * t_9
                + threshold_map(4, model_order) * t_4 * eta1[None, None, :, :]
            )
            + threshold_map(2, model_order)
            * t_2
            * (1 + eta1[None, None, :, :] - 2 * eta2[None, None, :, :])
            + threshold_map(3, model_order)
            * t_3
            * (-2 + eta1[None, None, :, :] + 4 * eta2[None, None, :, :])
        )
        + 4
        * (
            threshold_map(2, model_order) * t_2
            + threshold_map(3, model_order) * t_3
            + 3 * threshold_map(4, model_order) * t_4
        )
        * eta3[None, None, :, :]
        + 12
        * (
            threshold_map(2, model_order) * t_2
            + threshold_map(3, model_order) * t_3
            + 3 * threshold_map(4, model_order) * t_4
        )
        * eta4[None, None, :, :]
    ) / (
        -6
        + 3 * eta1[None, None, :, :]
        + 21 * eta2[None, None, :, :]
        + 2 * eta3[None, None, :, :]
        - 30 * eta4[None, None, :, :]
    )

    return calculate_alignment_tensor(lhs, rhs)


def construct_pope_integrity_basis(
    s_star: np.ndarray, w_star: np.ndarray
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """

    :param s_star: The normalized mean rate-of-strain as a NumPy ndarray of shape
        (3, 3, n, n), where n represents the available number of data points in
        of the interpolation grid.
    :param w_star: The normalized mean rotation tensor as a NumPy ndarray of shape
        (3, 3, n, n), where n represents the available number of data points in
        of the interpolation grid.
    :return: A tuple with containing the ten integrity basis tensors according to
        Pope (2000).
    """
    t_1 = s_star

    t_2 = mathutils.spatial_tensor_multiply(s_star, w_star) - spatial_tensor_multiply(
        w_star, s_star
    )

    t_3 = (
        mathutils.spatial_tensor_multiply(s_star, s_star)
        - (1 / 3)
        * np.trace(mathutils.spatial_tensor_multiply(s_star, s_star), axis1=0, axis2=1)[
            None, None, :, :
        ]
        * np.eye(3)[:, :, None, None]
    )

    t_4 = (
        mathutils.spatial_tensor_multiply(w_star, w_star)
        - (1 / 3)
        * np.trace(mathutils.spatial_tensor_multiply(w_star, w_star), axis1=0, axis2=1)[
            None, None, :, :
        ]
        * np.eye(3)[:, :, None, None]
    )

    t_5 = mathutils.spatial_tensor_multiply(
        w_star, mathutils.spatial_tensor_multiply(s_star, s_star)
    ) - mathutils.spatial_tensor_multiply(mathutils.spatial_tensor_multiply(s_star, s_star), w_star)

    t_6 = (
        mathutils.spatial_tensor_multiply(mathutils.spatial_tensor_multiply(w_star, w_star), s_star)
        + mathutils.spatial_tensor_multiply(s_star, mathutils.spatial_tensor_multiply(w_star, w_star))
        - (2 / 3)
        * np.trace(
            mathutils.spatial_tensor_multiply(s_star, mathutils.spatial_tensor_multiply(w_star, w_star)),
            axis1=0,
            axis2=1,
        )[None, None, :, :]
        * np.eye(3)[:, :, None, None]
    )

    t_7 = mathutils.spatial_tensor_multiply(
        mathutils.spatial_tensor_multiply(w_star, s_star), mathutils.spatial_tensor_multiply(w_star, w_star)
    ) - mathutils.spatial_tensor_multiply(
        mathutils.spatial_tensor_multiply(w_star, w_star), mathutils.spatial_tensor_multiply(s_star, w_star)
    )

    t_8 = mathutils.spatial_tensor_multiply(
        mathutils.spatial_tensor_multiply(s_star, w_star), mathutils.spatial_tensor_multiply(s_star, s_star)
    ) - mathutils.spatial_tensor_multiply(
        mathutils.spatial_tensor_multiply(s_star, s_star), mathutils.spatial_tensor_multiply(w_star, s_star)
    )

    t_9 = (
        mathutils.spatial_tensor_multiply(
            mathutils.spatial_tensor_multiply(w_star, w_star),
            mathutils.spatial_tensor_multiply(s_star, s_star),
        )
        + mathutils.spatial_tensor_multiply(
            mathutils.spatial_tensor_multiply(s_star, s_star),
            mathutils.spatial_tensor_multiply(w_star, w_star),
        )
        - (2 / 3)
        * np.trace(
            mathutils.spatial_tensor_multiply(
                mathutils.spatial_tensor_multiply(s_star, s_star),
                mathutils.spatial_tensor_multiply(w_star, w_star),
            ),
            axis1=0,
            axis2=1,
        )[None, None, :, :]
        * np.eye(3)[:, :, None, None]
    )

    t_10 = mathutils.spatial_tensor_multiply(
        w_star,
        mathutils.spatial_tensor_multiply(
            mathutils.spatial_tensor_multiply(s_star, s_star),
            mathutils.spatial_tensor_multiply(w_star, w_star),
        ),
    ) - mathutils.spatial_tensor_multiply(
        mathutils.spatial_tensor_multiply(
            mathutils.spatial_tensor_multiply(w_star, w_star),
            mathutils.spatial_tensor_multiply(s_star, s_star),
        ),
        w_star,
    )

    return t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10


def threshold_map(rank: int, model_order: int) -> int:
    """Map for thresholding

    :param rank: The rank of the tensor
    :param model_order: The model order
    :return: The mapping value
    """
    if rank <= model_order:
        return 1
    return 0
