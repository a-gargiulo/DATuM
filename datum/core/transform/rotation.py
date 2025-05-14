"""Transformation sub-module for rotation operations."""
from typing import TYPE_CHECKING, Literal, List
import numpy as np

if TYPE_CHECKING:
    from ..piv import Piv

Axis = Literal["x", "y", "z"]


DAT2ROT = [
    ("coordinates", ["X", "Y"]),
    ("mean_velocity", ["U", "V", "W"]),
    ("reynolds_stress", ["UU", "VV", "WW", "UV", "UW", "VW"]),
    ("velocity_snapshot", ["U", "V", "W"]),
]


def get_rotation_matrix(phi_deg: float, axis: Axis) -> np.ndarray:
    """Get Euler rotation matrix about a specified Cartesian axis.

    The matrix represents a body's Euler rotation about the specified axis of
    the body's Cartesian coordinate system.

    :param phi_deg: Rotation angle, deg.
    :param axis: Identifier string for the Cartesian rotation axis.
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


def retrieve_rotation_matrix_from_piv(piv: "Piv") -> np.ndarray:
    """Obtain rotation matrix from the PIV plane data.

    :param piv: PIV plane data.

    :return: Rotation matrix as :py:type:`np.ndarray` of shape (3, 3).
    """
    if piv.pose.angle2 != 0.0:
        phi1_deg = piv.pose.angle1
        phi2_deg = piv.pose.angle2
        rotmat1 = get_rotation_matrix(phi1_deg, "z")
        rotmat2 = get_rotation_matrix(-phi2_deg, "y")
        rotmat = rotmat2 @ rotmat1
    else:
        phi_deg = piv.pose.angle1
        rotmat = get_rotation_matrix(phi_deg, "z")

    return rotmat


############
# GENERICS #
############
def rotate_all(piv: "Piv"):
    """Rotate all PIV data from local to global coordinates."""
    rotation_matrix = retrieve_rotation_matrix_from_piv(piv)

    for quantity, components in DAT2ROT:
        if piv.data[quantity] is not None:
            rotate_planar(piv, quantity, components, rotation_matrix)


def rotate_planar(
    piv: "Piv",
    quantity: str,
    components: List[str],
    rotation_matrix: np.ndarray
):
    """Rotate a specified BeVERLI Hill stereo PIV flow quantity.

    :param piv: PIV plane data.
    :param quantity: Flow quantity identifier.
    :param components: A list of strings specifying the components of the
        vector or tensor quantity to rotate.
    :param rotation_matrix: The rotation matrix used to rotate the data as
        :py:type:`np.ndarray` of shape (3, 3).
    """
    if len(components) >= 6:
        tensor = prepare_tensor_planar(piv, quantity, components)
        rotated_tensor = rotate_tensor_planar(tensor, rotation_matrix)
        set_rotated_tensor_planar(
            piv, quantity, components, rotated_tensor
        )
    else:
        vector = prepare_vector_planar(piv, quantity, components)
        rotated_vector = rotate_vector_planar(vector, rotation_matrix)
        set_rotated_vector_planar(piv, quantity, components, rotated_vector)


###########
# VECTORS #
###########
def prepare_vector_planar(
    piv: "Piv", quantity: str, components: List[str]
) -> np.ndarray:
    """Build vector of a desired quantity from the corresponding components.

    :param piv: Instance of the class Piv.
    :param quantity: String specifying the vector quantity.
    :param components: List of strings specifying the vector components.
    :return: List of NumPy ndarrays of shape (m, n), each containing a specific
        vector component, where m and n represent the number of available data
        points in the x:sub:`1`- and x:sub:`2`-direction.
    """
    vector = np.array(
        [piv.data[quantity][component] for component in components]
    )

    return vector


def rotate_vector_planar(
    vector: np.ndarray, rotation_matrix: np.ndarray
) -> np.ndarray:
    """Rotate a 2d plane of 2d or 3d vectors.

    :param rotation_matrix: The rotation matrix as NumPy ndarray of
        shape (3, 3).
    :param vector: NumPy ndarray of shape (dim, m, n), where dim is the vector
        dimension (or number of vector components), and m and n represent the
        number of available data points in each dimension of the 2d data plane.

    :return: NumPy ndarray of shape (dim, m, n) representing the rotated data
        plane, where dim is the number of vector components, and m and n
        represent the number of available data points in each dimension of the
        2d data plane.
    """
    if vector.shape[0] < 2:
        raise TypeError("The vector must be at least 2d.")

    if vector.shape[0] == 2:
        vector = np.append(
            vector, np.zeros_like(vector[0])[None, :, :], axis=0
        )

    rotated_vector = (
        rotation_matrix[None, None, :, :]
        @ vector.transpose(1, 2, 0)[:, :, :, None]
    ).transpose(2, 0, 1, 3)[..., 0]
    return rotated_vector


def set_rotated_vector_planar(
    piv: "Piv",
    quantity: str,
    components: List[str],
    rotated_vector: np.ndarray
):
    """Update the BeVERLI stereo PIV data with the rotated vector.

    :param piv_obj: Instance of the :py:class:`datum.piv.Piv` class.
    :param quantity: String specifying the vector quantity.
    :param components: List of strings specifying the vector components.
    :param rotated_vector: List of NumPy ndarrays of shape (m, n), each
        representing a vector component, where m and n represent the number of
        available data points in the x:sub:`1`- and x:sub:`2`-direction.
    """
    for component, rotated_vector_component in zip(components, rotated_vector):
        piv.data[quantity][component] = rotated_vector_component


###########
# TENSORS #
###########
def prepare_tensor_planar(
    piv: "Piv", quantity: str, components: List[str]
) -> np.ndarray:
    """Construct tensor in appropriate form from planar PIV data for rotation.

    :param piv: PIV plane data.
    :param quantity: Tensor quantity identifier.
    :param components: Tensor componenets.
    :return: :py:type:`np.ndarray` of shape (3, 3, m, n), representing
        a 3 by 3 tensor, where each component is a planar PIV data field with
        dimensions m by n.
    """
    if len(components) == 6:
        n_diag = 3
        m = piv.data[quantity][components[0]].shape[0]
        n = piv.data[quantity][components[0]].shape[1]
        tensor = np.zeros((n_diag, n_diag, m, n))

        for i in range(n_diag):
            tensor[i, i] = piv.data[quantity][components[i]]

        k = n_diag
        for i in range(n_diag):
            for j in range(i + 1, n_diag):
                tensor[i, j] = tensor[j, i] = piv.data[quantity][components[k]]
                k += 1
    else:
        tensor = np.array(
            [
                [
                    piv.data[quantity][component]
                    for component in components[i : i + 3]
                ]
                for i in range(0, len(components), 3)
            ]
        )

    return tensor


def rotate_tensor_planar(
    tensor: np.ndarray, rotation_matrix: np.ndarray
) -> np.ndarray:
    """Rotate a planar tensor quantity.

    :param rotation_matrix: Rotation matrix as :py:type:`np.ndarray of shape
        (3, 3).
    :param tensor: Planar tensor as :py:type:`np.ndarray` of shape
        (3, 3, m, n).

    :return: Rotated planar tensor as :py:type:`np.ndarray` of shape
        (3, 3, m, n).
    """
    rotated_tensor = (
        rotation_matrix[None, None, :, :]
        @ tensor.transpose(2, 3, 0, 1)
        @ rotation_matrix.T[None, None, :, :]
    ).transpose(2, 3, 0, 1)

    return rotated_tensor


def set_rotated_tensor_planar(
    piv: "Piv",
    quantity: str,
    components: List[str],
    rotated_tensor: np.ndarray
):
    """Update the BeVERLI stereo PIV data with the rotated tensor.

    :param piv: PIV data plane.
    :param quantity: String specifying the tensor quantity.
    :param components: List of strings specifying the tensor components.
    :param rotated_tensor: Rotated tensor as :py:type:`np.ndarray` of shape
        (3, 3, m, n)
    """
    processed_keys = set()
    for component, rotated_tensor_component in zip(
        components, (i for sublist in rotated_tensor for i in sublist)
    ):
        if component not in processed_keys:
            piv.data[quantity][component] = rotated_tensor_component
            processed_keys.add(component)
        else:
            continue


##########
# BACKUP #
##########
# def rotate_profile(
#     profile: Dict[str, Dict[str, Union[np.ndarray, float, int]]],
#     rotation_matrix: np.ndarray,
# ) -> Tuple[
#     np.ndarray,
#     np.ndarray,
#     Optional[np.ndarray],
#     Optional[np.ndarray],
#     Optional[np.ndarray],
# ]:
#     """
#     Rotate profile data.

#     :param profile: The profile data.
#     :param rotation_matrix: The rotation matrix.

#     :return: A tuple containing the rotated quantities.
#     :rtype: Tuple[
#         np.ndarray,
#         np.ndarray,
#         Optional[np.ndarray],
#         Optional[np.ndarray],
#         Optional[np.ndarray]
#     ]
#     """
#     velocity_vector = np.array(
#         [profile["mean_velocity"][f"{i}"] for i in "UVW"]
#     )

#     components = ["UU", "UV", "UW", "UV", "VV", "VW", "UW", "VW", "WW"]
#     re_stress_tensor = np.array(
#         [
#             [
#                 profile["reynolds_stress"][component]
#                 for component in components[i : i + 3]
#             ]
#             for i in range(0, 9, 3)
#         ]
#     )

#     velocity_vector_rotated = rotation_matrix @ velocity_vector
#     re_stress_tensor_rotated = rotate_tensor_profile(
#         re_stress_tensor, rotation_matrix
#     )

#     if "strain_tensor" in profile:
#         strain_tensor = np.array(
#             [
#                 [profile["strain_tensor"][f"S_{i+1}{j+1}"] for j in range(3)]
#                 for i in range(3)
#             ]
#         )
#         rotation_tensor = np.array(
#             [
#                 [profile["rotation_tensor"][f"W_{i+1}{j+1}"] for j in range(3)]
#                 for i in range(3)
#             ]
#         )
#         normalized_rotation_tensor = np.array(
#             [
#                 [
#                     profile["normalized_rotation_tensor"][f"O_{i+1}{j+1}"]
#                     for j in range(3)
#                 ]
#                 for i in range(3)
#             ]
#         )

#         strain_tensor_rotated = rotate_tensor_profile(
#             strain_tensor, rotation_matrix
#         )
#         rotation_tensor_rotated = rotate_tensor_profile(
#             rotation_tensor, rotation_matrix
#         )
#         normalized_rotation_tensor_rotated = rotate_tensor_profile(
#             normalized_rotation_tensor, rotation_matrix
#         )

#         return (
#             velocity_vector_rotated,
#             re_stress_tensor_rotated,
#             strain_tensor_rotated,
#             rotation_tensor_rotated,
#             normalized_rotation_tensor_rotated,
#         )

#     return velocity_vector_rotated, re_stress_tensor_rotated, None, None, None


# def rotate_tensor_profile(
#     tensor: np.ndarray, rotation_matrix: np.ndarray
# ) -> np.ndarray:
#     """
#     Rotate the profile of a tensor quantity.

#     :param tensor: The tensor quantity as a (3, 3, N) array.
#     :param rotation_matrix: The rotation matrix.

#     :return: The rotated tensor profile.
#     :rtype: np.ndarray
#     """
#     return (
#         rotation_matrix[None, :, :]
#         @ tensor.transpose(2, 0, 1)
#         @ rotation_matrix.T[None, :, :]
#     ).transpose(1, 2, 0)


# def set_rotated_profiles(
#     profile: Dict[str, Dict[str, Union[np.ndarray, float, int]]],
#     velocity_vector_rotated: np.ndarray,
#     re_stress_tensor_rotated: np.ndarray,
#     strain_tensor_rotated: Optional[np.ndarray] = None,
#     rotation_tensor_rotated: Optional[np.ndarray] = None,
#     normalized_rotation_tensor_rotated: Optional[np.ndarray] = None,
# ):
#     """
#     Set the rotated profile.

#     :param profile: The profile data.
#     :param velocity_vector_rotated: The rotated velocity data.
#     :param re_stress_tensor_rotated: The rotated reynolds stress data.
#     :param strain_tensor_rotated: The rotated strain tensor data.
#     :param rotation_tensor_rotated: The rotated rotation tensor data.
#     :param normalized_rotation_tensor_rotated: The rotated normalized rotation tensor data.
#     """
#     for index, component in enumerate(["U_SS", "V_SS", "W_SS"]):
#         profile["mean_velocity"][component] = velocity_vector_rotated[index, :]

#     for index, component in enumerate(["UU_SS", "VV_SS", "WW_SS"]):
#         profile["reynolds_stress"][component] = re_stress_tensor_rotated[
#             index, index, :
#         ]

#     for indices, component in zip(
#         [(0, 1), (0, 2), (1, 2)], ["UV_SS", "UW_SS", "VW_SS"]
#     ):
#         profile["reynolds_stress"][component] = re_stress_tensor_rotated[
#             *indices, :
#         ]

#     if (
#         (strain_tensor_rotated is not None)
#         and (rotation_tensor_rotated is not None)
#         and (normalized_rotation_tensor_rotated is not None)
#     ):
#         indices_list = [(i, j) for i in range(3) for j in range(3)]
#         strain_tensor_components = [
#             f"S_{i+1}{j+1}_SS" for i in range(3) for j in range(3)
#         ]
#         rotation_tensor_components = [
#             f"W_{i + 1}{j + 1}_SS" for i in range(3) for j in range(3)
#         ]
#         normalized_rotation_tensor_components = [
#             f"O_{i + 1}{j + 1}_SS" for i in range(3) for j in range(3)
#         ]

#         for indices, s_component, w_component, o_component in zip(
#             indices_list,
#             strain_tensor_components,
#             rotation_tensor_components,
#             normalized_rotation_tensor_components,
#         ):
#             profile["strain_tensor"][s_component] = strain_tensor_rotated[
#                 *indices, :
#             ]
#             profile["rotation_tensor"][w_component] = rotation_tensor_rotated[
#                 *indices, :
#             ]
#             profile["normalized_rotation_tensor"][o_component] = (
#                 rotation_tensor_rotated[*indices, :]
#             )
