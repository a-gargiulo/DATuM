"""Provides functions for transforming the BeVERLI Hill stereo PIV data from the local
PIV coordinate system to the global Cartesian coordinate system of the corresponding
BeVERLI Hill experiment in the Virginia Tech Stability Wind Tunnel."""

from typing import Tuple, List, Dict, Union, Optional

import numpy as np
import sys

from ..utility import mathutils
# from .parser import InputFile
from .piv import Piv


def get_rotation_matrix(rotation_angle_deg: float, rotation_axis: Tuple[float, float, float]) -> np.ndarray:
    """Get the rotation matrix for a body's Euler rotation about a specified axis of its Cartesian coordinate system.

    :param rotation_angle_deg: Rotation angle measured in degrees.
    :param rotation_axis: 3D vector components of the axis of rotation as a tuple of shape (3, ).

    :return: Rotation matrix as :py:type:`ndarray` of shape (3, 3).
    """
    angle_rad = np.deg2rad(rotation_angle_deg)
    axis = calculate_unit_vector(rotation_axis)

    cosa = np.cos(angle_rad)
    sina = np.sin(angle_rad)

    # rotation matrix around unit vector
    matrix = np.diag([cosa, cosa, cosa])
    matrix += np.outer(axis, axis) * (1.0 - cosa)
    axis = axis * sina
    matrix += np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )

    return matrix


def calculate_unit_vector(data: Tuple[float, float, float]) -> np.ndarray:
    """Normalize a 3D vector.

    :param data: 3D vector components.

    :return: The normalized vector as :py:type:`ndarray` of shape (3, ).
    """
    np_data = np.array(data, dtype=np.float64)
    np_data /= np.sqrt(np.dot(np_data, np_data))
    return np_data


def rotate_data(piv_obj: Piv) -> None:
    """Rotate PIV data from local PIV coordinates to global coordinates in VT SWT."""
    rotation_matrix = _obtain_rotation_matrix(piv_obj)

    data_to_rotate = [
        ("coordinates", ["X", "Y"]),
        ("mean_velocity", ["U", "V", "W"]),
        ("reynolds_stress", ["UU", "UV", "UW", "UV", "VV", "VW", "UW", "VW", "WW"]),
        ("instantaneous_velocity_frame", ["U", "V", "W"]),
    ]

    # Rotate data
    for quantity, components in data_to_rotate:
        is_avail = piv_obj.search(quantity)

        if is_avail:
            rotate_flow_quantity(piv_obj, quantity, components, rotation_matrix)


def _obtain_rotation_matrix(piv_obj: Piv):
    if piv_obj.pose.angle2 != 0.0:
        rotation_angle_1_deg = piv_obj.pose.angle1
        rotation_angle_2_deg = piv_obj.pose.angle2
        rotation_matrix_1 = get_rotation_matrix(rotation_angle_1_deg, (0, 0, 1))
        rotation_matrix_2 = get_rotation_matrix(rotation_angle_2_deg, (0, -1, 0))
        rotation_matrix = rotation_matrix_2 @ rotation_matrix_1
    else:
        rotation_angle_deg = piv_obj.pose.angle1
        rotation_matrix = get_rotation_matrix(rotation_angle_deg, (0, 0, 1))

    return rotation_matrix


def rotate_flow_quantity(
    piv_obj, quantity: str, components: List[str], rotation_matrix: np.ndarray
) -> None:
    """Rotates a specified BeVERLI Hill stereo PIV flow quantity using a specified
    rotation matrix.

    :param piv_obj: Object containing the BeVERLI Hill stereo PIV data.
    :param quantity: String specifying the flow quantity to rotate.
    :param components: A list of strings specifying the components of the vector or
        tensor quantity to rotate.
    :param rotation_matrix: The rotation matrix used to rotate the data as
        :py:type:`ndarray` of shape (3, 3).

    This routine directly edits the :py:type:`Piv` object that is passed to it.
    """
    if len(components) == 9:
        tensor = prepare_tensor_quantity_for_rotation(piv_obj, quantity, components)
        rotated_tensor = rotate_tensor_quantity(tensor, rotation_matrix)
        set_rotated_tensor_quantity(piv_obj, quantity, components, rotated_tensor)
    else:
        vector = prepare_vector_quantity_for_rotation(piv_obj, quantity, components)
        rotated_vector = rotate_planar_vector_field(vector, rotation_matrix)
        set_rotated_vector_quantity(piv_obj, quantity, components, rotated_vector)


def rotate_planar_vector_field(vector: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """Rotate a 2d plane of 2d or 3d vectors.

    :param rotation_matrix: The rotation matrix as NumPy ndarray of shape (3, 3).
    :param vector: NumPy ndarray of shape (dim, m, n), where dim is the vector dimension (or number of vector
        components), and m and n represent the number of available data points in each dimension of the 2d data plane.

    :return: NumPy ndarray of shape (dim, m, n) representing the rotated data plane, where dim is the number of vector
        components, and m and n represent the number of available data points in each dimension of the 2d data plane.
    """
    if vector.shape[0] < 2:
        raise TypeError("The vector must be at least 2d.")

    if vector.shape[0] == 2:
        vector = np.append(vector, np.zeros_like(vector[0])[None, :, :], axis=0)

    rotated_vector = (
        rotation_matrix[None, None, :, :] @ vector.transpose(1, 2, 0)[:, :, :, None]
    ).transpose(2, 0, 1, 3)[..., 0]
    return rotated_vector


def rotate_tensor_quantity(
    tensor: np.ndarray, rotation_matrix: np.ndarray
) -> np.ndarray:
    """Rotate a BeVERLI stereo PIV tensor quantity based on the provided rotation
    matrix.

    This method takes a rotation matrix and a list of tensor component matrices as
    input. It then calculates the rotated quantity and returns it.

    :param rotation_matrix: NumPy ndarray of shape (3, 3) representing the rotation
        matrix.
    :param tensor: List of lists of NumPy ndarrays of shape (m, n), each containing
        a specific vector component, where m and n represent the number of available data
        points in the x:sub:`1`- and x:sub:`2`-direction.
    :return: List of lists of NumPy ndarrays of shape (m, n) containing the rotated
        vector quantity's components, where m and n represent the number of available
        data points in the x:sub:`1`- and x:sub:`2`-direction.
    """
    rotated_tensor = (
        rotation_matrix[None, None, :, :]
        @ tensor.transpose(2, 3, 0, 1)
        @ rotation_matrix.T[None, None, :, :]
    ).transpose(2, 3, 0, 1)

    return rotated_tensor


def prepare_vector_quantity_for_rotation(
    piv_obj, quantity: str, components: List[str]
) -> np.ndarray:
    """Build the vector of a desired quantity from the corresponding components of the
    BeVERLI stereo PIV data.

    :param piv_obj: Instance of the class :py:class:`datum.piv.Piv` class.
    :param quantity: String specifying the vector quantity.
    :param components: List of strings specifying the vector components.
    :return: List of NumPy ndarrays of shape (m, n), each containing a specific vector
        component, where m and n represent the number of available data points in the
        x:sub:`1`- and x:sub:`2`-direction.
    """
    vector = np.array([piv_obj.data[quantity][component] for component in components])

    return vector


def prepare_tensor_quantity_for_rotation(
    piv_obj, quantity: str, components: List[str]
) -> np.ndarray:
    """Build the tensor of a desired quantity from the corresponding components of the
    BeVERLI stereo PIV data.

    :param piv_obj: Instance of the :py:class:`datum.piv.Piv` class.
    :param quantity: String specifying the tensor quantity.
    :param components: List of strings specifying the tensor components.
    :return: List of lists of NumPy ndarrays of shape (m, n), containing specific
        tensor components, where m and n represent the number of available data
        points in the x:sub:`1`- and x:sub:`2`-direction.
    """
    tensor = np.array(
        [
            [piv_obj.data[quantity][component] for component in components[i : i + 3]]
            for i in range(0, len(components), 3)
        ]
    )
    return tensor


def set_rotated_vector_quantity(
    piv_obj, quantity: str, components: List[str], rotated_vector: np.ndarray
) -> None:
    """Update the BeVERLI stereo PIV data with the rotated vector.

    :param piv_obj: Instance of the :py:class:`datum.piv.Piv` class.
    :param quantity: String specifying the vector quantity.
    :param components: List of strings specifying the vector components.
    :param rotated_vector: List of NumPy ndarrays of shape (m, n), each representing a
        vector component, where m and n represent the number of available data points
        in the x:sub:`1`- and x:sub:`2`-direction.
    """
    for component, rotated_vector_component in zip(components, rotated_vector):
        piv_obj.data[quantity][component] = rotated_vector_component


def set_rotated_tensor_quantity(
    piv_obj, quantity: str, components: List[str], rotated_tensor: np.ndarray
) -> None:
    """Update the BeVERLI stereo PIV data with the rotated tensor.

    :param piv_obj: Instance of the :py:class:`datum.piv.Piv` class.
    :param quantity: String specifying the tensor quantity.
    :param components: List of strings specifying the tensor components.
    :param rotated_tensor: List of lists of NumPy ndarrays of shape (m, n). Each array
        represents a tensor component, where m and n represent the number of available
        data points in the x:sub:`1`- and x:sub:`2`-direction.
    """
    processed_keys = set()
    for component, rotated_tensor_component in zip(
        components, (i for sublist in rotated_tensor for i in sublist)
    ):
        if component not in processed_keys:
            piv_obj.data[quantity][component] = rotated_tensor_component
            processed_keys.add(component)
        else:
            continue


def interpolate_data(piv_obj, n_grid) -> None:
    """Perform a comprehensive routine to interpolate the BeVERLI stereo PIV data to a
    dense grid within the global BeVERLI coordinate system.

    :param piv_obj: Instance of the :py:class:`datum.piv.Piv` class.
    """
    x1_q, x2_q = get_interpolation_grid(piv_obj, n_grid)

    data_to_interpolate = [
        ("mean_velocity", ["U", "V", "W"]),
        ("reynolds_stress", ["UU", "VV", "WW", "UV", "UW", "VW"]),
        ("turbulence_scales", ["TKE", "epsilon"]),
        ("instantaneous_velocity_frame", ["U", "V", "W"]),
    ]

    coords = get_flattened_coordinates(piv_obj)
    for data_type, keys in data_to_interpolate:
        for key in keys:
            is_avail = all([piv_obj.search(data_type), piv_obj.search(key)])
            if is_avail:
                data = get_flattened_data(piv_obj, data_type, key)
                interp_data = mathutils.interpolate(coords, data, (x1_q, x2_q))
                set_interpolated_data(piv_obj, data_type, key, interp_data)

    piv_obj.data["coordinates"] = {"X": x1_q, "Y": x2_q}


def get_interpolation_grid(piv_obj, n_grid) -> Tuple[np.ndarray, np.ndarray]:
    """Build a structured, dense coordinate mesh for the interpolation of the BeVERLI
    stereo PIV data.

    :param piv_obj: Instance of the :py:class:`datum.piv.Piv` class.
    :return: A tuple containing NumPy ndarrays of shape (m, n) that represent a mesh
        grid, where m and n represent the number of available data points in the
        x:sub:`1`- and x:sub:`2`-direction.
    """
    num_of_pts = n_grid
    x1_range = np.linspace(
        np.min(piv_obj.data["coordinates"]["X"]),
        np.max(piv_obj.data["coordinates"]["X"]),
        num_of_pts,
    )
    x2_range = np.linspace(
        np.min(piv_obj.data["coordinates"]["Y"]),
        np.max(piv_obj.data["coordinates"]["Y"]),
        num_of_pts,
    )
    x1_q, x2_q = [matrix.T for matrix in np.meshgrid(x1_range, x2_range)]

    return x1_q, x2_q


def get_flattened_coordinates(piv_obj) -> np.ndarray:
    """Retrieve an array of flattened spatial coordinates from the BeVERLI stereo PIV
    data, necessary for subsequent interpolation.

    :param piv_obj: Instance of the :py:class:`datum.piv.Piv` class.
    :return: NumPy ndarray of shape (m, 2) containing the flattened set of spatial
        coordinates for the desired quantity from the stereo PIV data, where m
        represents the number of total available data points.
    """
    flattened_coordinates = np.column_stack(
        (
            piv_obj.data["coordinates"]["X"].flatten(),
            piv_obj.data["coordinates"]["Y"].flatten(),
        )
    )

    return flattened_coordinates


def get_flattened_data(piv_obj, data_type: str, key: str) -> np.ndarray:
    """Retrieve an array of flattened data of a specific vector or tensor quantity from
    the stereo PIV data, necessary for subsequent interpolation.

    :param piv_obj: Instance of the :py:class:`datum.piv.Piv` class.
    :param data_type: A string specifying the desired quantity to interpolate.
    :param key: A string indicating the desired vector or tensor component.
    :return: NumPy ndarrays of shape (n, ) containing the flattened set of values for
        the desired quantity's component from the stereo PIV data, where n represents
        the number of total available data points.
    """
    return piv_obj.data[data_type][key].flatten()


def set_interpolated_data(
    piv_obj, data_type: str, key: str, interpolated_data: np.ndarray
) -> None:
    """Update the PIV data with the interpolated data.

    :param piv_obj: Instance of the :py:class:`datum.piv.Piv` class.
    :param data_type: A string specifying the particular vector or tensor quantity to
        update.
    :param key: A string specifying the particular component of the vector or tensor
        quantity to update.
    :param interpolated_data: NumPy ndarrays of shape (n, ) representing the
        interpolated component, where n represents the number of total
        available data points.
    """
    piv_obj.data[data_type][key] = interpolated_data


def translate_data(piv_obj) -> None:
    """Perform a comprehensive routine to translate the BeVERLI stereo PIV data from
    its local to the global BeVERLI coordinate system."""
    x_1_shift, x_2_shift = get_translation_vector(piv_obj)

    translate_coordinates(piv_obj.data["coordinates"], x_1_shift, x_2_shift)


def get_translation_vector(piv_obj) -> Tuple[float, float]:
    """Obtain the translation vector from the available coordinate transformation
    parameters.

    :param piv_obj: Instance of the :py:class:`datum.piv.Piv` class.
    :return: A tuple of floats, representing the components of the translation vector.
    """
    # THIS FUNCTION IS QUITE SPECIFIC TO THE BEVERLI PIV DATA, WHICH IS ASSUMED TO
    # HAVE COORDINATES MEASURED IN MM WHEN RAW
    x_1_shift = piv_obj.pose.glob[0] * 1000 - piv_obj.pose.loc[0]
    x_2_shift = piv_obj.pose.glob[1] * 1000 - piv_obj.pose.loc[1]

    return x_1_shift, x_2_shift


def translate_coordinates(
    coordinates: Dict[str, np.ndarray],
    x_1_shift: float,
    x_2_shift: float,
) -> None:
    """Perform the coordinate translation from the local PIV to the global BeVERLI
    coordinate system.

    :param coordinates: Dictionary containing the stereo PIV coordinates.
    :param x_1_shift: Coordinate shift in the x_1 direction.
    :param x_2_shift: Coordinate shift in the x_2 direction.
    """
    coordinates["X"] += x_1_shift
    coordinates["Y"] += x_2_shift


def scale_coordinates(piv_obj, scale_factor: float) -> None:
    piv_obj.data["coordinates"]["X"] *= scale_factor
    piv_obj.data["coordinates"]["Y"] *= scale_factor


def rotate_profile(
    profile: Dict[str, Dict[str, Union[np.ndarray, float, int]]],
    rotation_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Rotate profile data.

    :param profile: The profile data.
    :param rotation_matrix: The rotation matrix.

    :return: A tuple containing the rotated quantities.
    :rtype: Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
    """
    velocity_vector = np.array([profile["mean_velocity"][f"{i}"] for i in "UVW"])

    components = ["UU", "UV", "UW", "UV", "VV", "VW", "UW", "VW", "WW"]
    re_stress_tensor = np.array(
        [
            [
                profile["reynolds_stress"][component]
                for component in components[i : i + 3]
            ]
            for i in range(0, 9, 3)
        ]
    )

    velocity_vector_rotated = rotation_matrix @ velocity_vector
    re_stress_tensor_rotated = rotate_tensor_profile(re_stress_tensor, rotation_matrix)

    if "strain_tensor" in profile:
        strain_tensor = np.array(
            [
                [profile["strain_tensor"][f"S_{i+1}{j+1}"] for j in range(3)]
                for i in range(3)
            ]
        )
        rotation_tensor = np.array(
            [
                [profile["rotation_tensor"][f"W_{i+1}{j+1}"] for j in range(3)]
                for i in range(3)
            ]
        )
        normalized_rotation_tensor = np.array(
            [
                [
                    profile["normalized_rotation_tensor"][f"O_{i+1}{j+1}"]
                    for j in range(3)
                ]
                for i in range(3)
            ]
        )

        strain_tensor_rotated = rotate_tensor_profile(strain_tensor, rotation_matrix)
        rotation_tensor_rotated = rotate_tensor_profile(
            rotation_tensor, rotation_matrix
        )
        normalized_rotation_tensor_rotated = rotate_tensor_profile(
            normalized_rotation_tensor, rotation_matrix
        )

        return (
            velocity_vector_rotated,
            re_stress_tensor_rotated,
            strain_tensor_rotated,
            rotation_tensor_rotated,
            normalized_rotation_tensor_rotated,
        )

    return velocity_vector_rotated, re_stress_tensor_rotated, None, None, None


def rotate_tensor_profile(tensor: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Rotate the profile of a tensor quantity.

    :param tensor: The tensor quantity as a (3, 3, N) array.
    :param rotation_matrix: The rotation matrix.

    :return: The rotated tensor profile.
    :rtype: np.ndarray
    """
    return (rotation_matrix[None, :, :] @ tensor.transpose(2, 0, 1) @ rotation_matrix.T[None, :, :]).transpose(1, 2, 0)


def set_rotated_profiles(
    profile: Dict[str, Dict[str, Union[np.ndarray, float, int]]],
    velocity_vector_rotated: np.ndarray,
    re_stress_tensor_rotated: np.ndarray,
    strain_tensor_rotated: Optional[np.ndarray] = None,
    rotation_tensor_rotated: Optional[np.ndarray] = None,
    normalized_rotation_tensor_rotated: Optional[np.ndarray] = None,
):
    """
    Set the rotated profile.

    :param profile: The profile data.
    :param velocity_vector_rotated: The rotated velocity data.
    :param re_stress_tensor_rotated: The rotated reynolds stress data.
    :param strain_tensor_rotated: The rotated strain tensor data.
    :param rotation_tensor_rotated: The rotated rotation tensor data.
    :param normalized_rotation_tensor_rotated: The rotated normalized rotation tensor data.
    """
    for index, component in enumerate(["U_SS", "V_SS", "W_SS"]):
        profile["mean_velocity"][component] = velocity_vector_rotated[index, :]

    for index, component in enumerate(["UU_SS", "VV_SS", "WW_SS"]):
        profile["reynolds_stress"][component] = re_stress_tensor_rotated[index, index, :]

    for indices, component in zip([(0, 1), (0, 2), (1, 2)], ["UV_SS", "UW_SS", "VW_SS"]):
        profile["reynolds_stress"][component] = re_stress_tensor_rotated[*indices, :]

    if (
        (strain_tensor_rotated is not None)
        and (rotation_tensor_rotated is not None)
        and (normalized_rotation_tensor_rotated is not None)
    ):
        indices_list = [(i, j) for i in range(3) for j in range(3)]
        strain_tensor_components = [f"S_{i+1}{j+1}_SS" for i in range(3) for j in range(3)]
        rotation_tensor_components = [f"W_{i + 1}{j + 1}_SS" for i in range(3) for j in range(3)]
        normalized_rotation_tensor_components = [f"O_{i + 1}{j + 1}_SS" for i in range(3) for j in range(3)]

        for indices, s_component, w_component, o_component in zip(
            indices_list,
            strain_tensor_components,
            rotation_tensor_components,
            normalized_rotation_tensor_components,
        ):
            profile["strain_tensor"][s_component] = strain_tensor_rotated[*indices, :]
            profile["rotation_tensor"][w_component] = rotation_tensor_rotated[*indices, :]
            profile["normalized_rotation_tensor"][o_component] = (rotation_tensor_rotated[*indices, :])
