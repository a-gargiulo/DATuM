"""Provides functions for transforming the BeVERLI Hill stereo PIV data from the local
PIV coordinate system to the global Cartesian coordinate system of the corresponding
BeVERLI Hill experiment in the Virginia Tech Stability Wind Tunnel."""

from typing import Dict, List, Tuple

import numpy as np

from . import log
from .my_math import interpolate
from .parser import InputFile


def get_rotation_matrix(
    rotation_angle_deg: float, rotation_axis: Tuple[float, float, float]
) -> np.ndarray:
    """Gets the rotation matrix for an Euler rotation of a body about a specified axis
    of its Cartesian coordinate system.

    :param rotation_angle_deg: Rotation angle measured in degrees.
    :param rotation_axis: 3D vector components of the axis of rotation as a tuple of
        shape (3, ).

    :return: Rotation matrix as :py:type:`ndarray` of shape (3, 3).
    :rtype: :py:type:`ndarray`
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
    """Normalizes an input vector.

    :param data: 3D vector components of an input vector.

    :return: The normalized vector as :py:type:`ndarray` of shape (3, ).
    :rtype: :py:type:`ndarray`
    """
    data = np.array(data, dtype=np.float64)
    data /= np.sqrt(np.dot(data, data))
    return data


@log.log_process("Rotate data", "subsub")
def rotate_data(piv_obj) -> None:
    """Rotates the `original` BeVERLI Hill stereo PIV data from its local PIV coordinate
    system to the Cartesian coordinate system used in the Virginia Tech Stability Wind
    Tunnel.

    The `original` data comprises coordinates, mean velocity, Reynolds stress tensor,
    and an instantaneous velocity frame (if available). Additional derived quantities,
    such as the mean rate-of-strain tensor, are computed directly in the (rotated)
    coordinate system of the Virginia Tech Stability Wind Tunnel.

    If you need to rotate a specific flow quantity to a coordinate system not handled by
    this routine, consider the :py:meth:`datum.transformations.rotate_flow_quantity`
    function.

    The present routine directly edits the :py:type:`Piv` object that is passed to it.

    :param piv_obj: Object containing the BeVERLI Hill stereo PIV data.
    """
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


def _obtain_rotation_matrix(piv_obj):
    input_data = InputFile().data

    if input_data["piv_data"]["plane_is_diagonal"]:
        rotation_angle_1_deg = piv_obj.transformation_parameters["rotation"][
            "angle_1_deg"
        ]
        rotation_angle_2_deg = piv_obj.transformation_parameters["rotation"][
            "angle_2_deg"
        ]
        rotation_matrix_1 = get_rotation_matrix(rotation_angle_1_deg, (0, 0, 1))
        rotation_matrix_2 = get_rotation_matrix(rotation_angle_2_deg, (0, -1, 0))
        rotation_matrix = rotation_matrix_2 @ rotation_matrix_1
    else:
        rotation_angle_deg = piv_obj.transformation_parameters["rotation"]["angle_deg"]
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
        rotated_vector = rotate_vector_quantity(vector, rotation_matrix)
        set_rotated_vector_quantity(piv_obj, quantity, components, rotated_vector)


# static
def rotate_vector_quantity(
    vector: np.ndarray, rotation_matrix: np.ndarray
) -> np.ndarray:
    """Rotate a BeVERLI stereo PIV vector quantity based on the provided rotation
    matrix.

    This method takes a rotation matrix and a list of vector component matrices as
    input. It then calculates the rotated quantity and returns it.

    :param rotation_matrix: NumPy ndarray of shape (3, 3) representing the rotation
        matrix.
    :param vector: List of NumPy ndarrays of shape (m, n), each containing a
        specific vector component, where m and n represent the number of available data
        points in the x:sub:`1`- and x:sub:`2`-direction.
    :return: List of NumPy ndarrays of shape (m, n) containing the rotated vector
        quantity's components, where m and n represent the number of available data
        points in the x:sub:`1`- and x:sub:`2`-direction.
    """
    if vector.shape[0] < 2:
        raise TypeError("You must provide a list of at least two quantities.")

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


@log.log_process("Interpolate data", "subsub")
def interpolate_data(piv_obj) -> None:
    """Perform a comprehensive routine to interpolate the BeVERLI stereo PIV data to a
    dense grid within the global BeVERLI coordinate system.

    :param piv_obj: Instance of the :py:class:`datum.piv.Piv` class.
    """
    x1_q, x2_q = get_interpolation_grid(piv_obj)

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
                interp_data = interpolate(coords, data, (x1_q, x2_q))
                set_interpolated_data(piv_obj, data_type, key, interp_data)

    piv_obj.data["coordinates"] = {"X": x1_q, "Y": x2_q}


def get_interpolation_grid(piv_obj) -> Tuple[np.ndarray, np.ndarray]:
    """Build a structured, dense coordinate mesh for the interpolation of the BeVERLI
    stereo PIV data.

    :param piv_obj: Instance of the :py:class:`datum.piv.Piv` class.
    :return: A tuple containing NumPy ndarrays of shape (m, n) that represent a mesh
        grid, where m and n represent the number of available data points in the
        x:sub:`1`- and x:sub:`2`-direction.
    """
    input_data = InputFile().data
    num_of_pts = input_data["preprocessor"]["coordinate_transformation"][
        "interpolation_grid_size"
    ]
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


@log.log_process("Translate data", "subsub")
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
    translation = piv_obj.transformation_parameters["translation"]
    x_1_shift = translation["x_1_glob_ref_m"] * 1000 - translation["x_1_loc_ref_mm"]
    x_2_shift = translation["x_2_glob_ref_m"] * 1000 - translation["x_2_loc_ref_mm"]

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


@log.log_process("Scale data", "subsub")
def scale_coordinates(piv_obj, scale_factor: float) -> None:
    piv_obj.data["coordinates"]["X"] *= scale_factor
    piv_obj.data["coordinates"]["Y"] *= scale_factor


def rotate_profile(profile, rotation_matrix):
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
    re_stress_tensor_rotated = (
        rotation_matrix @ re_stress_tensor.transpose(2, 0, 1) @ rotation_matrix.T
    ).transpose(1, 2, 0)

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


def rotate_tensor_profile(tensor, rotation_matrix):
    return (
        rotation_matrix[None, :, :]
        @ tensor.transpose(2, 0, 1)
        @ rotation_matrix.T[None, :, :]
    ).transpose(1, 2, 0)


def set_rotated_profiles(
    profile,
    velocity_vector_rotated,
    re_stress_tensor_rotated,
    strain_tensor_rotated=None,
    rotation_tensor_rotated=None,
    normalized_rotation_tensor_rotated=None,
):
    for index, component in enumerate(["U_SS", "V_SS", "W_SS"]):
        profile["mean_velocity"][component] = velocity_vector_rotated[index, :]

    for index, component in enumerate(["UU_SS", "VV_SS", "WW_SS"]):
        profile["reynolds_stress"][component] = re_stress_tensor_rotated[
            index, index, :
        ]

    for indices, component in zip(
        [(0, 1), (0, 2), (1, 2)], ["UV_SS", "UW_SS", "VW_SS"]
    ):
        profile["reynolds_stress"][component] = re_stress_tensor_rotated[*indices, :]

    if (
        (strain_tensor_rotated is not None)
        and (rotation_tensor_rotated is not None)
        and (normalized_rotation_tensor_rotated is not None)
    ):
        indices_list = [(i, j) for i in range(3) for j in range(3)]
        strain_tensor_components = [
            f"S_{i+1}{j+1}_SS" for i in range(3) for j in range(3)
        ]
        rotation_tensor_components = [
            f"W_{i + 1}{j + 1}_SS" for i in range(3) for j in range(3)
        ]
        normalized_rotation_tensor_components = [
            f"O_{i + 1}{j + 1}_SS" for i in range(3) for j in range(3)
        ]

        for indices, s_component, w_component, o_component in zip(
            indices_list,
            strain_tensor_components,
            rotation_tensor_components,
            normalized_rotation_tensor_components,
        ):
            profile["strain_tensor"][s_component] = strain_tensor_rotated[*indices, :]
            profile["rotation_tensor"][w_component] = rotation_tensor_rotated[
                *indices, :
            ]
            profile["normalized_rotation_tensor"][o_component] = (
                rotation_tensor_rotated[*indices, :]
            )
