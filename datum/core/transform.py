"""Define functions for transforming the BeVERLI Hill stereo PIV data.

The data is trasnformed from the local PIV coordinate system to the global
Cartesian coordinate system of the corresponding BeVERLI Hill experiment in
the Virginia Tech Stability Wind Tunnel.
"""

from typing import cast, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np

from ..utility import mathutils

if TYPE_CHECKING:
    from .piv import Piv








def _obtain_rotation_matrix(piv: "Piv"):
    if piv.pose.angle2 != 0.0:
        rotation_angle_1_deg = piv.pose.angle1
        rotation_angle_2_deg = piv.pose.angle2
        rotation_matrix_1 = get_rotation_matrix(
            rotation_angle_1_deg, (0, 0, 1)
        )
        rotation_matrix_2 = get_rotation_matrix(
            rotation_angle_2_deg, (0, -1, 0)
        )
        rotation_matrix = rotation_matrix_2 @ rotation_matrix_1
    else:
        rotation_angle_deg = piv.pose.angle1
        rotation_matrix = get_rotation_matrix(rotation_angle_deg, (0, 0, 1))

    return rotation_matrix


def rotate_flow_quantity(
    piv: "Piv",
    quantity: str,
    components: List[str],
    rotation_matrix: np.ndarray
):
    """Rotate a specified BeVERLI Hill stereo PIV flow quantity.

    :param piv: Object containing the BeVERLI Hill stereo PIV data.
    :param quantity: String specifying the flow quantity to rotate.
    :param components: A list of strings specifying the components of the
        vector or tensor quantity to rotate.
    :param rotation_matrix: The rotation matrix used to rotate the data as
        NumPy array of shape (3, 3).
    """
    if len(components) == 9:
        tensor = prepare_tensor_quantity_for_rotation(
            piv, quantity, components
        )
        rotated_tensor = rotate_tensor_quantity(tensor, rotation_matrix)
        set_rotated_tensor_quantity(
            piv, quantity, components, rotated_tensor
        )
    else:
        vector = prepare_vector_quantity_for_rotation(
            piv, quantity, components
        )
        rotated_vector = rotate_planar_vector_field(vector, rotation_matrix)
        set_rotated_vector_quantity(
            piv, quantity, components, rotated_vector
        )


def rotate_planar_vector_field(
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


def rotate_tensor_quantity(
    tensor: np.ndarray, rotation_matrix: np.ndarray
) -> np.ndarray:
    """Rotate a tensor quantity based on the provided rotation matrix.

    This method takes a rotation matrix and a list of tensor component matrices
    as input. It then calculates the rotated quantity and returns it.

    :param rotation_matrix: NumPy ndarray of shape (3, 3) representing the
        rotation matrix.
    :param tensor: List of lists of NumPy ndarrays of shape (m, n), each
        containing a specific vector component, where m and n represent the
        number of available data points in the x:sub:`1`- and
        x:sub:`2`-direction.
    :return: List of lists of NumPy ndarrays of shape (m, n) containing the
        rotated vector quantity's components, where m and n represent the
        number of available data points in the x:sub:`1`- and
        x:sub:`2`-direction.
    """
    rotated_tensor = (
        rotation_matrix[None, None, :, :]
        @ tensor.transpose(2, 3, 0, 1)
        @ rotation_matrix.T[None, None, :, :]
    ).transpose(2, 3, 0, 1)

    return rotated_tensor


def prepare_vector_quantity_for_rotation(
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
    assert piv.data is not None
    vector = np.array(
        [piv.data[quantity][component] for component in components]
    )

    return vector


def prepare_tensor_quantity_for_rotation(
    piv: "Piv", quantity: str, components: List[str]
) -> np.ndarray:
    """Build tensor of a desired quantity from the corresponding components.

    :param piv_obj: Instance of the :py:class:`datum.piv.Piv` class.
    :param quantity: String specifying the tensor quantity.
    :param components: List of strings specifying the tensor components.
    :return: List of lists of NumPy ndarrays of shape (m, n), containing
        specific tensor components, where m and n represent the number of
        available data points in the x:sub:`1`- and x:sub:`2`-direction.
    """
    assert piv.data is not None
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


def set_rotated_vector_quantity(
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
    assert piv.data is not None
    for component, rotated_vector_component in zip(components, rotated_vector):
        piv.data[quantity][component] = rotated_vector_component


def set_rotated_tensor_quantity(
    piv: "Piv",
    quantity: str,
    components: List[str],
    rotated_tensor: np.ndarray
):
    """Update the BeVERLI stereo PIV data with the rotated tensor.

    :param piv_obj: Instance of the :py:class:`datum.piv.Piv` class.
    :param quantity: String specifying the tensor quantity.
    :param components: List of strings specifying the tensor components.
    :param rotated_tensor: List of lists of NumPy ndarrays of shape (m, n).
        Each array represents a tensor component, where m and n represent the
        number of available data points in the x:sub:`1`- and
        x:sub:`2`-direction.
    """
    assert piv.data is not None

    processed_keys = set()
    for component, rotated_tensor_component in zip(
        components, (i for sublist in rotated_tensor for i in sublist)
    ):
        if component not in processed_keys:
            piv.data[quantity][component] = rotated_tensor_component
            processed_keys.add(component)
        else:
            continue


def interpolate_data(piv: "Piv", n_grid):
    """Perform a comprehensive routine to interpolate the stereo PIV data.

    Interpolates onto a dense grid within the global BeVERLI coordinate system.

    :param piv: Instance of the Piv class.
    """
    assert piv.data is not None

    x1_q, x2_q = get_interpolation_grid(piv, n_grid)

    data_to_interpolate = [
        ("mean_velocity", ["U", "V", "W"]),
        ("reynolds_stress", ["UU", "VV", "WW", "UV", "UW", "VW"]),
        ("turbulence_scales", ["TKE", "epsilon"]),
        ("instantaneous_velocity_frame", ["U", "V", "W"]),
    ]

    coords = get_flattened_coordinates(piv)
    for data_type, keys in data_to_interpolate:
        for key in keys:
            is_avail = all([piv.search(data_type), piv.search(key)])
            if is_avail:
                data = get_flattened_data(piv, data_type, key)
                interp_data = mathutils.interpolate(coords, data, (x1_q, x2_q))
                set_interpolated_data(piv, data_type, key, interp_data)

    piv.data["coordinates"] = {"X": x1_q, "Y": x2_q, "Z": None}


def get_interpolation_grid(
    piv: "Piv", n_grid
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a structured, dense coordinate mesh for interpolation.

    :param piv: Instance of the :py:class:`datum.piv.Piv` class.

    :return: A tuple containing NumPy ndarrays of shape (m, n) that represent a
        mesh grid, where m and n represent the number of available data points
        in the x:sub:`1`- and x:sub:`2`-direction.
    """
    assert piv.data is not None

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

    :param piv_obj: Instance of the :py:class:`datum.piv.Piv` class.
    :return: NumPy ndarray of shape (m, 2) containing the flattened set of
        spatial coordinates for the desired quantity from the stereo PIV data,
        where m represents the number of total available data points.
    """
    assert piv.data is not None
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
    assert piv.data is not None

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
    assert piv.data is not None
    piv.data[data_type][key] = interpolated_data


def translate_data(piv: "Piv"):
    """Translate the stereo PIV data."""
    assert piv.data is not None

    x_1_shift, x_2_shift = get_translation_vector(piv)
    coords = cast(Dict[str, np.ndarray], piv.data["coordinates"])
    translate_coordinates(coords, x_1_shift, x_2_shift)


def get_translation_vector(piv: "Piv") -> Tuple[float, float]:
    """Obtain the translation vector.

    :param piv: Instance of the :py:class:`datum.piv.Piv` class.
    :return: A tuple of floats, representing the components of the translation
        vector.
    """
    # THIS FUNCTION IS QUITE SPECIFIC TO THE BEVERLI PIV DATA, WHICH IS ASSUMED
    # TO HAVE COORDINATES MEASURED IN MM WHEN RAW
    x_1_shift = piv.pose.glob[0] * 1000 - piv.pose.loc[0]
    x_2_shift = piv.pose.glob[1] * 1000 - piv.pose.loc[1]

    return x_1_shift, x_2_shift


def translate_coordinates(
    coordinates: Dict[str, np.ndarray],
    x_1_shift: float,
    x_2_shift: float,
):
    """Perform the coordinate translation.

    :param coordinates: Dictionary containing the stereo PIV coordinates.
    :param x_1_shift: Coordinate shift in the x_1 direction.
    :param x_2_shift: Coordinate shift in the x_2 direction.
    """
    coordinates["X"] += x_1_shift
    coordinates["Y"] += x_2_shift


def scale_coordinates(piv: "Piv", scale_factor: float):
    """Scale PIV data plane."""
    assert piv.data is not None

    piv.data["coordinates"]["X"] *= scale_factor
    piv.data["coordinates"]["Y"] *= scale_factor


def rotate_profile(
    profile: Dict[str, Dict[str, Union[np.ndarray, float, int]]],
    rotation_matrix: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """
    Rotate profile data.

    :param profile: The profile data.
    :param rotation_matrix: The rotation matrix.

    :return: A tuple containing the rotated quantities.
    :rtype: Tuple[
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray]
    ]
    """
    velocity_vector = np.array(
        [profile["mean_velocity"][f"{i}"] for i in "UVW"]
    )

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
    re_stress_tensor_rotated = rotate_tensor_profile(
        re_stress_tensor, rotation_matrix
    )

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

        strain_tensor_rotated = rotate_tensor_profile(
            strain_tensor, rotation_matrix
        )
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


def rotate_tensor_profile(
    tensor: np.ndarray, rotation_matrix: np.ndarray
) -> np.ndarray:
    """
    Rotate the profile of a tensor quantity.

    :param tensor: The tensor quantity as a (3, 3, N) array.
    :param rotation_matrix: The rotation matrix.

    :return: The rotated tensor profile.
    :rtype: np.ndarray
    """
    return (
        rotation_matrix[None, :, :]
        @ tensor.transpose(2, 0, 1)
        @ rotation_matrix.T[None, :, :]
    ).transpose(1, 2, 0)


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
        profile["reynolds_stress"][component] = re_stress_tensor_rotated[
            index, index, :
        ]

    for indices, component in zip(
        [(0, 1), (0, 2), (1, 2)], ["UV_SS", "UW_SS", "VW_SS"]
    ):
        profile["reynolds_stress"][component] = re_stress_tensor_rotated[
            *indices, :
        ]

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
            profile["strain_tensor"][s_component] = strain_tensor_rotated[
                *indices, :
            ]
            profile["rotation_tensor"][w_component] = rotation_tensor_rotated[
                *indices, :
            ]
            profile["normalized_rotation_tensor"][o_component] = (
                rotation_tensor_rotated[*indices, :]
            )
