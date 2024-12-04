"""Provides routines for preprocessing the BeVERLI Hill stereo PIV data."""
import sys
from typing import Dict

import numpy as np

from . import log, my_math, parser, pose, transformations, utility
from .analysis import calculate_eddy_viscosities, get_base_tensors
from .cfd import get_tecplot_derivatives
from .my_types import PivData
from .piv import Piv


@log.log_process("Preprocess data", "main")
def preprocess_data(piv_obj: Piv) -> None:
    """Comprehensively preprocesses the BeVERLI Hill stereo PIV data.

    This function directly edits the :py:type:`Piv` object that is passed to it.

    :param piv_obj: Object containing the BeVERLI Hill stereo PIV data.

    """
    input_data = parser.InputFile().data
    output_file_path = utility.get_output_file_path()

    if input_data["preprocessor"]["active"]:
        get_coordinate_transformation_parameters(piv_obj)

        if input_data["piv_data"]["plane_is_diagonal"]:
            transform_data_without_interpolation(piv_obj)
        else:
            transform_data(piv_obj)

        if (
                input_data["preprocessor"]["mean_velocity_gradient_tensor"][
                    "computation_active"
                ]
                and not input_data["piv_data"]["plane_is_diagonal"]
        ):
            compute_velocity_gradient(piv_obj)
            get_strain_and_rotation_tensor(piv_obj)
            get_eddy_viscosity(piv_obj)

        utility.write_pickle(output_file_path, piv_obj.data)
    else:
        piv_obj.data = _get_preprocessed_data()


@log.log_process("Obtain PIV coordinate transformation parameters", "sub")
def get_coordinate_transformation_parameters(piv_obj: Piv) -> None:
    """Obtains the parameters transforming the BeVERLI Hill stereo PIV data from their
    local Cartesian PIV coordinate system to the global Cartesian coordinate system of
    the BeVERLI experiment in the Virginia Tech Stability Wind Tunnel.

    This function directly edits the :py:type:`Piv` object that is passed to it.

    :param piv_obj: Object containing the BeVERLI Hill stereo PIV data.
    """
    input_data = parser.InputFile().data
    config = input_data["preprocessor"]["coordinate_transformation"]["parameters"]
    warning_message_1 = (
        "WARNING:\n"
        "\t +--------------------------------------------------------------------------------+\n"
        "\t | The PIV plane angle is zero and you chose not to compute the PIV plane's pose. |\n"
        "\t | Please ensure a transformation parameter file exists before proceeding.        |\n"
        "\t +--------------------------------------------------------------------------------+\n\n"
    )
    warning_message_2 = (
        "WARNING:\n"
        "\t +--------------------------------------------------------------------------------+\n"
        "\t | You chose to only compute the PIV plane's local pose.                          |\n"
        "\t | Please ensure a transformation parameter file containing the global pose       |\n"
        "\t | exists before proceeding.                                                      |\n"
        "\t +--------------------------------------------------------------------------------+\n\n"
    )
    warning_message_3 = (
        "WARNING:\n"
        "\t +--------------------------------------------------------------------------------+\n"
        "\t | The computation of the global pose is not available for diagonal planes.       |\n"
        "\t | Please restart without the global pose option.                                 |\n"
        "\t +--------------------------------------------------------------------------------+\n\n"
    )

    if input_data["piv_data"]["plane_is_diagonal"]:
        angle_param = "angle_1_deg"
    else:
        angle_param = "angle_deg"

    if (
        (piv_obj.transformation_parameters["rotation"][angle_param] == 0)
        and (config["compute_global_active"] is False)
        and (config["compute_local_active"] is False)
    ):
        print(warning_message_1)
        input("Press enter to continue...")

    if config["compute_global_active"]:
        if input_data["piv_data"]["plane_is_diagonal"]:
            print(warning_message_3)
            sys.exit(1)
        pose.obtain_global_pose(piv_obj)

    if config["compute_local_active"]:
        if not config["compute_global_active"]:
            print(warning_message_2)
            input("Press enter to continue...")
        pose.obtain_local_pose(piv_obj)


@log.log_process("Transform data", "sub")
def transform_data(piv_obj: Piv) -> None:
    """Transforms the BeVERLI Hill stereo PIV data from their local Cartesian
    coordinate system to the global Cartesian coordinate system of the corresponding
    experiment in the Virginia Tech Stability Wind Tunnel.

    This routine rotates, translates and scales (mm -> m) the PIV data. Additionally,
    it interpolates the data onto a fine regular grid for the computation of gradients.

    This function directly edits the :py:type:`Piv` object that is passed to it.

    :param piv_obj: Object containing the BeVERLI Hill stereo PIV data.
    """
    transformations.rotate_data(piv_obj)
    transformations.interpolate_data(piv_obj)
    transformations.translate_data(piv_obj)
    transformations.scale_coordinates(piv_obj, scale_factor=1e-3)


@log.log_process(msg="Transform data w/o interpolation", proc_type="sub")
def transform_data_without_interpolation(piv_obj: Piv) -> None:
    """Transforms the BeVERLI Hill stereo PIV data like
    :py:meth:`datum.preprocessor.transform_data` but without interpolation onto a finer
    grid.

    This routine is intended for data that was not acquired in
    :math:`x_1`-:math:`x_2`-planes of the BeVERLI Hill experiment in the Virginia Tech
    Stability Wind Tunnel. The computation of gradients is unfeasible for such planes.

    This function directly edits the :py:type:`Piv` object that is passed to it.

    :param piv_obj: Object containing the BeVERLI Hill stereo PIV data.
    """
    input_data = parser.InputFile().data
    transformations.rotate_data(piv_obj)
    transformations.translate_data(piv_obj)
    transformations.scale_coordinates(piv_obj, scale_factor=1e-3)
    if input_data["piv_data"]["plane_is_diagonal"]:
        piv_obj.data["coordinates"]["Z"] = piv_obj.data["coordinates"]["X"]


@log.log_process("Calculate mean velocity gradient tensor", "sub")
def compute_velocity_gradient(piv_obj: Piv) -> None:
    """Computes the mean velocity gradient tensor from the BeVERLI Hill stereo PIV mean
    velocity data. Note that this function should be used with interpolated data!

    This function directly edits the :py:type:`Piv` object that is passed to it.

    :param piv_obj: Object containing the BeVERLI Hill stereo PIV data.
    """
    mean_vel_grad = {}

    # Obtain computable gradient components
    computable_components = _get_computable_velocity_gradient_components(piv_obj)
    utility.update_nested_dict(mean_vel_grad, computable_components)

    # Use incompressibility assumption for dWdZ
    mean_vel_grad["dWdZ"] = -mean_vel_grad["dUdX"] - mean_vel_grad["dVdY"]

    # Get missing gradient components from CFD data
    print("Getting Tecplot derivatives... ", end="")
    x1_q, x2_q = (piv_obj.data["coordinates"]["X"], piv_obj.data["coordinates"]["Y"])
    cfd_data = get_tecplot_derivatives()
    cfd_coords = 1000 * np.column_stack(
        (cfd_data["X"].flatten(), cfd_data["Y"].flatten())
    )
    for key, _ in cfd_data.items():
        if key not in {"x_1", "x_2"}:
            mean_vel_grad[key] = my_math.interpolate(
                cfd_coords, cfd_data[key], (x1_q, x2_q)
            )
    print("Done!")

    # Set the mean velocity gradient data
    piv_obj.data["mean_velocity_gradient"] = mean_vel_grad


@log.log_process("Computable components", "subsub")
def _get_computable_velocity_gradient_components(piv_obj: Piv) -> Dict[str, np.ndarray]:
    """Calculates the directly computable components of the mean velocity gradient
    tensor from the BeVERLI Hill stereo PIV mean velocity data.

    :param piv_obj: Object containing the BeVERLI Hill stereo PIV data.
    :return: A dictionary containing NumPy ndarrays of shape (m, n), where m and n
        represent the number of available data points in the :math:`x_1`- and
        :math:`x_2`-direction. Each array represents a computable component of the mean
        velocity gradient tensor.
    """
    coords = piv_obj.data["coordinates"]
    mean_vel = piv_obj.data["mean_velocity"]
    computable_gradients = [("dUdX", "dUdY"), ("dVdX", "dVdY"), ("dWdX", "dWdY")]

    components = {}
    for key, (du_key, dv_key) in zip(mean_vel, computable_gradients):
        ddx, ddy = my_math.compute_derivative_2d(
            coords["X"], coords["Y"], mean_vel[key]
        )
        components[du_key] = ddx
        components[dv_key] = ddy

    return components


@log.log_process("Calculate strain and rotation tensors", "sub")
def get_strain_and_rotation_tensor(piv_obj: Piv) -> None:
    """Obtains the mean rate-of-strain and rotation tensors.

    This function directly edits the :py:type:`Piv` object that is passed to it.

    :param piv_obj: Object containing the BeVERLI Hill stereo PIV data.
    """
    base_tensors = get_base_tensors(piv_obj)
    piv_obj.data["strain_tensor"] = {
        f"S_{i+1}{j+1}": base_tensors["S"][i, j] for i in range(3) for j in range(3)
    }
    piv_obj.data["rotation_tensor"] = {
        f"W_{i+1}{j+1}": base_tensors["W"][i, j] for i in range(3) for j in range(3)
    }
    piv_obj.data["normalized_rotation_tensor"] = {
        f"O_{i+1}{j+1}": base_tensors["O"][i, j] for i in range(3) for j in range(3)
    }


@log.log_process("Calculate eddy viscosity", "sub")
def get_eddy_viscosity(piv_obj: Piv) -> None:
    """Obtains the eddy viscosity.

    This function directly edits the :py:type:`Piv` object that is passed to it.

    :param piv_obj: Object containing the BeVERLI Hill stereo PIV data.
    """
    base_tensors = get_base_tensors(piv_obj)
    (piv_obj.data["turbulence_scales"]["NUT"]) = calculate_eddy_viscosities(
        base_tensors
    )


def _get_preprocessed_data() -> PivData:
    """Attempts to load pre-processed BeVERLI Hill stereo PIV data if the user runs the
    pre-processor with the `deactivated` flag.

    :rtype: :py:data:`datum.my_types.PivData`
    :return: The pre-processed BeVERLI Hill stereo PIV data.
    """
    print(
        "Data will not be preprocessed.\n\nSearching and loading preprocessed data...\n"
    )
    try:
        data = utility.load_pickle(utility.get_output_file_path())
    except FileNotFoundError:
        print("No preprocessed data found!")
        sys.exit(1)

    return data
