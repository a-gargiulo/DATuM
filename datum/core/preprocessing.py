"""Define preprocessing functions."""
import sys
import numpy as np
from typing import cast, Dict, Union
from .piv import Piv
from ..utility import apputils, mathutils, tputils
from .load import load_raw_data
from . import analysis, transform


def preprocess_data(piv_obj: Piv, state: Dict[str, Union[bool, int, str]], opts: Dict[str, bool], data_paths: Dict[str, str], should_load: Dict[str, bool]) -> bool:
    """Enter docstring here."""
    load_raw_data(piv_obj, data_paths, should_load, opts)
    if piv_obj.data is None:
        return False
    if not state["compute_gradients"]:
        transform.rotate_data(piv_obj)
        transform.translate_data(piv_obj)
        transform.scale_coordinates(piv_obj, scale_factor=1e-3)
        if piv_obj.pose.angle2 != 0.0:
            piv_obj.data["coordinates"]["Z"] = piv_obj.data["coordinates"]["X"]
    else:
        transform.rotate_data(piv_obj)
        transform.interpolate_data(piv_obj, state["num_interpolation_pts"])
        transform.translate_data(piv_obj)
        transform.scale_coordinates(piv_obj, scale_factor=1e-3)

        if state["compute_gradients"] and piv_obj.pose.angle2 == 0.0:
            compute_velocity_gradient(
                piv_obj, cast(str, state["slice_path"]), cast(str, state["slice_name"]), opts
            )
            get_strain_and_rotation_tensor(piv_obj)
            get_eddy_viscosity(piv_obj)

    apputils.write_pickle("./outputs/preprocessed.pkl", piv_obj.data)
    return True


def compute_velocity_gradient(piv_obj: Piv, slice_path: str, zone_name: str, opts: Dict[str, bool]) -> None:
    """Computes the mean velocity gradient tensor from the BeVERLI Hill stereo PIV mean
    velocity data. Note that this function should be used with interpolated data!

    This function directly edits the :py:type:`Piv` object that is passed to it.

    :param piv_obj: Object containing the BeVERLI Hill stereo PIV data.
    """
    mean_vel_grad = {}

    # Obtain computable gradient components
    computable_components = _get_computable_velocity_gradient_components(piv_obj)
    apputils.update_nested_dict(mean_vel_grad, computable_components)

    # Use incompressibility assumption for dWdZ
    mean_vel_grad["dWdZ"] = -mean_vel_grad["dUdX"] - mean_vel_grad["dVdY"]

    # Get missing gradient components from CFD data
    print("Getting Tecplot derivatives... ", end="")
    x1_q, x2_q = (cast(dict, piv_obj.data)["coordinates"]["X"], cast(dict, piv_obj.data)["coordinates"]["Y"])
    cfd_data = tputils.get_tecplot_derivatives(slice_path, zone_name, opts)
    # cfd_coords = 1000 * np.column_stack(
    #     (cfd_data["X"].flatten(), cfd_data["Y"].flatten())
    # )
    cfd_coords = np.column_stack(
        (cfd_data["X"].flatten(), cfd_data["Y"].flatten())
    )
    for key, _ in cfd_data.items():
        if key not in {"x_1", "x_2"}:
            mean_vel_grad[key] = mathutils.interpolate(
                cfd_coords, cfd_data[key], (x1_q, x2_q)
            )
    print("Done!")

    # Set the mean velocity gradient data
    cast(dict, piv_obj.data)["mean_velocity_gradient"] = mean_vel_grad


def _get_computable_velocity_gradient_components(piv_obj: Piv) -> Dict[str, np.ndarray]:
    """Calculates the directly computable components of the mean velocity gradient
    tensor from the BeVERLI Hill stereo PIV mean velocity data.

    :param piv_obj: Object containing the BeVERLI Hill stereo PIV data.
    :return: A dictionary containing NumPy ndarrays of shape (m, n), where m and n
        represent the number of available data points in the :math:`x_1`- and
        :math:`x_2`-direction. Each array represents a computable component of the mean
        velocity gradient tensor.
    """
    if piv_obj.data is None:
        sys.exit(-1)
    coords = piv_obj.data["coordinates"]
    mean_vel = piv_obj.data["mean_velocity"]
    computable_gradients = [("dUdX", "dUdY"), ("dVdX", "dVdY"), ("dWdX", "dWdY")]

    components = {}
    for key, (du_key, dv_key) in zip(mean_vel, computable_gradients):
        ddx, ddy = mathutils.compute_derivative_2d(
            cast(np.ndarray, coords["X"]), cast(np.ndarray, coords["Y"]), cast(np.ndarray, mean_vel[key])
        )
        components[du_key] = ddx
        components[dv_key] = ddy

    return components


def get_strain_and_rotation_tensor(piv_obj: Piv) -> None:
    """Obtains the mean rate-of-strain and rotation tensors.

    This function directly edits the :py:type:`Piv` object that is passed to it.

    :param piv_obj: Object containing the BeVERLI Hill stereo PIV data.
    """
    base_tensors = analysis.get_base_tensors(piv_obj)
    piv_obj.data["strain_tensor"] = {
        f"S_{i+1}{j+1}": base_tensors["S"][i, j] for i in range(3) for j in range(3)
    }
    piv_obj.data["rotation_tensor"] = {
        f"W_{i+1}{j+1}": base_tensors["W"][i, j] for i in range(3) for j in range(3)
    }
    piv_obj.data["normalized_rotation_tensor"] = {
        f"O_{i+1}{j+1}": base_tensors["O"][i, j] for i in range(3) for j in range(3)
    }


def get_eddy_viscosity(piv_obj: Piv) -> None:
    """Obtains the eddy viscosity.

    This function directly edits the :py:type:`Piv` object that is passed to it.

    :param piv_obj: Object containing the BeVERLI Hill stereo PIV data.
    """
    base_tensors = analysis.get_base_tensors(piv_obj)
    (piv_obj.data["turbulence_scales"]["NUT"]) = analysis.calculate_eddy_viscosities(
        base_tensors
    )


