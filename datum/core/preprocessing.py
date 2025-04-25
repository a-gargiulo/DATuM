"""Define preprocessing core functions."""

from typing import TYPE_CHECKING, Dict, cast

import numpy as np

from ..utility import apputils, mathutils, tputils
from . import analysis, load, transform
from .my_types import (
    MeanVelocityGradient,
    NestedDict,
    NormalizedRotationTensor,
    PPInputs,
    RotationTensor,
    StrainTensor,
    TurbulenceScales,
)

if TYPE_CHECKING:
    from .piv import Piv


def preprocess_data(piv: "Piv", ui: PPInputs) -> bool:
    """Core preprocessing function.

    :param piv: PIV plane.
    :param ui: User inputs.

    :return: False in case of an error. True otherwise.
    :rtype: bool
    """
    load.load_raw_data(piv, ui)

    try:
        if ui["interpolate_data"]:
            if piv.pose.angle2 != 0.0:
                print("[ERROR]: Interpolation not allowed for diagonal planes.")
                return False
            transform_data(piv, ui["num_interpolation_pts"])
            if ui["compute_gradients"]:
                calculate_velocity_gradient(piv, ui)
                calculate_strain_and_rotation_tensor(piv)
                calculate_eddy_viscosity(piv)
        else:
            transform_data_no_interp(piv)
            if piv.pose.angle2 != 0.0:
                piv.data["coordinates"]["Z"] = piv.data["coordinates"]["X"]

        apputils.write_pickle(
            "./outputs/preprocessed.pkl", cast(NestedDict, piv.data)
        )
        return True
    except ValueError:
        print("[ERROR]: PIV data was not loaded correctly.")
        return False


def transform_data(piv: "Piv", num_interp_pts: int):
    """Rotate, interpolate, translate, and scale the PIV data.

    :param piv: PIV data.
    :param num_interp_pts: Number of grid points for interpolation.
    """
    transform.rotate_data(piv)
    transform.interpolate_data(piv, num_interp_pts)
    transform.translate_data(piv)
    transform.scale_coordinates(piv, scale_factor=1e-3)


def transform_data_no_interp(piv: "Piv"):
    """Rotate, translate, and scale the PIV data.

    :param piv: PIV data.
    """
    transform.rotate_data(piv)
    transform.translate_data(piv)
    transform.scale_coordinates(piv, scale_factor=1e-3)


def calculate_velocity_gradient(piv: "Piv", ui: PPInputs):
    """Compute the mean velocity gradient tensor from the PIV data.

    Note, this function is used with interpolated data.

    :param piv: PIV plane data.
    :param ui: User inputs from the GUI.
    """
    assert piv.data is not None

    mean_vel_grad = {}

    # Computable components
    computable_components = _calculate_computable_components(piv)
    apputils.update_nested_dict(mean_vel_grad, computable_components)

    # Incompressibility
    mean_vel_grad["dWdZ"] = -mean_vel_grad["dUdX"] - mean_vel_grad["dVdY"]

    # Gradient components from CFD data
    x1_q, x2_q = (piv.data["coordinates"]["X"], piv.data["coordinates"]["Y"])
    cfd_data = tputils.get_tecplot_derivatives(
        ui["slice_path"], ui["slice_name"], ui["use_cfd_dwdx_and_dwdy"]
    )
    cfd_coords = np.column_stack(
        (cfd_data["X"].flatten(), cfd_data["Y"].flatten())
    )
    for key, _ in cfd_data.items():
        if key not in {"x_1", "x_2"}:
            mean_vel_grad[key] = mathutils.interpolate(
                cfd_coords, cfd_data[key], (x1_q, x2_q)
            )

    piv.data["mean_velocity_gradient"] = cast(
        MeanVelocityGradient, mean_vel_grad
    )


def _calculate_computable_components(piv: "Piv") -> Dict[str, np.ndarray]:
    assert piv.data is not None

    coords = piv.data["coordinates"]
    mean_vel = piv.data["mean_velocity"]
    computable_gradients = [
        ("dUdX", "dUdY"),
        ("dVdX", "dVdY"),
        ("dWdX", "dWdY"),
    ]

    components = {}
    for key, (du_key, dv_key) in zip(mean_vel, computable_gradients):
        ddx, ddy = mathutils.compute_derivative_2d(
            coords["X"], coords["Y"], mean_vel[key]
        )
        components[du_key] = ddx
        components[dv_key] = ddy

    return components


def calculate_strain_and_rotation_tensor(piv: "Piv"):
    """Calculate the mean rate-of-strain and rotation tensors.

    :param piv: PIV plane data.
    """
    assert piv.data is not None

    base_tensors = analysis.get_base_tensors(piv)
    S = {
        f"S_{i+1}{j+1}": base_tensors["S"][i, j]
        for i in range(3)
        for j in range(3)
    }
    piv.data["strain_tensor"] = cast(StrainTensor, S)
    W = {
        f"W_{i+1}{j+1}": base_tensors["W"][i, j]
        for i in range(3)
        for j in range(3)
    }
    piv.data["rotation_tensor"] = cast(RotationTensor, W)
    Wn = {
        f"O_{i+1}{j+1}": base_tensors["O"][i, j]
        for i in range(3)
        for j in range(3)
    }
    piv.data["normalized_rotation_tensor"] = cast(NormalizedRotationTensor, Wn)


def calculate_eddy_viscosity(piv: "Piv"):
    """Calculate the eddy viscosity.

    :param piv: PIV plane data.
    """
    assert piv.data is not None

    base_tensors = analysis.get_base_tensors(piv)
    (cast(TurbulenceScales, piv.data["turbulence_scales"])["NUT"]) = (
        analysis.calculate_eddy_viscosity(base_tensors)
    )
