"""BeVERLI Hill stereo PIV data preprocessing core functionalities."""
from typing import TYPE_CHECKING, Dict, cast

import numpy as np

from datum.core import analysis, io, transform
from datum.core.my_types import (
    MeanVelocityGradient,
    NestedDict,
    NormalizedRotationTensor,
    PPInputs,
    RotationTensor,
    StrainTensor,
    TurbulenceScales,
)
from datum.utility import apputils, mathutils, tputils

if TYPE_CHECKING:
    from .piv import Piv


def preprocess_all(piv: "Piv", ui: PPInputs) -> None:
    """Preprocess all data.

    :param piv: PIV plane.
    :param ui: User inputs from the preprocessing GUI.
    :raises RuntimeError: If loading/writing PIV data or gradient computation
        are unsuccessful.
    :raises ValueError: If a wrong user input occured.
    """
    io.load_raw_data(piv, ui)
    if ui["interpolate_data"]:
        if piv.pose.angle2 != 0.0:
            raise ValueError("No interpolation for diagonal planes.")
        transform_data(piv, cast(int, ui["num_interpolation_pts"]))
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


def transform_data(piv: "Piv", num_interp_pts: int):
    """Rotate, interpolate, translate, and scale the PIV data.

    :param piv: PIV data.
    :param num_interp_pts: Number of grid points for interpolation.
    """
    transform.rotation.rotate_all(piv)
    transform.interpolation.interpolate_all(piv, num_interp_pts)
    transform.translation.translate_all(piv)
    transform.scaling.scale_all(piv, scale_factor=1e-3)


def transform_data_no_interp(piv: "Piv"):
    """Rotate, translate, and scale the PIV data.

    :param piv: PIV data.
    """
    transform.rotation.rotate_all(piv)
    transform.translation.translate_all(piv)
    transform.scaling.scale_all(piv, scale_factor=1e-3)


def calculate_velocity_gradient(piv: "Piv", ui: PPInputs) -> None:
    """Compute the mean velocity gradient tensor from the PIV data.

    Note, this function can only be used with interpolated data.

    :param piv: PIV plane data.
    :param ui: User inputs from the preprocessing GUI.
    :raises RuntimeError: If gradient computation fails.
    """
    COMPONENTS = (
        "dUdX", "dUdY", "dUdZ",
        "dVdX", "dVdY", "dVdZ",
        "dWdX", "dWdY", "dWdZ"
    )
    mean_vel_grad = {}

    # Computable components
    computable_components = _calculate_computable_components(piv)
    apputils.update_nested_dict(mean_vel_grad, computable_components)

    # Incompressibility
    mean_vel_grad["dWdZ"] = -mean_vel_grad["dUdX"] - mean_vel_grad["dVdY"]

    # Gradient components from CFD data
    x1_q, x2_q = (piv.data["coordinates"]["X"], piv.data["coordinates"]["Y"])
    cfd_data = tputils.get_tecplot_derivatives(
        cast(str, ui["slice_path"]),
        cast(str, ui["slice_name"]),
        cast(bool, ui["use_cfd_dwdx_and_dwdy"])
    )
    cfd_coords = np.column_stack(
        (cfd_data["X"].flatten(), cfd_data["Y"].flatten())
    )
    for key, _ in cfd_data.items():
        if key not in {"X", "Y"}:
            mean_vel_grad[key] = mathutils.interpolate(
                cfd_coords, cfd_data[key], (x1_q, x2_q)
            )

    for key, _ in mean_vel_grad.items():
        if key not in COMPONENTS:
            raise RuntimeError("Missing mean velocity gradient component.")

    piv.data["mean_velocity_gradient"] = cast(
        MeanVelocityGradient, mean_vel_grad
    )


def _calculate_computable_components(piv: "Piv") -> Dict[str, np.ndarray]:
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
