"""Define custom type alias."""
from typing import Any, Callable, Dict, Tuple, TypedDict, Union, Optional

import numpy as np
import trimesh

UserInputs = Dict[str, Union[bool, float, int, str]]

NestedDict = Dict[str, Union["NestedDict", Any]]
CadGeometry = trimesh.Trimesh
AnalyticGeometry = Dict[str, np.ndarray]
HillGeometry = Union[CadGeometry, AnalyticGeometry]
FloatOrArray = Union[float, np.ndarray]
Properties = Dict[str, Dict[str, float]]
TunnelConditions = Dict[str, Dict[str, Union[float, np.ndarray]]]

ProfileDictAll = Dict[str, Dict[str, Dict[str, Dict[str, Union[float, np.ndarray, Dict[str, Dict[str, float]]]]]]]
ProfileDictSingle = Dict[str, Dict[str, Union[float, np.ndarray, Dict[str, Dict[str, float]]]]]

SingleProfileDict = Dict[str, Dict[str, Dict[str, FloatOrArray]]]
SingleProfile = Dict[str, Dict[str, FloatOrArray]]

Interp1DCallable = Callable[[FloatOrArray], FloatOrArray]


# TRANSFOMRATION PARAMETERS
# -------------------------
class RotationParameters(TypedDict):
    """Type definition for rotation parameters."""

    angle_1_deg: float
    angle_2_deg: float


class TranslationParameters(TypedDict):
    """Type definition for translation parameters."""

    x_1_glob_ref_m: float
    x_2_glob_ref_m: float
    x_3_glob_ref_m: float
    x_1_loc_ref_mm: float
    x_2_loc_ref_mm: float


class TransformationParameters(TypedDict):
    """Type definition for transformation parameters."""

    rotation: RotationParameters
    translation: TranslationParameters


# POSE MEASUREMENT
# -------------------------
class CalibrationPlateLocation(TypedDict):
    """Type definition for calibration plate location."""

    x_1_m: float
    x_3_m: float


class Triangulation(TypedDict):
    """Type definition for triangulation."""

    upstream_plate_corner_arclength_position_m: float
    downstream_plate_corner_arclength_position_m: float


class CalibrationPlateAngle(TypedDict):
    """Type definitin for calibration plate angle."""

    direct_measurement_deg: float
    triangulation: Triangulation


class PoseMeasurement(TypedDict):
    """Type definition for pose measurement."""

    calibration_plate_angle: CalibrationPlateAngle
    calibration_plate_location: CalibrationPlateLocation


# PREPROCESSOR USER INPUTS
# -------------------------
class PPInputs(TypedDict):
    """Type definition for preprocessor user inputs."""

    piv_data_paths: Dict[str, str]
    load_set: Dict[str, bool]
    flip_u3: bool
    interpolate_data: bool
    num_interpolation_pts: Optional[int]
    compute_gradients: bool
    use_cfd_dwdx_and_dwdy: Optional[bool]
    slice_path: Optional[str]
    slice_name: Optional[str]


class Coordinates(TypedDict):
    """Type definition for PIV coordinates."""

    X: np.ndarray
    Y: np.ndarray
    Z: Optional[np.ndarray]


class MeanVelocity(TypedDict):
    """Type definition for PIV mean velocity."""

    U: np.ndarray
    V: np.ndarray
    W: np.ndarray


class ReynoldsStress(TypedDict):
    """Type definition for PIV Reynolds stress."""

    UU: np.ndarray
    VV: np.ndarray
    WW: np.ndarray
    UV: np.ndarray
    UW: np.ndarray
    VW: np.ndarray


class VelocitySnapshot(TypedDict):
    """Type definition for PIV inst. velocity frame."""

    U: np.ndarray
    V: np.ndarray
    W: np.ndarray


class TurbulenceScales(TypedDict):
    """Type definition for PIV turbulence scales."""

    TKE: Optional[np.ndarray]
    EPSILON: Optional[np.ndarray]
    NUT: Optional[np.ndarray]


class MeanVelocityGradient(TypedDict):
    """Type definition for PIV mean velocity gradient."""

    dUdX: np.ndarray
    dUdY: np.ndarray
    dUdZ: np.ndarray
    dVdX: np.ndarray
    dVdY: np.ndarray
    dVdZ: np.ndarray
    dWdX: np.ndarray


class StrainTensor(TypedDict):
    """Type definition for PIV strain tensor."""

    S11: np.ndarray
    S12: np.ndarray
    S13: np.ndarray
    S21: np.ndarray
    S22: np.ndarray
    S23: np.ndarray
    S31: np.ndarray
    S32: np.ndarray
    S33: np.ndarray


class RotationTensor(TypedDict):
    """Type definition for PIV rotation tensor."""

    W11: np.ndarray
    W12: np.ndarray
    W13: np.ndarray
    W21: np.ndarray
    W22: np.ndarray
    W23: np.ndarray
    W31: np.ndarray
    W32: np.ndarray
    W33: np.ndarray


class NormalizedRotationTensor(TypedDict):
    """Type definition for PIV normalized rotation tensor."""

    O11: np.ndarray
    O12: np.ndarray
    O13: np.ndarray
    O21: np.ndarray
    O22: np.ndarray
    O23: np.ndarray
    O31: np.ndarray
    O32: np.ndarray
    O33: np.ndarray


class PivData(TypedDict):
    """Type definition for PIV data structure."""

    coordinates: Coordinates
    mean_velocity: MeanVelocity
    reynolds_stress: Optional[ReynoldsStress]
    velocity_snapshot: Optional[VelocitySnapshot]
    turbulence_scales: Optional[TurbulenceScales]
    mean_velocity_gradient: Optional[MeanVelocityGradient]
    strain_tensor: Optional[StrainTensor]
    rotation_tensor: Optional[RotationTensor]
    normalized_rotation_tensor: Optional[NormalizedRotationTensor]


SecParams = Tuple[float, float, float, float, float, float, float, float]

Vec3 = Tuple[float, float, float]
