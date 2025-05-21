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
# Properties = Dict[str, Dict[str, float]]
# TunnelConditions = Dict[str, Dict[str, Union[float, np.ndarray]]]

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


class PRInputs(TypedDict):
    """Type definition for profiler user inputs."""

    hill_orientation: float
    reference_stat_file: str
    add_gradients: bool
    reynolds_number: float
    tunnel_entry: int
    bypass_properties: bool
    gas_constant: Optional[float]
    gamma: Optional[float]
    density: Optional[float]
    mu: Optional[float]
    uinf: Optional[float]
    add_cfd: bool
    fluent_case: Optional[str]
    fluent_data: Optional[str]
    number_of_profiles: int
    number_of_profile_pts: int
    coordinate_system: str
    profile_height: float
    port_wall_pressure: Optional[str]
    hill_pressure: Optional[str]
    pressure_readme: Optional[str]
    add_reconstruction_points: Optional[bool]
    number_of_reconstruction_points: Optional[int]


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


class ReferenceProperties(TypedDict):
    """Type definition for reference properties."""

    T_ref: float
    p_ref: float
    U_ref: float
    M_ref: float
    density_ref: float
    dynamic_viscosity_ref: float


class FlowProperties(TypedDict):
    """Type definition for flow properties."""

    U_inf: float
    p_0: float
    p_inf: float
    p_atm: float
    T_0: float


class FluidProperties(TypedDict):
    """Type definition for fluid properties."""

    density: float
    dynamic_viscosity: float
    heat_capacity_ratio: float
    gas_constant: float


class Properties(TypedDict):
    """Type definition for fluid, flow, and reference properties."""

    fluid: FluidProperties
    flow: FlowProperties
    reference: ReferenceProperties


class StatFileRunData(TypedDict):
    """Type definition for .stat file data."""

    data: np.ndarray
    p_atm: float
    p_0: float
    T_0: float
    p_inf: float
    p_ref: float
    T_ref: float
    U_ref: float
    M_ref: float
    density_ref: float
    dynamic_viscosity_ref: float


class CFDRefConditions(TypedDict):
    """Type definition for cfd reference conditions."""

    p_0: float
    T_0: float
    p_ref: float
    T_ref: float
    density_ref: float
    U_ref: float
    dynamic_viscosity_ref: float


StatFileData = Dict[str, StatFileRunData]

SecParams = Tuple[float, float, float, float, float, float, float, float]

Vec3 = Tuple[float, float, float]

STAT = {
    "PORTNUM": 0,
    "X_L": 1,
    "Y_L": 2,
    "Z_L": 3,
    "U_REF": 4,
    "RE_L": 5,
    "T_F": 6,
    "C_P": 7,
    "P_IN_H2O": 8,
    "PRMS_IN_H2O": 9
}


# PROFILES
class Uncertainty(TypedDict):
    """Definition of profile Reynolds stress."""

    dU: Optional[np.ndarray]
    dV: Optional[np.ndarray]
    dW: Optional[np.ndarray]
    dU_SS: Optional[np.ndarray]
    dV_SS: Optional[np.ndarray]
    dW_SS: Optional[np.ndarray]
    dUU: Optional[np.ndarray]
    dVV: Optional[np.ndarray]
    dWW: Optional[np.ndarray]
    dUV: Optional[np.ndarray]
    dUW: Optional[np.ndarray]
    dVW: Optional[np.ndarray]
    dUU_SS: Optional[np.ndarray]
    dVV_SS: Optional[np.ndarray]
    dWW_SS: Optional[np.ndarray]
    dUV_SS: Optional[np.ndarray]
    dUW_SS: Optional[np.ndarray]
    dVW_SS: Optional[np.ndarray]


class BLParams(TypedDict):
    """Definition of profile Reynolds stress."""
    DELTA: float
    THRESHOLD: Optional[float]
    U_E: float
    DELTA_STAR: float
    THETA: float


class BLMethods(TypedDict):
    """Definition of profile Reynolds stress."""
    GRIFFIN: BLParams
    VINUESA: BLParams


class ProfileProperties(TypedDict):
    """Definition of profile Reynolds stress."""
    NU: float
    RHO: float
    U_REF: float
    U_INF: Optional[float]
    U_TAU: Optional[float]
    X_CORRECTION: Optional[float]
    Y_CORRECTION: Optional[float]
    Y_SS_CORRECTION: Optional[float]
    ANGLE_SS_DEG: Optional[float]
    BL_PARAMS: Optional[BLMethods]


class ProfileReynoldsStress(TypedDict):
    """Definition of profile Reynolds stress."""

    UU: np.ndarray
    VV: np.ndarray
    WW: np.ndarray
    UV: np.ndarray
    UW: np.ndarray
    VW: np.ndarray
    UU_SS: Optional[np.ndarray]
    VV_SS: Optional[np.ndarray]
    WW_SS: Optional[np.ndarray]
    UV_SS: Optional[np.ndarray]
    UW_SS: Optional[np.ndarray]
    VW_SS: Optional[np.ndarray]
    UU_SS_PLUS: Optional[np.ndarray]
    VV_SS_PLUS: Optional[np.ndarray]
    WW_SS_PLUS: Optional[np.ndarray]
    UV_SS_PLUS: Optional[np.ndarray]
    UW_SS_PLUS: Optional[np.ndarray]
    VW_SS_PLUS: Optional[np.ndarray]


class ProfileMeanVelocity(TypedDict):
    """Definition of profile mean velocity."""

    U: np.ndarray
    V: np.ndarray
    W: np.ndarray
    U_SS: Optional[np.ndarray]
    V_SS: Optional[np.ndarray]
    W_SS: Optional[np.ndarray]
    U_SS_PLUS: Optional[np.ndarray]
    V_SS_PLUS: Optional[np.ndarray]
    W_SS_PLUS: Optional[np.ndarray]
    U_SS_MODELED: Optional[np.ndarray]


class ProfileCoordinates(TypedDict):
    """Definition of profile coordinates."""

    X: np.ndarray
    Y: np.ndarray
    Y_SS: Optional[np.ndarray]
    Y_SS_PLUS: Optional[np.ndarray]
    Y_SS_MODELED: Optional[np.ndarray]
    Y_W: Optional[np.ndarray]


class ProfileTurbulenceScales(TypedDict):
    """Definition of profile turbulence scales."""

    NUT: np.ndarray


class ProfileData(TypedDict):
    """Definition of experimental profile."""

    coordinates: ProfileCoordinates
    mean_velocity: ProfileMeanVelocity
    reynolds_stress: Optional[ProfileReynoldsStress]
    strain_tensor: Optional[StrainTensor]
    rotation_tensor: Optional[RotationTensor]
    normalized_rotation_tensor: Optional[NormalizedRotationTensor]
    turbulence_scales: Optional[ProfileTurbulenceScales]
    uncertainty: Optional[Uncertainty]
    properties: ProfileProperties


class Profile(TypedDict):
    """Definition of single profile."""

    exp: ProfileData
    cfd: Optional[ProfileData]


class PressureInterpolants(TypedDict):
    """Interpolants types."""

    P_THIRD_ORDER: Interp1DCallable
    CP_THIRD_ORDER: Interp1DCallable
    CP_FIRST_ORDER: Interp1DCallable


class PressureProperties(TypedDict):
    """Pressure propeties."""

    P_INF: float
    U_INF: float
    P_0: float


class CenterlinePressure(TypedDict):
    """Centerline Pressure type."""

    interpolants: PressureInterpolants
    properties: PressureProperties
