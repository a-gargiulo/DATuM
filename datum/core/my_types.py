"""Define custom type alias."""
from typing import Any, Callable, Dict, TypedDict, Union

import numpy as np
import trimesh

UserInputs = Dict[str, Union[bool, float, int, str]]

NestedDict = Dict[str, Union["NestedDict", Any]]
PivData = Dict[str, Union["PivData", np.ndarray]]
PoseMeasurement = Dict[str, Union["PoseMeasurement", float]]
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
