"""Define custom type alias."""
from typing import Any, Dict, Union

import numpy as np
import trimesh

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
