"""Define custom type alias."""
from typing import Any, Dict, Union

import numpy as np

NestedDict = Dict[str, Union["NestedDict", Any]]
PivData = Dict[str, Union["PivData", np.ndarray]]
