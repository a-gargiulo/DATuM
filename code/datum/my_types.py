"""This module defines and isolates useful type hint aliases, used throughout the DATuM
package."""
from typing import Dict, Union

import numpy as np

# User input data dictionaries
InputData = Dict[
    str,
    Dict[
        str,
        Union[
            bool,
            float,
            int,
            str,
            Dict[
                str,
                Union[bool, float, int, str, Dict[str, Union[bool, float, int, str]]],
            ],
        ],
    ],
]

PoseMeasurement = Dict[
    str, Dict[str, Union[float, int, str, Dict[str, Union[int, float]]]]
]

TransformationParameters = Dict[str, Dict[str, float]]

# Main PIV dictionary containing the BeVERLI stereo PIV data.
PivQuantityComponent = np.ndarray
PivFlowQuantity = Dict[str, PivQuantityComponent]
PivData = Dict[str, Dict[str, Union[PivQuantityComponent, PivFlowQuantity]]]

# General aliases
NestedDict = Dict[
    str, Union[bool, float, int, str, np.ndarray, np.ndarray, "NestedDict"]
]

# Profiles Dict
ProfileDictAll = Dict[str, Dict[str, Dict[str, Dict[str, Union[float, np.ndarray, Dict[str, Dict[str, float]]]]]]]
ProfileDictSingle = Dict[str, Dict[str, Union[float, np.ndarray, Dict[str, Dict[str, float]]]]]
