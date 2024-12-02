"""Pose module."""
import numpy as np
from typing import Any, cast, List, Optional, Tuple, Type, TypeVar, Literal, Union
from .beverli import Beverli
from ..utility import apputils, mathutils
from .my_types import PoseMeasurement

T = TypeVar("T")


class Pose:
    """
    This class defines a container for the pose data.

    It further provides functionalities to calculate and manipulate the pose.
    """

    def __init__(self, angle: float = 0.0, loc: List[float] = [0.0, 0.0], glob: List[float] = [0.0, 0.0]):
        """Initialize a PIV plane pose."""
        self.angle = angle
        self.loc = loc
        self.glob = glob

    def calculate_global_pose(self, geometry: Beverli, meas_path: str) -> Optional[Tuple[float, float, float]]:
        measurement = apputils.read_json(meas_path)
        if measurement is None:
            return None

        if not Pose._check_pose_measurement(measurement):
            print("ISSUE WITH POSE MEASUREMENT.")
            return None

        x3_profile = measurement["calibration_plate_location"]["x_3"]
        if isinstance(x3_profile, float):
            x1_profile, x2_profile = geometry.calculate_x1_x2(x3_profile)
        else:
            print(f"[Error]: Expected float, got {type(x3_profile)}.")
            return None

        x2_prime_profile = mathutils.calculate_derivative_1d(x1_profile, x2_profile)
        secant_tangent_parameters = self._obtain_secant_tangent_parameters(x1_profile, x2_profile, x2_prime_profile, measurement)

    def _obtain_secant_tangent_parameters(
        self,
        x1_profile: np.ndarray,
        x2_profile: np.ndarray,
        x2_prime_profile: np.ndarray,
        measurement: PoseMeasurement,
    ) -> Optional[List[float]]:
        """Obtain the secant parameters for the PIV plane."""
        hill_side = {-1: "windward", 1: "leeward"}


        plate_location = np.sign(cast(float, cast(dict, measurement["calibration_plate_location"])["x_1"]))
        triangulation = cast(dict, cast(dict, measurement["calibration_plate_angle"])["triangulation"])

        plate_corners_arclength_coordinates = [
            cast(float, triangulation["upstream_plate_corner_arclength_position"]),
            cast(float, triangulation["downstream_plate_corner_arclength_position"]),
        ]

#         # Get secant parameters
#         is_windward = hill_side[plate_location] == "windward"
#         secant_parameters = calculate_secant_parameters(
#             plate_corners_arclength_coordinates,
#             x_1_hill_profile_m,
#             x_2_hill_profile_m,
#             x_2_prime_hill_profile,
#             is_windward,
#         )

        # return secant_parameters
        return None

    @staticmethod
    def _check_pose_measurement(measurement: PoseMeasurement) -> bool:
        keys = [
            ("calibration_plate_angle", dict, ["calibration_plate_angle"]),
            ("direct_measurement", float, ["calibration_plate_angle", "direct_measurement"]),
            ("triangulation", dict, ["calibration_plate_angle", "triangulation"]),
            (
                "upstream_plate_corner_arclength_position",
                float,
                ["calibration_plate_angle", "upstream_plate_corner_arclength_position"]
            ),
            (
                "downstream_plate_corner_arclength_position",
                float,
                ["calibration_plate_angle", "downstream_plate_corner_arclength_position"]
            ),
            ("calibration_plate_location", dict, ["calibration_plate_location"]),
            ("x_1", float, ["calibration_plate_location", "x_1"]),
            ("x_3", float, ["calibration_plate_location", "x_2"]),
        ]

        keys_ok = all([apputils.search_nested_dict(measurement, key[0]) for key in keys])
        if not keys_ok:
            return False

        types_ok = all(
            [
                Pose._check_measurement_type(
                    Pose._retrieve_from_measurement(measurement, key[2], len(key[2])), key[0], key[1]
                ) for key in keys
            ]
        )

        if not types_ok:
            return False

        return True

    @staticmethod
    def _retrieve_from_measurement(measurement: PoseMeasurement, keys: List[str], lvl: int):
        if lvl == 1:
            return measurement[keys[0]]
        elif lvl == 2:
            return cast(dict, measurement[keys[0]])[keys[1]]
        elif lvl == 3:
            return cast(dict, cast(dict, measurement[keys[0]])[keys[1]])[keys[2]]

    @staticmethod
    def _check_measurement_type(value: Any, name: str, clss: Type[T]) -> bool:
        if not isinstance(value, clss):
            print(f"[Error]: Expected {clss.__name__} type for {name}, got {type(value).__name__}.")
            return False
        return True
