"""Pose module."""
import numpy as np
from typing import Any, cast, Dict, List, Optional, Tuple, Type, TypeVar, Literal, Union
from .beverli import Beverli
from ..utility import apputils, mathutils
from .my_types import PoseMeasurement

from scipy import integrate, interpolate

T = TypeVar("T")


class Pose:
    """
    This class defines a container for the pose data.

    It further provides functionalities to calculate and manipulate the pose.
    """

    def __init__(self, angle1: float = 0.0, angle2: float = 0.0, loc: List[float] = [0.0, 0.0], glob: List[float] = [0.0, 0.0, 0.0]):
        """Initialize a PIV plane pose."""
        self.angle1 = angle1
        self.angle2 = angle2
        self.loc = loc
        self.glob = glob

    def calculate_global_pose(
        self, geometry: Beverli, meas_path: str, opts: Dict[str, bool]
    ) -> Optional[List[float]]:
        """Calculate the global pose of the PIV plane."""
        measurement = apputils.read_json(meas_path)
        if measurement is None:
            return None

        # CHECKS ENTIRE MEASUREMENT FILE
        if not Pose._check_pose_measurement(measurement):
            print("ISSUE WITH POSE MEASUREMENT.")
            return None

        x3_profile = cast(float, cast(dict, measurement["calibration_plate_location"])["x_3"])
        x1_profile, x2_profile = geometry.calculate_x1_x2(x3_profile)

        x2_prime_profile = mathutils.calculate_derivative_1d(x1_profile, x2_profile)

        secant_tangent_parameters = Pose._obtain_secant_tangent_parameters(
            x1_profile, x2_profile, x2_prime_profile, measurement
        )

        secant_tangent_parameters = Pose._correct_secant_tangent_parameters(
            secant_tangent_parameters, x1_profile, x2_profile, measurement, opts
        )

        secant_tangent_parameters.append(x3_profile)
        # secant_tangent_center_point_x1 = secant_tangent_parameters[4]
        # secant_tangent_center_point_x2 = secant_tangent_parameters[5]
        # secant_tangent_angle_deg = secant_tangent_parameters[6]

        # self.angle = secant_tangent_angle_deg
        # self.glob[0] = secant_tangent_center_point_x1
        # self.glob[1] = secant_tangent_center_point_x2
        # self.glob[2] = x3_profile

        return secant_tangent_parameters

    @staticmethod
    def _obtain_secant_tangent_parameters(
        x1_profile: np.ndarray,
        x2_profile: np.ndarray,
        x2_prime_profile: np.ndarray,
        measurement: PoseMeasurement,
    ) -> List[float]:
        """Obtain the secant parameters for the PIV plane."""
        hill_side = {-1: "windward", 1: "leeward"}

        plate_location = np.sign(cast(float, cast(dict, measurement["calibration_plate_location"])["x_1"]))
        triangulation = cast(dict, cast(dict, measurement["calibration_plate_angle"])["triangulation"])

        plate_corners_arclength_coordinates = [
            cast(float, triangulation["upstream_plate_corner_arclength_position"]),
            cast(float, triangulation["downstream_plate_corner_arclength_position"]),
        ]

        # Get secant parameters
        is_windward = hill_side[plate_location] == "windward"
        secant_parameters = Pose._calculate_secant_parameters(
            plate_corners_arclength_coordinates,
            x1_profile,
            x2_profile,
            x2_prime_profile,
            is_windward,
        )

        return secant_parameters

    @staticmethod
    def _calculate_secant_parameters(
        plate_corners_arclength_coordinates: List[float],
        x1_profile: np.ndarray,
        x2_profile: np.ndarray,
        x2_prime_profile: np.ndarray,
        is_windward: bool,
    ) -> List[float]:
        calplate_width = 0.106

        if is_windward:
            hill_side_indices = np.flipud(np.where(x1_profile <= 0)[0])
        else:
            hill_side_indices = np.where(x1_profile >= 0)[0]

        hill_side_arclength_coordinates = integrate.cumulative_trapezoid(
            np.sqrt(1 + x2_prime_profile[hill_side_indices] ** 2),
            x1_profile[hill_side_indices],
        )

        if is_windward:
            corner_condition_1 = plate_corners_arclength_coordinates[1]
            corner_condition_2 = plate_corners_arclength_coordinates[0]
            delta_sign = 1
        else:
            corner_condition_1 = plate_corners_arclength_coordinates[0]
            corner_condition_2 = plate_corners_arclength_coordinates[1]
            delta_sign = -1

        # Cartesian coordinates of corners:
        # Corner 1 is upstream for the windward and downstream for the leeward case, respectively.
        # Corner 2 is downstream for the windward and upstream for the leeward case, respectively.
        corner_2_idx = np.where(hill_side_arclength_coordinates * delta_sign <= corner_condition_1 * delta_sign)[0][0]
        corner_2_x1 = x1_profile[hill_side_indices][corner_2_idx]
        corner_2_x2 = x2_profile[hill_side_indices][corner_2_idx]

        if (
            (plate_corners_arclength_coordinates[0] < hill_side_arclength_coordinates[-1] and is_windward)
            or
            (plate_corners_arclength_coordinates[1] > hill_side_arclength_coordinates[-1] and not is_windward)
        ):
            delta_x1 = delta_sign * np.sqrt(calplate_width**2 - corner_2_x2**2)
            corner_1_x1 = corner_2_x1 - delta_x1
            corner_1_x2 = 0
        else:
            corner_1_idx = np.where(
                hill_side_arclength_coordinates * delta_sign <= corner_condition_2 * delta_sign
            )[0][0]
            corner_1_x1 = x1_profile[hill_side_indices][corner_1_idx]
            corner_1_x2 = x2_profile[hill_side_indices][corner_1_idx]

        # Center point coordinates
        center_x1 = (corner_1_x1 + corner_2_x1) / 2.0
        center_x2 = (corner_1_x2 + corner_2_x2) / 2.0

        # Inclination angle
        calplate_angle_deg = 180 / np.pi * np.arctan2((corner_2_x2 - corner_1_x2), (corner_2_x1 - corner_1_x1))

        if not is_windward:
            calplate_angle_deg -= 180

        secant_parameters = [
            corner_1_x1 if is_windward else corner_2_x1,
            corner_1_x2 if is_windward else corner_2_x2,
            corner_2_x1 if is_windward else corner_1_x1,
            corner_2_x2 if is_windward else corner_1_x2,
            center_x1,
            center_x2,
            calplate_angle_deg,
        ]

        return secant_parameters

    @staticmethod
    def _correct_secant_tangent_parameters(
        secant_parameters: List[float],
        x1_profile: np.ndarray,
        x2_profile: np.ndarray,
        measurement: PoseMeasurement,
        opts: Dict[str, bool],
    ) -> List[float]:
        # Initialization
        calplate_width = 0.106
        measured_angle_deg = float(cast(dict, measurement["calibration_plate_angle"])["direct_measurement"])

        hill_side = np.sign(cast(float, cast(dict, measurement["calibration_plate_location"])["x_1"]))
        hill_prof_interpolant = interpolate.interp1d(x1_profile, x2_profile, kind="linear")
        is_on_convex_curvature = opts["apply_convex_curvature_correction"]
        use_manual_angle = opts["use_measured_rotation_angle"]

        # Use manual angle
        if use_manual_angle:
            secant_parameters[6] = -hill_side * measured_angle_deg
            secant_parameters[2] = secant_parameters[4] + (calplate_width / 2) * np.cos(
                secant_parameters[6] * np.pi / 180
            )
            secant_parameters[3] = secant_parameters[5] + (calplate_width / 2) * np.sin(
                secant_parameters[6] * np.pi / 180
            )
            secant_parameters[0] = secant_parameters[4] - (calplate_width / 2) * np.cos(
                secant_parameters[6] * np.pi / 180
            )
            secant_parameters[1] = secant_parameters[5] - (calplate_width / 2) * np.sin(
                secant_parameters[6] * np.pi / 180
            )

            return secant_parameters

        # PIV plane is on convex curvature
        if is_on_convex_curvature:
            secant_parameters[2] = secant_parameters[0] + calplate_width * np.cos(
                measured_angle_deg * np.pi / 180
            )
            secant_parameters[3] = secant_parameters[1] - hill_side * calplate_width * np.sin(
                measured_angle_deg * np.pi / 180
            )
            secant_parameters[4] = (secant_parameters[0] + secant_parameters[2]) / 2
            secant_parameters[5] = (secant_parameters[1] + secant_parameters[3]) / 2
            secant_parameters[6] = -hill_side * measured_angle_deg

            return secant_parameters

        # Calculated secant center point is invalid
        if secant_parameters[5] < hill_prof_interpolant(secant_parameters[4]):
            delta_x2 = hill_prof_interpolant(secant_parameters[4]) - secant_parameters[5]
            secant_parameters[1] = secant_parameters[1] + delta_x2
            secant_parameters[3] = secant_parameters[3] + delta_x2
            secant_parameters[5] = secant_parameters[5] + delta_x2

            return secant_parameters

        return secant_parameters

    @staticmethod
    def _check_pose_measurement(measurement: PoseMeasurement) -> bool:
        keys = [
            ("calibration_plate_angle", dict, ["calibration_plate_angle"]),
            ("direct_measurement", float, ["calibration_plate_angle", "direct_measurement"]),
            ("triangulation", dict, ["calibration_plate_angle", "triangulation"]),
            (
                "upstream_plate_corner_arclength_position",
                float,
                ["calibration_plate_angle", "triangulation", "upstream_plate_corner_arclength_position"]
            ),
            (
                "downstream_plate_corner_arclength_position",
                float,
                ["calibration_plate_angle", "triangulation", "downstream_plate_corner_arclength_position"]
            ),
            ("calibration_plate_location", dict, ["calibration_plate_location"]),
            ("x_1", float, ["calibration_plate_location", "x_1"]),
            ("x_3", float, ["calibration_plate_location", "x_3"]),
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
