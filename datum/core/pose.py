"""Pose module."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import integrate, interpolate

from ..utility import apputils, mathutils
from .beverli import Beverli
from .my_types import PoseMeasurement, SecParams


class Pose:
    """
    Define a container to hold the pose data.

    Includes functionalities to calculate and manipulate pose data.
    """

    def __init__(
        self,
        angle1: float = 0.0,
        angle2: float = 0.0,
        loc: Tuple[float, float] = (0.0, 0.0),
        glob: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """Initialize the pose parameters of the PIV plane.

        :param angle1: First rotation angle.
        :param angle2: Second rotation angle.
        :param loc: Local pose parameters.
        :param glob: Global pose data parameters.
        """
        self.angle1 = angle1
        self.angle2 = angle2
        self.loc = loc
        self.glob = glob

    def calculate_global_pose(
        self, geometry: Beverli, meas_path: str, opts: Dict[str, bool]
    ) -> Optional[SecParams]:
        """Calculate the global pose.

        :param geometry: BeVERLI Hill geometry.
        :param meas_path: System path to the pose measurement file.
        :param opts: User input options.
        """
        measurement = apputils.load_pose_measurement(meas_path)
        if measurement is None:
            return None

        x3_profile = measurement["calibration_plate_location"]["x_3_m"]

        x1_profile, x2_profile = geometry.calculate_x1_x2(x3_profile)

        x2_prime_profile = mathutils.calculate_derivative_1d(
            x1_profile, x2_profile
        )

        secant_tangent_parameters = Pose._obtain_secant_tangent_parameters(
            x1_profile, x2_profile, x2_prime_profile, measurement
        )

        secant_tangent_parameters = Pose._correct_secant_tangent_parameters(
            secant_tangent_parameters,
            x1_profile,
            x2_profile,
            measurement,
            opts,
        )

        secant_tangent_parameters.append(x3_profile)

        return (
            secant_tangent_parameters[0],
            secant_tangent_parameters[1],
            secant_tangent_parameters[2],
            secant_tangent_parameters[3],
            secant_tangent_parameters[4],
            secant_tangent_parameters[5],
            secant_tangent_parameters[6],
            secant_tangent_parameters[7],
        )

    @staticmethod
    def _obtain_secant_tangent_parameters(
        x1_profile: np.ndarray,
        x2_profile: np.ndarray,
        x2_prime_profile: np.ndarray,
        measurement: PoseMeasurement,
    ) -> List[float]:
        hill_side = {-1: "windward", 1: "leeward"}

        plate_location = np.sign(
            measurement["calibration_plate_location"]["x_1_m"]
        )
        triangulation = measurement["calibration_plate_angle"]["triangulation"]

        plate_corners_arclength_coordinates = (
            triangulation["upstream_plate_corner_arclength_position_m"],
            triangulation["downstream_plate_corner_arclength_position_m"],
        )

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
        plate_corners_arclength_coordinates: Tuple[float, float],
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
        # Corner 1 is upstream for the windward and downstream for the leeward
        # case, respectively.
        # Corner 2 is downstream for the windward and upstream for the leeward
        # case, respectively.
        corner_2_idx = np.where(
            hill_side_arclength_coordinates * delta_sign
            <= corner_condition_1 * delta_sign
        )[0][0]
        corner_2_x1 = x1_profile[hill_side_indices][corner_2_idx]
        corner_2_x2 = x2_profile[hill_side_indices][corner_2_idx]

        if (
            plate_corners_arclength_coordinates[0]
            < hill_side_arclength_coordinates[-1]
            and is_windward
        ) or (
            plate_corners_arclength_coordinates[1]
            > hill_side_arclength_coordinates[-1]
            and not is_windward
        ):
            delta_x1 = delta_sign * np.sqrt(calplate_width**2 - corner_2_x2**2)
            corner_1_x1 = corner_2_x1 - delta_x1
            corner_1_x2 = 0
        else:
            corner_1_idx = np.where(
                hill_side_arclength_coordinates * delta_sign
                <= corner_condition_2 * delta_sign
            )[0][0]
            corner_1_x1 = x1_profile[hill_side_indices][corner_1_idx]
            corner_1_x2 = x2_profile[hill_side_indices][corner_1_idx]

        # Center point coordinates
        center_x1 = (corner_1_x1 + corner_2_x1) / 2.0
        center_x2 = (corner_1_x2 + corner_2_x2) / 2.0

        # Inclination angle
        calplate_angle_deg = (
            180
            / np.pi
            * np.arctan2(
                (corner_2_x2 - corner_1_x2), (corner_2_x1 - corner_1_x1)
            )
        )

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
        measured_angle_deg = measurement["calibration_plate_angle"][
            "direct_measurement_deg"
        ]

        hill_side = np.sign(measurement["calibration_plate_location"]["x_1_m"])
        hill_prof_interpolant = interpolate.interp1d(
            x1_profile, x2_profile, kind="linear"
        )
        is_on_convex_curvature = opts["apply_convex_curvature_correction"]
        use_manual_angle = opts["use_measured_rotation_angle"]

        # Use manual angle
        if use_manual_angle:
            secant_parameters[6] = -hill_side * measured_angle_deg
            secant_parameters[2] = secant_parameters[4] + (
                calplate_width / 2
            ) * np.cos(secant_parameters[6] * np.pi / 180)
            secant_parameters[3] = secant_parameters[5] + (
                calplate_width / 2
            ) * np.sin(secant_parameters[6] * np.pi / 180)
            secant_parameters[0] = secant_parameters[4] - (
                calplate_width / 2
            ) * np.cos(secant_parameters[6] * np.pi / 180)
            secant_parameters[1] = secant_parameters[5] - (
                calplate_width / 2
            ) * np.sin(secant_parameters[6] * np.pi / 180)

            return secant_parameters

        # PIV plane is on convex curvature
        if is_on_convex_curvature:
            secant_parameters[2] = secant_parameters[
                0
            ] + calplate_width * np.cos(measured_angle_deg * np.pi / 180)
            secant_parameters[3] = secant_parameters[
                1
            ] - hill_side * calplate_width * np.sin(
                measured_angle_deg * np.pi / 180
            )
            secant_parameters[4] = (
                secant_parameters[0] + secant_parameters[2]
            ) / 2
            secant_parameters[5] = (
                secant_parameters[1] + secant_parameters[3]
            ) / 2
            secant_parameters[6] = -hill_side * measured_angle_deg

            return secant_parameters

        # Calculated secant center point is invalid
        if secant_parameters[5] < hill_prof_interpolant(secant_parameters[4]):
            delta_x2 = (
                hill_prof_interpolant(secant_parameters[4])
                - secant_parameters[5]
            )
            secant_parameters[1] = secant_parameters[1] + delta_x2
            secant_parameters[3] = secant_parameters[3] + delta_x2
            secant_parameters[5] = secant_parameters[5] + delta_x2

            return secant_parameters

        return secant_parameters
