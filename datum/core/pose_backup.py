"""Define functions for computing the BeVERLI stereo PIV coordinate transformation parameters."""
import sys
from typing import Tuple, Optional

from ..utility import apputils
from .beverli import Beverli

# import numpy as np
# import scipy.integrate as spintegrate
# import scipy.interpolate as spinterpolate

# from . import log, plotting, utility
# from .beverli import Beverli
# from .cfd import get_shape_of_ijk_ordered_tecplot_file
# from .my_math import compute_derivative_1d
# from .parser import InputFile, PoseFile 
# from .transformations import get_rotation_matrix, rotate_vector_quantity


# @log.log_process("Obtain global pose", "subsub")
# def obtain_global_pose(piv_obj) -> None:
#     """Determine and save the location and orientation of the chosen PIV plane within
#     BeVERLI's global coordinate system.

#     :param piv_obj: Instance of the :py:class:`datum.piv.Piv` class.
#     """
#     input_data = InputFile().data
#     piv_input = input_data["piv_data"]
#     system_input = input_data["system"]

#     pose_measurement = PoseFile().pose_measurement
#     angle_measured_deg = pose_measurement["calibration_plate_angle"][
#         "direct_measurement"
#     ]["angle"]

#     transform_params_file_path = utility.find_file(
#         system_input["piv_plane_data_folder"],
#         piv_input["coordinate_transformation"],
#     )

#     warning_message = (
#         "WARNING:\n"
#         "\t +---------------------------------------------------------------------------+\n"
#         "\t | The PIV plane angle is zero. No computation of the global pose needed.    |\n"
#         "\t | Please ensure the transformation parameter file contains the global pose. |\n"
#         "\t +---------------------------------------------------------------------------+\n\n"
#     )

#     if angle_measured_deg == 0:
#         print(warning_message)
#         input("Press enter to continue...\n\n")
#     else:
#         (
#             x_1_global_ref_m,
#             x_2_global_ref_m,
#             plane_angle_deg,
#         ) = calculate_global_pose()

#         transform_params_updates = {
#             "rotation": {"angle_deg": plane_angle_deg},
#             "translation": {
#                 "x_1_glob_ref_m": x_1_global_ref_m,
#                 "x_2_glob_ref_m": x_2_global_ref_m,
#             },
#         }
#         utility.update_nested_dict(
#             piv_obj.transformation_parameters, transform_params_updates
#         )

#     utility.write_json(transform_params_file_path, piv_obj.transformation_parameters)

# @log.log_process("Obtain local pose", "subsub")
# def obtain_local_pose(piv_obj) -> None:
#     """Calculate/determine the local position of the PIV plane's calibration plate edge
#     adjacent to the surface of the BeVERLI Hill.

#     This is achieved by manually selecting and extracting the location corresponding to
#     the global position of the calibration plate's edge using the corresponding PIV
#     plane's calibration plate image.

#     :param piv_obj: Instance of the :py:class:`datum.piv.Piv` class.
#     """
#     input_data = InputFile().data
#     pose_measurement = PoseFile().pose_measurement

#     cal_img_path = utility.find_file(
#         input_data["system"]["piv_plane_data_folder"],
#         input_data["piv_data"]["calibration_image"],
#     )

#     transform_params_file_path = utility.find_file(
#         input_data["system"]["piv_plane_data_folder"],
#         input_data["piv_data"]["coordinate_transformation"],
#     )

#     cal_img = np.loadtxt(cal_img_path, skiprows=4)
#     dims = get_shape_of_ijk_ordered_tecplot_file(cal_img_path)
#     img_coords_mm = np.array([np.reshape(cal_img[:, i], (dims[1], dims[0])) for i in range(2)])
#     img_vals = np.reshape(cal_img[:, 2], (dims[1], dims[0]))

#     rotation_angle_deg = piv_obj.transformation_parameters["rotation"]["angle_deg"]
#     rotation_matrix = get_rotation_matrix(rotation_angle_deg, (0, 0, 1))
#     img_coords_mm = rotate_vector_quantity(img_coords_mm, rotation_matrix)

#     pts = plotting.local_reference_selector(
#         img_coords_mm[0], img_coords_mm[1], img_vals, pose_measurement
#     )
#     x_1_loc_ref_mm = pts[0][0]
#     x_2_loc_ref_mm = pts[0][1]

#     transform_params_updates = {
#         "translation": {
#             "x_1_loc_ref_mm": x_1_loc_ref_mm,
#             "x_2_loc_ref_mm": x_2_loc_ref_mm,
#         }
#     }
#     utility.update_nested_dict(
#         piv_obj.transformation_parameters, transform_params_updates
#     )
#     utility.write_json(transform_params_file_path, piv_obj.transformation_parameters)


def calculate_global_pose(hill_orientation_deg: float, measurement: str) -> Optional[Tuple[float, float, float]]:
    """Calculate the PIV plane's global pose.

    Calculate the PIV plane's global pose based on measurements collected during the
    experiment.

    :param hill_orientation_deg: Hill orientation measured in degrees.
    :param measurement: File path to the pose measurement.

    :return: Cartesian (x, y) coordinates, measured in meters, of the center point of
        the calibration plate's edge closest to the hill's surface, and the inclination
        angle, measured in degrees.
    """
    hill = Beverli(hill_orientation_deg, "cad")

    pose_measurement = apputils.read_json(measurement)
    if pose_measurement is None:
        return None

    x_3_hill_profile_m = pose_measurement["calibration_plate_location"]["x_3"]
    if isinstance(x_3_hill_profile_m, float):
        x_1_hill_profile_m, x_2_hill_profile_m = hill.compute_x1_x2_profile(x_3_hill_profile_m)
    else:
        print(f"Error: Expected a numeric value, got {type(x_3_hill_profile_m)}")
        return None

    x_2_prime_hill_profile = compute_derivative_1d(
        x_1_hill_profile_m, x_2_hill_profile_m
    )

    # Obtain secant/tangent line parameters
    secant_tangent_parameters = obtain_secant_tangent_parameters(
        x_1_hill_profile_m, x_2_hill_profile_m, x_2_prime_hill_profile
    )

    # Apply corrections
    secant_tangent_parameters = correct_secant_tangent_parameters(
        secant_tangent_parameters, x_1_hill_profile_m, x_2_hill_profile_m
    )

    # Print report of the calculation's result
    log_secant_tangent_calculation_results(secant_tangent_parameters)

    # Generate plot
    plotting.plot_global_pose(
        x_1_hill_profile_m, x_2_hill_profile_m, secant_tangent_parameters
    )

    secant_tangent_center_point_x_1_m = secant_tangent_parameters[4]
    secant_tangent_center_point_x_2_m = secant_tangent_parameters[5]
    secant_tangent_angle_deg = secant_tangent_parameters[6]

    global_pose = (
        secant_tangent_center_point_x_1_m,
        secant_tangent_center_point_x_2_m,
        secant_tangent_angle_deg,
    )

    return global_pose


# def obtain_secant_tangent_parameters(
#     x_1_hill_profile_m: np.ndarray,
#     x_2_hill_profile_m: np.ndarray,
#     x_2_prime_hill_profile: np.ndarray,
# ) -> List[float]:
#     """Obtain the secant parameters for the PIV plane.

#     :param x_1_hill_profile_m: NumPy ndarray of shape (n, ) containing the x:sub:`1`
#         coordinates of the hill's local cross-sectional profile, measured in meters.
#         Here, n equals the number of profile points.
#     :param x_2_hill_profile_m: NumPy ndarray of shape (n, ) containing the profile
#         coordinates (x:sub:`2`) of the hill's local cross-sectional profile, measured
#         in meters. Here, n equals the number of profile points.
#     :param x_2_prime_hill_profile: NumPy ndarray of shape (n, ) containing the first
#         derivative of the hill's local cross-sectional profile. Here, n equals the
#         number of profile points.
#     :return: List of shape (6, ) containing the secant parameters.
#     """
#     pose_measurement = PoseFile().pose_measurement
#     # Initialization
#     hill_side = {-1: "windward", 1: "leeward"}
#     plate_location = np.sign(pose_measurement["calibration_plate_location"]["x_1"])
#     triangulation = pose_measurement["calibration_plate_angle"]["triangulation"]
#     plate_corners_arclength_coordinates = [
#         triangulation["upstream_plate_corner_arclength_position"],
#         triangulation["downstream_plate_corner_arclength_position"],
#     ]

#     # Get secant parameters
#     is_windward = hill_side[plate_location] == "windward"
#     secant_parameters = calculate_secant_parameters(
#         plate_corners_arclength_coordinates,
#         x_1_hill_profile_m,
#         x_2_hill_profile_m,
#         x_2_prime_hill_profile,
#         is_windward,
#     )

#     return secant_parameters


# # pylint: disable=too-many-locals
# def calculate_secant_parameters(
#     plate_corners_arclength_coordinates_m: List[float],
#     x_1_hill_profile_m: np.ndarray,
#     x_2_hill_profile_m: np.ndarray,
#     x_2_prime_profile: np.ndarray,
#     is_windward: bool,
# ) -> List[float]:
#     """Calculate the characteristic secant parameters.

#     :param plate_corners_arclength_coordinates_m: List of floats containing the
#         arclength coordinates of the upstream and downstream corners of the PIV
#         calibration plate's edge that is nearest to the BeVERLI Hill's surface. The
#         coordinates are measured in meters.
#     :param x_1_hill_profile_m: NumPy ndarray of shape (n, ) containing the x:sub:`1`
#         coordinates of the hill's local cross-sectional profile, measured in meters.
#         Here, n equals the number of profile points.
#     :param x_2_hill_profile_m: NumPy ndarray of shape (n, ) containing the profile
#         coordinates (x:sub:`2`) of the hill's local cross-sectional profile, measured
#         in meters. Here, n equals the number of profile points.
#     :param x_2_prime_profile: NumPy ndarray of shape (n, ) containing the first
#         derivative of the hill's local cross-sectional profile. Here, n equals the
#         number of profile points.
#     :param is_windward: Boolean indicating whether the calibration plate is on the
#         windward side.
#     :return: A list of floats containing the characteristic secant parameters,
#         comprising the Cartesian coordinates of the upstream and downstream corners of
#         the PIV calibration plate's edge that is nearest to the BeVERLI Hill's surface,
#         the center point coordinates of the edge, and the calibration plate's
#         inclination angle. All coordinates are measured in meters, whereas the
#         inclination angle is measured in degrees.
#     """
#     calibration_plate_width_m = 0.106

#     if is_windward:
#         hill_side_indices = np.flipud(np.where(x_1_hill_profile_m <= 0)[0])
#     else:
#         hill_side_indices = np.where(x_1_hill_profile_m >= 0)[0]

#     hill_side_arclength_coordinates_m = spintegrate.cumtrapz(
#         np.sqrt(1 + x_2_prime_profile[hill_side_indices] ** 2),
#         x_1_hill_profile_m[hill_side_indices],
#     )

#     if is_windward:
#         corner_condition_1 = plate_corners_arclength_coordinates_m[1]
#         corner_condition_2 = plate_corners_arclength_coordinates_m[0]
#         delta_sign = 1
#     else:
#         corner_condition_1 = plate_corners_arclength_coordinates_m[0]
#         corner_condition_2 = plate_corners_arclength_coordinates_m[1]
#         delta_sign = -1

#     # Cartesian coordinates of corners:
#     # Corner 1 is upstream for the windward and downstream for the leeward case, respectively.
#     # Corner 2 is downstream for the windward and upstream for the leeward case, respectively.
#     corner_2_idx = np.where(
#         hill_side_arclength_coordinates_m * delta_sign
#         <= corner_condition_1 * delta_sign
#     )[0][0]
#     corner_2_x_1_m = x_1_hill_profile_m[hill_side_indices][corner_2_idx]
#     corner_2_x_2_m = x_2_hill_profile_m[hill_side_indices][corner_2_idx]

#     if (
#         plate_corners_arclength_coordinates_m[0] < hill_side_arclength_coordinates_m[-1]
#         and is_windward
#     ) or (
#         plate_corners_arclength_coordinates_m[1] > hill_side_arclength_coordinates_m[-1]
#         and not is_windward
#     ):
#         delta_x_1_m = delta_sign * np.sqrt(
#             calibration_plate_width_m**2 - corner_2_x_2_m**2
#         )
#         corner_1_x_1_m = corner_2_x_1_m - delta_x_1_m
#         corner_1_x_2_m = 0
#     else:
#         corner_1_idx = np.where(
#             hill_side_arclength_coordinates_m * delta_sign
#             <= corner_condition_2 * delta_sign
#         )[0][0]
#         corner_1_x_1_m = x_1_hill_profile_m[hill_side_indices][corner_1_idx]
#         corner_1_x_2_m = x_2_hill_profile_m[hill_side_indices][corner_1_idx]

#     # Center point coordinates
#     center_x_1_m = (corner_1_x_1_m + corner_2_x_1_m) / 2
#     center_x_2_m = (corner_1_x_2_m + corner_2_x_2_m) / 2

#     # Inclination angle
#     calibration_plate_angle_deg = (
#         180
#         / np.pi
#         * np.arctan2(
#             (corner_2_x_2_m - corner_1_x_2_m),
#             (corner_2_x_1_m - corner_1_x_1_m),
#         )
#     )

#     if not is_windward:
#         calibration_plate_angle_deg -= 180

#     secant_parameters = [
#         corner_1_x_1_m if is_windward else corner_2_x_1_m,
#         corner_1_x_2_m if is_windward else corner_2_x_2_m,
#         corner_2_x_1_m if is_windward else corner_1_x_1_m,
#         corner_2_x_2_m if is_windward else corner_1_x_2_m,
#         center_x_1_m,
#         center_x_2_m,
#         calibration_plate_angle_deg,
#     ]

#     return secant_parameters


# def correct_secant_tangent_parameters(
#     secant_parameters: List[float],
#     x_1_hill_profile_m: np.ndarray,
#     x_2_hill_profile_m: np.ndarray,
# ) -> List[float]:
#     """Correct the computed secant parameters if necessary.

#     :param secant_parameters: List of floats representing the computed secant
#         parameters.
#     :param x_1_hill_profile_m: NumPy ndarray of shape (n, ) containing the x:sub:`1`
#         coordinates of the hill's local cross-sectional profile, measured in meters.
#         Here, n equals the number of profile points.
#     :param x_2_hill_profile_m: NumPy ndarray of shape (n, ) containing the profile
#         coordinates (x:sub:`2`) of the hill's local cross-sectional profile, measured
#         in meters. Here, n equals the number of profile points.
#     :return: List of floats representing the corrected secant parameters.
#     """
#     input_data = InputFile().data
#     pose_measurement = PoseFile().pose_measurement

#     # Initialization
#     plate_width_m = 0.106
#     measured_angle_deg = pose_measurement["calibration_plate_angle"][
#         "direct_measurement"
#     ]["angle"]
#     hill_side = np.sign(pose_measurement["calibration_plate_location"]["x_1"])
#     hill_prof_interpolant = spinterpolate.interp1d(
#         x_1_hill_profile_m, x_2_hill_profile_m, kind="linear"
#     )
#     is_on_convex_curvature = input_data["preprocessor"]["coordinate_transformation"][
#         "parameters"
#     ]["apply_convex_curvature_correction"]
#     use_manual_angle = input_data["preprocessor"]["coordinate_transformation"][
#         "parameters"
#     ]["use_measured_rotation_angle"]

#     # Use manual angle
#     if use_manual_angle:
#         secant_parameters[6] = -hill_side * measured_angle_deg
#         secant_parameters[2] = secant_parameters[4] + (plate_width_m / 2) * np.cos(
#             secant_parameters[6] * np.pi / 180
#         )
#         secant_parameters[3] = secant_parameters[5] + (plate_width_m / 2) * np.sin(
#             secant_parameters[6] * np.pi / 180
#         )
#         secant_parameters[0] = secant_parameters[4] - (plate_width_m / 2) * np.cos(
#             secant_parameters[6] * np.pi / 180
#         )
#         secant_parameters[1] = secant_parameters[5] - (plate_width_m / 2) * np.sin(
#             secant_parameters[6] * np.pi / 180
#         )

#         return secant_parameters

#     # PIV plane is on convex curvature
#     if is_on_convex_curvature:
#         secant_parameters[2] = secant_parameters[0] + plate_width_m * np.cos(
#             measured_angle_deg * np.pi / 180
#         )
#         secant_parameters[3] = secant_parameters[
#             1
#         ] - hill_side * plate_width_m * np.sin(measured_angle_deg * np.pi / 180)
#         secant_parameters[4] = (secant_parameters[0] + secant_parameters[2]) / 2
#         secant_parameters[5] = (secant_parameters[1] + secant_parameters[3]) / 2
#         secant_parameters[6] = -hill_side * measured_angle_deg

#         return secant_parameters

#     # Calculated secant center point is invalid
#     if secant_parameters[5] < hill_prof_interpolant(secant_parameters[4]):
#         delta_x_2_m = hill_prof_interpolant(secant_parameters[4]) - secant_parameters[5]
#         secant_parameters[1] = secant_parameters[1] + delta_x_2_m
#         secant_parameters[3] = secant_parameters[3] + delta_x_2_m
#         secant_parameters[5] = secant_parameters[5] + delta_x_2_m

#         return secant_parameters

#     return secant_parameters


# def log_secant_tangent_calculation_results(secant_parameters: List[float]) -> None:
#     """Write a table to the standard output, summarizing the results of the secant
#     parameters computation.

#     :param secant_parameters: List of characteristic secant parameters.
#     """
#     cal_plate_width_calculated = np.sqrt(
#         (secant_parameters[2] - secant_parameters[0]) ** 2
#         + (secant_parameters[3] - secant_parameters[1]) ** 2
#     )

#     print("Calibration Plate Position and Orientation:\n")
#     headers = ["Angle", "Center", "UpStrm Corner", "DownStrm Corner", "Width"]
#     data = [
#         [
#             f"{secant_parameters[6]:.2f} deg",
#             f"{secant_parameters[4]:.3f} m",
#             f"{secant_parameters[0]:.3f} m",
#             f"{secant_parameters[2]:.3f} m",
#             f"{cal_plate_width_calculated:.4f} m",
#         ]
#     ]

#     log.create_ascii_table(headers, data)
