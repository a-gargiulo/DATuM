"""Extract 1D profiles from 2D stereo PIV planes.

Profiles can be extracted either perpendicular to the VT SWT's wall or perpendicular to the hill's surface in a local
coordinate system aligned with the surface shear stress direction.
"""
import os
import sys
from typing import cast, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import griddata

# from . import (boundary_layer, cfd, log, parser, plotting, preprocessor,
# reference, spalding, transformations, uncertainty, utility)
from . import cfd, reference
from .beverli import Beverli
from .my_types import NestedDict
from .piv import Piv


def extract_data(piv_obj_no_intrp: Piv, piv_obj_intrp: Piv, opts: Dict[str, Union[int, float, str, bool]]) -> None:
    """Extract profile data from the 2D stereo PIV data."""
    # Obtain reference quantities (if .stat available) for experimental data.
    properties = reference.get_all_plane_properties(opts)

    cfd_reference_conditions = None
    if opts["add_cfd"]:
        cfd.load_fluent_data(
            case_file=cast(str, opts["fluent_case"]),
            data_file=cast(str, opts["fluent_data"]),
            connected=False,
        )
        cfd_reference_conditions = cfd.calculate_reference_conditions(
            reynolds_number=int(cast(float, opts["reynolds_number"]) * 1e-3),
            properties=properties,
        )
        cfd.normalize_variables_by_reference(
            reference_conditions=cfd_reference_conditions,
            calculate_reynolds_stress=False,
        )

    # Extract profiles in selected coordinate frame
    profile_locations = select_profile_locations(piv_obj, number_of_profiles)

    if coordinate_system_type == "shear":
        check_shear_case_computability(
            pose_measurement["calibration_plate_location"]["x_3"],
            input_data["general"]["hill_orientation"],
        )

    profiles = {}
    profile_number = 0
    for point in profile_locations:
        profile_number += 1
        profiles[f"profile_{profile_number}"] = extract_normal_profile(
            piv_obj,
            piv_obj_intrp,
            point,
            number_of_profile_points,
            profile_height_m,
            coordinate_system_type,
            properties,
            cfd_reference_conditions=cfd_reference_conditions,
        )

    utility.write_pickle(output_file_name, profiles)


def check_shear_case_computability(x_3_m: float, hill_orientation_deg: float) -> None:
    """Verify the feasibility of computing profiles perpendicular to the surface of the
    BeVERLI Hill within a Cartesian coordinate frame aligned with the surface shear
    stress direction for the selected PIV plane.

    :param x_3_m: Spanwise Cartesian coordinate of the selected profile location,
        measured in meters.
    :param hill_orientation_deg: The yaw angle orientation of the BeVERLI Hill with
        respect to the oncoming flow of the Virginia Tech Stability Wind Tunnel,
        measured in degrees.
    """
    symmetric_hill_orientations = [0, 90, 180, 270, 45, 135, 225, 315]
    try:
        if (x_3_m != 0) and (hill_orientation_deg in symmetric_hill_orientations):
            raise ValueError(
                "Cannot extract profiles in local shear stress coordinates for "
                "off-center BeVERLI stereo PIV planes at symmetric orientations.\n\n"
                "NOTE:\n"
                "The BeVERLI stereo PIV data currently only exists in x1-x2 planes.\n"
                "Consequently, the data planes only contains the local BeVERLI Hill surface\n"
                "normal when they are located on the hill centerline at symmetric "
                "orientations."
            )

        if hill_orientation_deg not in symmetric_hill_orientations:
            raise ValueError(
                "\n"
                "YIKES! Profile extraction at non-symmetric hill orientations \n"
                "is currently not supported.\n\n"
                "NOTE:\n"
                "The BeVERLI stereo PIV data currently only exists in x1-x2 planes.\n"
                "Consequently, the data planes only contains the local BeVERLI Hill surface\n"
                "normal when they are located on the hill centerline at symmetric "
                "orientations."
            )
    except ValueError as err:
        print(f"ERROR: {err}\n")
        sys.exit(1)


def extract_normal_profile(
    piv_obj,
    piv_obj_intrp,
    point: Tuple[float, float],
    number_of_profile_points: int,
    profile_height_m: float,
    coordinate_system_type: str,
    properties: NestedDict,
    cfd_reference_conditions: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, Dict[str, Union[float, np.ndarray]]]]:
    """Retrieve a profile by extracting it either perpendicular to the surface of the
    BeVERLI Hill within a coordinate frame aligned with the local surface shear stress
    direction, or perpendicular to the wall of the Virginia Tech Stability Wind Tunnel
    where the hill is situated.

    :param piv_obj: An instance of the :py:class:`datum.piv.Piv` class containing the
        BeVERLI stereo PIV data.
    :param piv_obj_intrp: An instance of the :py:class:`datum.piv.Piv` class containing the
        fully processed BeVERLI stereo PIV data.
    :param point: A tuple containing the x:sub:`1` and x:sub:`3` coordinates of the
        profile's hill surface point, measured in meters.
    :param number_of_profile_points: An integer number of points to extract along each
        profile.
    :param profile_height_m: The height of each profile, measured in meters.
    :param coordinate_system_type: A string indicating the desired orientation for
        profile extraction: either normal to the hill surface in local shear stress
        coordinates (`shear`) or normal to the tunnel wall hosting the BeVERLI Hill
        (`tunnel`).
    :param properties: A nested dictionary containing the fluid and flow properties of
        the current PIV plane being analyzed.
    :param cfd_reference_conditions: A dictionary containing the reference conditions
        of the BeVERLI ANSYS Fluent RANS CFD solution.
    :return: A dictionary containing the extracted profile data.
    """
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments
    # Extract user input data
    input_data = InputFile().data
    pose_measurement = PoseFile().pose_measurement

    # Instance of the Beverli class
    hill = Beverli()

    # Extract fluid properties
    density = properties["fluid"]["density"]
    dynamic_viscosity = properties["fluid"]["dynamic_viscosity"]
    kinematic_viscosity = dynamic_viscosity / density

    # Obtain 3D coordinates of the profile
    x_1_m = point[0]
    if coordinate_system_type == "shear":
        x_3_m = 0  # Since `shear` coordinates only available for symmetric orientations along centerline.
        nvec = hill.get_surface_normal_symmetric_hill_centerline(x_1_m)
    else:
        x_3_m = pose_measurement["calibration_plate_location"]["x_3"]
        nvec = np.array([0, 1, 0])
    x_2_m = hill.probe_hill(x_1_m, x_3_m)

    # Extract surface-normal profiles
    x1q = np.linspace(
        x_1_m,
        x_1_m + nvec[0] * profile_height_m,
        number_of_profile_points,
    )
    x2q = np.linspace(
        x_2_m,
        x_2_m + nvec[1] * profile_height_m,
        number_of_profile_points,
    )

    data_to_interpolate = [
        ("mean_velocity", ["U", "V", "W"]),
        ("reynolds_stress", ["UU", "VV", "WW", "UV", "UW", "VW"]),
    ]

    if piv_obj_intrp:
        data_to_interpolate.extend(
            [
                (
                    "strain_tensor",
                    [f"S_{i+1}{j+1}" for i in range(3) for j in range(3)],
                ),
                (
                    "rotation_tensor",
                    [f"W_{i + 1}{j + 1}" for i in range(3) for j in range(3)],
                ),
                (
                    "normalized_rotation_tensor",
                    [f"O_{i + 1}{j + 1}" for i in range(3) for j in range(3)],
                ),
                ("turbulence_scales", ["NUT"]),
            ]
        )

    # Create a dictionary for the profile
    profile = {
        "exp": {
            "coordinates": {},
            "mean_velocity": {},
            "reynolds_stress": {},
            "properties": {},
        }
    }

    if piv_obj_intrp:
        profile["exp"]["strain_tensor"] = {}
        profile["exp"]["rotation_tensor"] = {}
        profile["exp"]["normalized_rotation_tensor"] = {}
        profile["exp"]["turbulence_scales"] = {}

    # Interpolate data
    if coordinate_system_type == "shear":
        profile["exp"]["coordinates"]["Y_SS"] = np.sqrt(
            (x1q - x_1_m) ** 2 + (x2q - x_2_m) ** 2
        )
    else:
        profile["exp"]["coordinates"]["Y_W"] = x2q - x_2_m

    profile["exp"]["coordinates"]["X"] = x1q
    profile["exp"]["coordinates"]["Y"] = x2q

    # BIIIIIIIIIG BOTTLENECK DUE TO GRIDDATA!!! Can it be sped up much, though?
    print("Extracting profile data...", end="")
    for quantity, components in data_to_interpolate:
        for component in components:
            profile["exp"][quantity][component] = griddata(
                points=(
                    piv_obj.data["coordinates"]["X"].flatten(),
                    piv_obj.data["coordinates"]["Y"].flatten(),
                )
                if quantity
                not in [
                    "strain_tensor",
                    "rotation_tensor",
                    "normalized_rotation_tensor",
                    "turbulence_scales",
                ]
                else (
                    piv_obj_intrp.data["coordinates"]["X"].flatten(),
                    piv_obj_intrp.data["coordinates"]["Y"].flatten(),
                ),
                values=piv_obj.data[quantity][component].flatten()
                if quantity
                not in [
                    "strain_tensor",
                    "rotation_tensor",
                    "normalized_rotation_tensor",
                    "turbulence_scales",
                ]
                else piv_obj_intrp.data[quantity][component].flatten(),
                xi=(x1q, x2q),
                method="linear",
                rescale=True,
            )
    print("DONE!")

    # Rotate profile to shear stress coordinate system
    if coordinate_system_type == "shear":
        print("Rotating shear data", end="")
        tangential_angle_rad = np.arccos(
            np.matmul(np.array([nvec[1], -nvec[0], 0]), np.array([1, 0, 0]))
        )  # angle results always positive here

        # if leeward
        if x_1_m > 0:
            tangential_angle_rad *= -1  # angle becomes positive again in matrix

        rotation_matrix_tangential = transformations.get_rotation_matrix(
            -np.rad2deg(tangential_angle_rad), rotation_axis=(0, 0, 1)
        )  # rotate coordinate system counterclockwise

        (
            velocity_vector_rotated,
            re_stress_tensor_rotated,
            strain_tensor_rotated,
            rotation_tensor_rotated,
            normalized_rotation_tensor_rotated
        ) = transformations.rotate_profile(profile["exp"], rotation_matrix_tangential)
        transformations.set_rotated_profiles(
            profile["exp"],
            velocity_vector_rotated,
            re_stress_tensor_rotated,
            strain_tensor_rotated,
            rotation_tensor_rotated,
            normalized_rotation_tensor_rotated
        )
        profile["exp"]["properties"]["ANGLE_SS_DEG"] = np.rad2deg(tangential_angle_rad)
        print("DONE!")

    # Extract equivalent profile from CFD solution
    if input_data["profiles"]["add_cfd"]:
        profile["cfd"] = cfd.extract_bl_profile(
            profile_location=(x_1_m, x_2_m, x_3_m),
            number_of_profile_points=500,
            profile_height=0.2,
            use_sigmoid=True,
            system_type=coordinate_system_type,
            reference_conditions=cfd_reference_conditions,
            reynolds_stress_available=False,
        )

    # Retrieve experimental reference conditions
    stat_file_exists = os.path.exists(
        os.path.join(
            input_data["system"]["piv_plane_data_folder"],
            input_data["general"]["tunnel_conditions"],
        )
    )

    profile["exp"]["properties"]["NU"] = kinematic_viscosity
    profile["exp"]["properties"]["RHO"] = density
    if stat_file_exists:
        profile["exp"]["properties"]["U_INF"] = properties["flow"]["U_inf"]
        profile["exp"]["properties"]["U_REF"] = properties["flow"]["U_ref"]

    # Run a spalding fit for surface-normal profiles
    if coordinate_system_type == "shear":
        spalding.spalding_fit_profile(
            profile, add_cfd=input_data["profiles"]["add_cfd"]
        )  # updates the mutable properties

        # calculate bl parameters
        boundary_layer.calculate_boundary_layer_integral_parameters(profile["exp"])

    uncertainty.calculate_random_and_rotation_uncertainty(
        profile["exp"], n_eff=1000, coordinate_system_type=coordinate_system_type
    )

    return profile


@log.log_process("Select profile points", "sub")
def select_profile_locations(
    piv_obj, number_of_profiles: int
) -> List[Tuple[float, float]]:
    """Select the locations of the profiles to extract from the analyzed stereo PIV
    plane.

    :param piv_obj: AAn instance of the :py:class:`datum.piv.Piv` class, containing the
        BeVERLI stereo PIV data.
    :param number_of_profiles: An integer number of profiles to extract.
    :return: A list of tuples containing the x:sub:`1` and x:sub:`2` Cartesian
        coordinates of the profiles' points on the hill surface.
    """
    pose_measurement = PoseFile().pose_measurement
    x_1_m = pose_measurement["calibration_plate_location"]["x_1"]
    x_3_m = pose_measurement["calibration_plate_location"]["x_3"]

    hill = Beverli()

    coordinates = piv_obj.data["coordinates"]
    quantity = piv_obj.data["mean_velocity"]["U"]
    properties = {
        "colormap": "jet",
        "contour_range": {"start": 0, "end": 25, "num_of_contours": 100},
        "zpos": x_3_m,
        "xlim": [x_1_m - 0.15, x_1_m + 0.15],
        "ylim": [
            hill.probe_hill(x_1_m, x_3_m) - 0.08,
            hill.probe_hill(x_1_m, x_3_m) + 0.08,
        ],
        "xmajor_locator": 0.05,
        "ymajor_locator": None,
        "cbar_range": {"start": 0, "end": 25, "num_of_ticks": 6},
        "cbar_label": r"$U_1$ (m/s)",
    }

    return plotting.point_selector(
        number_of_profiles, coordinates, quantity, properties
    )
