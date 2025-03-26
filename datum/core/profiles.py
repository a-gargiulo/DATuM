"""
Extract 1D profiles from 2D stereo PIV planes.

Profiles are extracted either perpendicular to the VT SWT's wall or perpendicular to the hill's surface in a local
coordinate system aligned with the surface shear stress direction. The latter is only possible along the hill centerline
at symmetric orientations.
"""
import os
import sys
from typing import cast, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import griddata

# from . import (boundary_layer, cfd, log, parser, plotting, preprocessor,
# reference, spalding, transformations, uncertainty, utility)
from . import cfd, plotting, transform, spalding, boundary_layer, uncertainty, expref
from ..utility import apputils
from .beverli import Beverli
from .my_types import Properties
from .piv import Piv


def extract_data(
    piv_obj_no_intrp: Piv,
    piv_obj_intrp: Optional[Piv],
    geometry: Beverli,
    inputs: Dict[str, Union[int, float, str, bool]]
) -> bool:
    """
    Extract the profile data.

    :param piv_obj_no_intrp: Piv object containing non-interpolated data.
    :param piv_obj_intrp: Piv object containing interpolated data.
    :param geometry: An object containing the BeVERLI Hill geometry.
    :param inputs: Dictionary containing the user inputs.
    """
    # TODO: Create a robust type checking mechanism for the 'inputs' dictionary.
    # TODO: Code is currently specific for air as fluid. Make heat capacity ratio and gas constant a user input later.
    properties = expref.calculate_fluid_flow_reference_properties(
        str(inputs["stat_file_path"]),
        1.4,
        287.0,
        float(inputs["reynolds_number"]),
        int(inputs["tunnel_entry_num"])
    )
    if properties is None:
        return False

    cfd_reference_conditions = None
    if inputs["add_cfd"]:
        cfd.load_fluent_data(
            case_file=str(inputs["fluent_case"]),
            data_file=str(inputs["fluent_data"]),
            connected=False,
        )
        cfd_reference_conditions = cfd.calculate_reference_conditions(
            reynolds_number=int(float(inputs["reynolds_number"]) * 1e-3),
            heat_capacity_ratio=properties["fluid"]["heat_capacity_ratio"],
            gas_constant=properties["fluid"]["gas_constant"]
        )
        if cfd_reference_conditions is None:
            return False

        cfd.calculate_qoi_and_normalize_by_reference(
            reference_conditions=cfd_reference_conditions,
            include_reynolds_stress=False,
        )

    # Extract profiles in selected coordinate frame
    profile_locations = _select_profile_locations(piv_obj_no_intrp, int(inputs["number_of_profiles"]), geometry)
    if profile_locations is None:
        return False

    profiles = {}
    profile_number = 0
    for point in profile_locations:
        profile_number += 1
        profiles[f"profile_{profile_number}"] = _extract_normal_profile(
            piv_obj_no_intrp,
            piv_obj_intrp,
            point,
            properties,
            geometry,
            inputs,
            cfd_reference_conditions=cfd_reference_conditions,
        )

    apputils.write_pickle("./outputs/profiles.pkl", profiles)
    return True


def _extract_normal_profile(
    piv_obj: Piv,
    piv_obj_intrp: Optional[Piv],
    point: Tuple[float, float],
    properties: Properties,
    geometry: Beverli,
    inputs: Dict[str, Union[int, float, str, bool]],
    cfd_reference_conditions: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, Dict[str, Dict[str, Union[float, np.ndarray]]]]]:
    density = float(cast(dict, properties["fluid"])["density"])
    dynamic_viscosity = float(cast(dict, properties["fluid"])["dynamic_viscosity"])
    kinematic_viscosity = dynamic_viscosity / density

    # Obtain 3D coordinates and normal vector of the profile
    x_1_m = point[0]
    if inputs["coordinate_system_type"] == "shear":
        x_3_m = 0  # Since `shear` coordinates only available for symmetric orientations along centerline.
        nvec = geometry.get_surface_normal_symmetric_hill_centerline(x_1_m)
    else:
        x_3_m = piv_obj.pose.glob[2]
        nvec = np.array([0, 1, 0])
    x_2_m = geometry.probe_hill(x_1_m, x_3_m)

    # Extract surface-normal profiles
    x1q = np.linspace(
        x_1_m,
        x_1_m + nvec[0] * float(inputs["profile_height_m"]),
        int(inputs["number_of_profile_points"]),
    )
    x2q = np.linspace(
        x_2_m,
        x_2_m + nvec[1] * float(inputs["profile_height_m"]),
        int(inputs["number_of_profile_points"]),
    )

    data_to_interpolate = [
        ("mean_velocity", ["U", "V", "W"]),
        ("reynolds_stress", ["UU", "VV", "WW", "UV", "UW", "VW"]),
    ]

    if piv_obj_intrp is not None:
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

    if piv_obj_intrp is not None:
        profile["exp"]["strain_tensor"] = {}
        profile["exp"]["rotation_tensor"] = {}
        profile["exp"]["normalized_rotation_tensor"] = {}
        profile["exp"]["turbulence_scales"] = {}

    # Interpolate data
    if inputs["coordinate_system_type"] == "shear":
        profile["exp"]["coordinates"]["Y_SS"] = np.sqrt(
            (x1q - x_1_m) ** 2 + (x2q - x_2_m) ** 2
        )
    else:
        profile["exp"]["coordinates"]["Y_W"] = x2q - x_2_m

    profile["exp"]["coordinates"]["X"] = x1q
    profile["exp"]["coordinates"]["Y"] = x2q

    # BIIIIIIIIIG BOTTLENECK DUE TO GRIDDATA!!! Can it be sped up much, though?
    if piv_obj.data is None:
        return None

    print("Extracting profile data...", end="")
    for quantity, components in data_to_interpolate:
        for component in components:
            if quantity not in ["strain_tensor", "rotation_tensor", "normalized_rotation_tensor", "turbulence_scales"]:
                profile["exp"][quantity][component] = griddata(
                    points=(
                        cast(dict, piv_obj.data["coordinates"])["X"].flatten(),
                        cast(dict, piv_obj.data["coordinates"])["Y"].flatten(),
                    ),
                    values=cast(dict, piv_obj.data[quantity])[component].flatten(),
                    xi=(x1q, x2q),
                    method="linear",
                    rescale=True,
                )
            else:
                if piv_obj_intrp is None:
                    return None
                if piv_obj_intrp.data is None:
                    return None
                profile["exp"][quantity][component] = griddata(
                    points=(
                        cast(dict, piv_obj_intrp.data["coordinates"])["X"].flatten(),
                        cast(dict, piv_obj_intrp.data["coordinates"])["Y"].flatten(),
                    ),
                    values=cast(dict, piv_obj_intrp.data[quantity])[component].flatten(),
                    xi=(x1q, x2q),
                    method="linear",
                    rescale=True,
                )
    print("DONE!")

    # Rotate profile to shear stress coordinate system
    if inputs["coordinate_system_type"] == "shear":
        print("Rotating shear data", end="")
        tangential_angle_rad = np.arccos(
            np.matmul(np.array([nvec[1], -nvec[0], 0]), np.array([1, 0, 0]))
        )  # angle results always positive here

        # if leeward
        if x_1_m > 0:
            tangential_angle_rad *= -1  # angle becomes positive again in matrix

        rotation_matrix_tangential = transform.get_rotation_matrix(
            -np.rad2deg(tangential_angle_rad), rotation_axis=(0, 0, 1)
        )  # rotate coordinate system counterclockwise

        (
            velocity_vector_rotated,
            re_stress_tensor_rotated,
            strain_tensor_rotated,
            rotation_tensor_rotated,
            normalized_rotation_tensor_rotated
        ) = transform.rotate_profile(profile["exp"], rotation_matrix_tangential)
        transform.set_rotated_profiles(
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
    if inputs["add_cfd"]:
        if cfd_reference_conditions is None:
            return None
        sol = cfd.extract_bl_profile(
            profile_location=(x_1_m, x_2_m, x_3_m),
            number_of_profile_points=500,
            profile_height=0.2,
            use_sigmoid=True,
            system_type=str(inputs["coordinate_system_type"]),
            reference_conditions=cfd_reference_conditions,
            reynolds_stress_available=False,
        )
        if sol is None:
            return None
        profile["cfd"], cfd_ds = sol

    profile["exp"]["properties"]["NU"] = kinematic_viscosity
    profile["exp"]["properties"]["RHO"] = density
    profile["exp"]["properties"]["U_INF"] = properties["flow"]["U_inf"]
    profile["exp"]["properties"]["U_REF"] = properties["flow"]["U_ref"]

    # Run a spalding fit for surface-normal profiles
    if inputs["coordinate_system_type"] == "shear":
        spalding.spalding_fit_profile(
            profile, add_cfd=bool(inputs["add_cfd"])
        )  # updates the mutable properties

        # calculate bl parameters
        boundary_layer.calculate_boundary_layer_integral_parameters(profile["exp"], inputs, properties)

    uncertainty.calculate_random_and_rotation_uncertainty(
        piv_obj_intrp, profile["exp"], n_eff=1000, coordinate_system_type=cast(str, inputs["coordinate_system_type"])
    )

    return profile


def _select_profile_locations(
    piv_obj: Piv, number_of_profiles: int, geometry: Beverli,
) -> Optional[List[Tuple[float, float]]]:
    x_1_m = piv_obj.pose.glob[0]
    x_2_m = piv_obj.pose.glob[1]
    x_3_m = piv_obj.pose.glob[2]

    if piv_obj.data is None:
        return None
    coordinates = cast(Dict[str, np.ndarray], piv_obj.data["coordinates"])
    quantity = cast(np.ndarray, piv_obj.data["mean_velocity"]["U"])

    properties = {
        "colormap": "jet",
        "contour_range": {"start": 0, "end": 25, "num_of_contours": 100},
        "zpos": x_3_m,
        "xlim": [x_1_m - 0.15, x_1_m + 0.15],
        "ylim": [x_2_m - 0.02, x_2_m + 0.02],
        "xmajor_locator": 0.05,
        "ymajor_locator": None,
        "cbar_range": {"start": 0, "end": 25, "num_of_ticks": 6},
        "cbar_label": r"$U_1$ (m/s)",
    }

    return plotting.points_selector(number_of_profiles, coordinates, quantity, properties, geometry)
