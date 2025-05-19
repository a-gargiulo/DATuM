"""Extract 1D profiles from the 2D stereo PIV BeVERLI Hill data.

Important notes:
----------------
1. The profiler requires pre-processed PIV data. Run the pre-processor
before attempting to extract profiles.

2. The 1D profiles can be extracted either perpendicular to the VT SWT's (port)
wall in the global Cartesian coordinate system reported for the BeVERLI Hill
experiments, or locally perpendicular to the hill's surface in a local
Cartesian coordinate system, whose x1-axis is aligned with the surface shear
stress direction.

3. The 'shear' coordinate system extraction is (currently) only available for
symmetric hill orientations along the hill centerline, due to the lack of
surface shear stress direction information in the PIV experiment.

4. The profiler includes the calculation of reference conditions, which is tied
to reference locations of measurement sensors (e.g, pressure sensors) in
the VT SWT that are specific to a BeVERLI Hill experimental campaign. The
present code (as of 05/15/25) implements the reference locations for the
BeVERLI Hill wind tunnel entries labeled 2 and 3,
executed between the years 2019 and 2022.

5. The profiler is also capable of calculating integral boundary layer
parameters when operated in 'shear' mode. The calculation requires (centerline)
surface pressure data on the port wall and the BeVERLI Hill in the VT SWT.

6. The profiler additionally estimates various uncertainty sources in the data.
"""
from typing import cast, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from scipy.interpolate import griddata

# from . import (boundary_layer, cfd, log, parser, plotting, preprocessor,
# reference, spalding, transformations, uncertainty, utility)
from datum.core import (
    boundary_layer,
    cfd,
    plotting,
    properties,
    spalding,
    uncertainty,
)
from datum.core.transform import rotation
from datum.utility import apputils
from datum.core.my_types import (
    CFDRefConditions,
    NormalizedRotationTensor,
    PRInputs,
    Profile,
    ProfileCoordinates,
    ProfileData,
    ProfileMeanVelocity,
    ProfileProperties,
    ProfileReynoldsStress,
    ProfileTurbulenceScales,
    Properties,
    RotationTensor,
    StrainTensor,
    Uncertainty,
)

if TYPE_CHECKING:
    from datum.core.piv import Piv
    from datum.core.beverli import Beverli


PROFILES_OUTFILE = "./outputs/profiles.pkl"


def extract_profiles(
    piv: "Piv",
    piv_interp: Optional["Piv"],
    geometry: "Beverli",
    ui: PRInputs
) -> None:
    """
    Extract 1D profiles from 2D PIV data.

    :param piv: Non-interpolated PIV data.
    :param piv_interp: Interpolated PIV data.
    :param geometry: BeVERLI Hill geometry.
    :param ui: User inputs from the GUI.
    """
    try:
        props = properties.calculate(ui)

        reynolds = ui["reynolds_number"]
        gamma = props["fluid"]["heat_capacity_ratio"]
        R = props["fluid"]["gas_constant"]

        cfd_ref = None
        if ui["add_cfd"]:
            cfd.load_fluent_data(
                case_file=cast(str, ui["fluent_case"]),
                data_file=cast(str, ui["fluent_data"]),
                connected=False,
            )
            cfd_ref = cfd.calculate_ref(reynolds, gamma, R)
            cfd.calculate_qois_and_normalize(cfd_ref, False)

        profile_locations = _select_profile_locations(
            piv, ui["number_of_profiles"], geometry
        )

        profiles: Dict[str, Profile] = {}
        profile_number = 0
        for loc in profile_locations:
            profile_number += 1
            profiles[f"p{profile_number}"] = _extract_profile(
                piv,
                piv_interp,
                geometry,
                loc,
                props,
                ui,
                cfd_ref,
            )

        apputils.write_pickle(PROFILES_OUTFILE, profiles)
    except Exception as e:
        raise RuntimeError(f"Profile extraction failed: {e}")


def _extract_profile(
    piv: "Piv",
    piv_interp: Optional["Piv"],
    geometry: "Beverli",
    loc: Tuple[float, float],
    props: Properties,
    ui: PRInputs,
    cfd_ref: Optional[CFDRefConditions] = None,
) -> Profile:
    """Extract a single 1D profile from 2D PIV data.

    :param piv: Non-interpolated PIV data.
    :param piv_interp: Interpolated PIV data.
    :param geometry: BeVERLI Hill geometry.
    :param loc: Profile location.
    :param props: Fluid, flow, and reference properties.
    :param ui: User inputs from the GUI.
    :param cfd_ref: CFD reference conditions.

    :raises RuntimeError: If the profile extraction fails at any point.
    :return: Profile data.
    :rtype: Profile
    """
    # ---------- PREPARATION ----------
    def tensor_field(prefix: str) -> List[str]:
        return [f"{prefix}_{i+1}{j+1}" for i in range(3) for j in range(3)]

    is_shear = True if ui["coordinate_system"] == "Shear" else False

    density = props["fluid"]["density"]
    dyn_viscosity = props["fluid"]["dynamic_viscosity"]
    kin_viscosity = dyn_viscosity / density

    nvec, surf, coords = _get_nvec_surf_and_coords(loc, piv, geometry, ui)
    x_1_m, x_2_m, x_3_m = surf
    x1q, x2q = coords

    data_to_profile = [
        ("mean_velocity", ["U", "V", "W"]),
        ("reynolds_stress", ["UU", "VV", "WW", "UV", "UW", "VW"]),
    ]
    gradients = [
        ("strain_tensor", tensor_field("S")),
        ("rotation_tensor", tensor_field("W")),
        ("normalized_rotation_tensor", tensor_field("O")),
        ("turbulence_scales", ["NUT"]),
    ]
    if ui["add_gradients"]:
        data_to_profile += gradients

    profile: Profile = profile_init(ui["add_cfd"], ui["add_gradients"])

    # ---------- GET EXP PROFILE DATA ----------
    # Properties
    profile["exp"]["properties"]["RHO"] = density
    profile["exp"]["properties"]["NU"] = kin_viscosity
    profile["exp"]["properties"]["U_INF"] = props["flow"]["U_inf"]
    profile["exp"]["properties"]["U_REF"] = props["reference"]["U_ref"]

    # Coordinates (local)
    if is_shear:
        profile["exp"]["coordinates"]["Y_SS"] = np.sqrt(
            (x1q - x_1_m) ** 2 + (x2q - x_2_m) ** 2
        )
    else:
        profile["exp"]["coordinates"]["Y_W"] = x2q - x_2_m

    # Coordinates (global)

    profile["exp"]["coordinates"]["X"] = x1q
    profile["exp"]["coordinates"]["Y"] = x2q

    # Other quantities
    gradient_keys = {q for q, _ in gradients}
    for q, components in data_to_profile:
        source = cast("Piv", piv_interp) if q in gradient_keys else piv
        xx = source.data["coordinates"]["X"].flatten()
        yy = source.data["coordinates"]["Y"].flatten()

        for c in components:
            profile["exp"][q][c] = griddata(
                points=(xx, yy),
                values=source.data[q][c].flatten(),
                xi=(x1q, x2q),
                method="linear",
                rescale=True,
            )

    # ---------- GET CFD PROFILE DATA ----------
    #
    # Adds the following data to 'cfd':
    #       - NU, RHO, U_REF, U_TAU
    #       - X, Y; optionally: Y_SS, Y_SS_PLUS
    #       - U, V, W; optionally: {U, V, W}_SS, {U, V, W}_SS_PLUS
    #       - NUT
    #       - Optionally:
    #         UU, VV, WW, UV, UW, VW
    #         {UU, VV, WW, UV, UW, VW}_SS
    #         {UU, VV, WW, UV, UW, VW}_SS_PLUS
    if ui["add_cfd"]:
        sol = cfd.extract_normal_profile(
            profile_location=(x_1_m, x_2_m, x_3_m),
            number_of_profile_points=500,
            profile_height=0.2,
            use_sigmoid=True,
            system_type=ui["coordinate_system"],
            reference_conditions=cast(CFDRefConditions, cfd_ref),
            reynolds_stress_available=False,
        )
        profile["cfd"], _ = sol

    # ---------- ROTATE PROFILE TO SS ----------
    #
    # Adds the following properties to 'exp' data:
    #
    #       - U_SS, V_SS, W_SS
    #       - UU_SS, VV_SS, WW_SS, UV_SS, UW_SS, VW_SS
    #
    # Optionally:
    #
    #       - {S, W, O}_ij_SS
    if is_shear:
        phi_ss = np.arccos(nvec @ np.array([0, 1, 0]))
        phi_ss = float(phi_ss)
        if x_1_m < 0:
            phi_ss *= -1
        phi_ss_deg = phi_ss * 180.0 * np.pi

        rotmat_ss = rotation.get_rotation_matrix(phi_ss * 180.0 / np.pi, 'z')
        rotdat = rotation.rotate_profile(
            profile["exp"], rotmat_ss, ui["add_gradients"]
        )
        #
        rotation.set_rotated_profile_shear(profile["exp"], rotdat)
        profile["exp"]["properties"]["ANGLE_SS_DEG"] = phi_ss_deg

        # ---------- SPALDING FIT --> U_TAU, Y_0 ----------
        #
        # Adds the following properties to 'exp' data:
        #
        #       - U_TAU
        #       - Y_SS_CORRECTION - same as y_0; in SS coordinates
        #       - X_CORRECTION, Y_CORRECTION - y_0 in global coordinates
        spalding.spalding_fit_profile(profile, add_cfd=ui["add_cfd"])

        # ---------- BL PARAMETERS  ----------
        # Adds the following quantities to 'exp' data
        #
        #       - U_SS_MODELED, Y_SS_MODELED
        #       - BL_PARAMS
        boundary_layer.calculate_boundary_layer_integral_parameters(profile["exp"], ui, props)

    # ---------- UNCERTAINTY ----------
    # Extract uncertainties due to random sampling and profile rotation
    # The module currently selectively extracts uncertainties based on
    # the coordinate system of choice, i.e., if 'Shear' is selected, only the
    # uncertainties of the profile in SS coordinates will be considered, and
    # vice versa for 'Tunnel'.
    #
    # The prefactors used in the module come from error propagation, whose
    # derivation was performed in a separate code by the author. It considers
    # the uncertainty across two INDEPENDENT rotations, the first placing the 
    # local PIV data to a global frame, and the second rotating the profile to SS
    # coordinates. The resulting factor, hence, for the SS case is sqrt(2). That
    # also assumes that the std. in the uncertain rotation angles is the same.
    #
    # TODO: It would be nice to add some bias error too.
    # TODO: Rotation uncertainty could be fancyfied.
    if is_shear:
        uq_obj = cast("Piv", piv_interp)
    else:
        uq_obj = piv
    uncertainty.calculate_random_and_rotation_uncertainty(
        uq_obj, profile["exp"], n_eff=1000, coordinate_system_type=ui["coordinate_system"]
    )

    return profile


def _select_profile_locations(
    piv: "Piv", num_profiles: int, geometry: "Beverli",
) -> Tuple[Tuple[float, float], ...]:
    x_1_m = piv.pose.glob[0]
    x_2_m = piv.pose.glob[1]
    x_3_m = piv.pose.glob[2]

    settings = {
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

    return plotting.points_selector(
        num_profiles,
        piv.data["coordinates"],
        piv.data["mean_velocity"]["U"],
        settings,
        geometry
    )


def _get_nvec_surf_and_coords(
    loc: Tuple[float, float],
    piv: "Piv",
    geometry: "Beverli",
    ui: PRInputs
) -> Tuple[np.ndarray, Tuple[float, float, float], Tuple[np.ndarray, np.ndarray]]:
    x_1_m = loc[0]
    if ui["coordinate_system"] == "Shear":
        x_3_m = 0  # centerline profiles only
        nvec = geometry.get_surface_normal_symmetric_hill_centerline(x_1_m)
    else:
        x_3_m = piv.pose.glob[2]
        nvec = np.array([0, 1, 0])
    x_2_m = geometry.probe_hill(x_1_m, x_3_m)

    surf = (x_1_m, x_2_m, x_3_m)

    x1q = np.linspace(
        x_1_m,
        x_1_m + nvec[0] * ui["profile_height"],
        ui["number_of_profile_pts"],
    )
    x2q = np.linspace(
        x_2_m,
        x_2_m + nvec[1] * ui["profile_height"],
        ui["number_of_profile_pts"],
    )

    coords = (x1q, x2q)

    return nvec, surf, coords


def profile_init(add_cfd: bool, add_gradients: bool) -> Profile:
    """Initialize empty profile data.

    :param add_cfd: Bool for cfd data.
    """
    def empty() -> np.ndarray:
        return np.empty((0,))

    def coordinates_init() -> ProfileCoordinates:
        return {
            "X": empty(),
            "Y": empty(),
            "Y_SS": None,
            "Y_SS_PLUS": None,
            "Y_SS_MODELED": None,
            "Y_W": None,
        }

    def mean_velocity_init() -> ProfileMeanVelocity:
        return {
            "U": empty(),
            "V": empty(),
            "W": empty(),
            "U_SS": None,
            "V_SS": None,
            "W_SS": None,
            "U_SS_PLUS": None,
            "V_SS_PLUS": None,
            "W_SS_PLUS": None,
            "U_SS_MODELED": None,
        }

    def reynolds_stress_init() -> ProfileReynoldsStress:
        return {
            "UU": empty(),
            "VV": empty(),
            "WW": empty(),
            "UV": empty(),
            "UW": empty(),
            "VW": empty(),
            "UU_SS": None,
            "VV_SS": None,
            "WW_SS": None,
            "UV_SS": None,
            "UW_SS": None,
            "VW_SS": None,
            "UU_SS_PLUS": None,
            "VV_SS_PLUS": None,
            "WW_SS_PLUS": None,
            "UV_SS_PLUS": None,
            "UW_SS_PLUS": None,
            "VW_SS_PLUS": None,
        }

    def strain_tensor_init() -> StrainTensor:
        return {
            "S11": empty(),
            "S12": empty(),
            "S13": empty(),
            "S21": empty(),
            "S22": empty(),
            "S23": empty(),
            "S31": empty(),
            "S32": empty(),
            "S33": empty(),
        }

    def rotation_tensor_init() -> RotationTensor:
        return {
            "W11": empty(),
            "W12": empty(),
            "W13": empty(),
            "W21": empty(),
            "W22": empty(),
            "W23": empty(),
            "W31": empty(),
            "W32": empty(),
            "W33": empty(),
        }

    def norm_rotation_tensor_init() -> NormalizedRotationTensor:
        return {
            "O11": empty(),
            "O12": empty(),
            "O13": empty(),
            "O21": empty(),
            "O22": empty(),
            "O23": empty(),
            "O31": empty(),
            "O32": empty(),
            "O33": empty(),
        }

    def turbulence_scales_init() -> ProfileTurbulenceScales:
        return {
            "NUT": empty(),
        }

    def uncertainty_init() -> Uncertainty:
        return {
            "dU": None,
            "dV": None,
            "dW": None,
            "dU_SS": None,
            "dV_SS": None,
            "dW_SS": None,
            "dUU": None,
            "dVV": None,
            "dWW": None,
            "dUV": None,
            "dUW": None,
            "dVW": None,
            "dUU_SS": None,
            "dVV_SS": None,
            "dWW_SS": None,
            "dUV_SS": None,
            "dUW_SS": None,
            "dVW_SS": None,
        }

    def properties_init() -> ProfileProperties:
        return {
            "NU": 0.0,
            "RHO": 0.0,
            "U_REF": 0.0,
            "U_INF": None,
            "U_TAU": None,
            "X_CORRECTION": None,
            "Y_CORRECTION": None,
            "Y_SS_CORRECTION": None,
            "ANGLE_SS_DEG": None,
            "BL_PARAMS": {
                "GRIFFIN": {
                    "DELTA": 0.0,
                    "U_E": 0.0,
                    "DELTA_STAR": 0.0,
                    "THETA": 0.0,
                },
                "VINUESA": {
                    "DELTA": 0.0,
                    "U_E": 0.0,
                    "DELTA_STAR": 0.0,
                    "THETA": 0.0,
                },
            } if not add_cfd else None
        }

    return {
        "exp": {
            "coordinates": coordinates_init(),
            "mean_velocity": mean_velocity_init(),
            "reynolds_stress": reynolds_stress_init(),
            "strain_tensor": strain_tensor_init() if add_gradients else None,
            "rotation_tensor": rotation_tensor_init() if add_gradients else None,
            "normalized_rotation_tensor": norm_rotation_tensor_init() if add_gradients else None,
            "turbulence_scales": turbulence_scales_init() if add_gradients else None,
            "uncertainty": uncertainty_init(),
            "properties": properties_init(),
        },
        "cfd": {
            "coordinates": coordinates_init(),
            "mean_velocity": mean_velocity_init(),
            "reynolds_stress": None,
            "strain_tensor": None,
            "rotation_tensor": None,
            "normalized_rotation_tensor": None,
            "turbulence_scales": turbulence_scales_init(),
            "uncertainty": None,
            "properties": properties_init(),
        } if add_cfd else None
    }
