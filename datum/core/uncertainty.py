"""UQ Module."""
from typing import cast, Tuple, TYPE_CHECKING

import numpy as np

from .my_types import ProfileData, Uncertainty, ProfileReynoldsStress

if TYPE_CHECKING:
    from datum.core.piv import Piv


def calculate_random_and_rotation_uncertainty(
    piv: "Piv", profile: ProfileData, n_eff: int, coordinate_system_type: str
):
    prefactor = 1
    if piv.pose.angle1 <= 1:
        prefactor = 0

    velocities = ["dU", "dV", "dW"]
    stresses = ["dUU", "dVV", "dWW", "dUV", "dUW", "dVW"]
    if coordinate_system_type == "Shear":
        velocities = ["dU_SS", "dV_SS", "dW_SS"]
        stresses = ["dUU_SS", "dVV_SS", "dWW_SS", "dUV_SS", "dUW_SS", "dVW_SS"]

    uq_data = cast(Uncertainty, profile["uncertainty"])

    du_rot, dv_rot, dw_rot = velocity_std_due_to_rotation(
        profile, coordinate_system_type
    )
    du_rand, dv_rand, dw_rand = velocity_std_due_to_random(
        profile, n_eff, coordinate_system_type
    )

    (duu_rot, dvv_rot, dww_rot, duv_rot, duw_rot, dvw_rot) = stress_std_due_to_rotation(
        profile, coordinate_system_type
    )
    (
        duu_rand,
        dvv_rand,
        dww_rand,
        duv_rand,
        duw_rand,
        dvw_rand,
    ) = stress_std_due_to_random(profile, n_eff, coordinate_system_type)

    uq_data[velocities[0]] = 1.96 * np.sqrt(
        prefactor * du_rot**2 + du_rand**2
    )
    uq_data[velocities[1]] = 1.96 * np.sqrt(
        prefactor * dv_rot**2 + dv_rand**2
    )
    uq_data[velocities[2]] = 1.96 * np.sqrt(
        prefactor * dw_rot**2 + dw_rand**2
    )

    uq_data[stresses[0]] = 1.96 * np.sqrt(
        prefactor * duu_rot**2 + duu_rand**2
    )
    uq_data[stresses[1]] = 1.96 * np.sqrt(
        prefactor * dvv_rot**2 + dvv_rand**2
    )
    uq_data[stresses[2]] = 1.96 * np.sqrt(
        prefactor * dww_rot**2 + dww_rand**2
    )

    uq_data[stresses[3]] = 1.96 * np.sqrt(
        prefactor * duv_rot**2 + duv_rand**2
    )
    uq_data[stresses[4]] = 1.96 * np.sqrt(
        prefactor * duw_rot**2 + duw_rand**2
    )
    uq_data[stresses[5]] = 1.96 * np.sqrt(
        prefactor * dvw_rot**2 + dvw_rand**2
    )


def velocity_std_due_to_rotation(
    profile: ProfileData, coordinate_system_type: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefactor = 1
    components = ["U", "V", "W"]
    if coordinate_system_type == "Shear":
        prefactor = np.sqrt(2)
        components = ["U_SS", "V_SS", "W_SS"]

    du = prefactor * np.pi * np.sqrt(
        cast(np.ndarray, profile["mean_velocity"][components[1]]) ** 2
    ) / 180
    dv = prefactor * np.pi * np.sqrt(
        cast(np.ndarray, profile["mean_velocity"][components[0]]) ** 2
    ) / 180
    dw = np.zeros_like(
        cast(np.ndarray, profile["mean_velocity"][components[2]])
    )

    return du, dv, dw


def stress_std_due_to_rotation(
    profile: ProfileData, coordinate_system_type: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rs = cast(ProfileReynoldsStress, profile["reynolds_stress"])
    prefactor = 1
    components = ["UU", "VV", "WW", "UV", "UW", "VW"]
    if coordinate_system_type == "Shear":
        prefactor = np.sqrt(2)
        components = ["UU_SS", "VV_SS", "WW_SS", "UV_SS", "UW_SS", "VW_SS"]

    duu = (
        prefactor * np.pi * np.sqrt(
            cast(np.ndarray, rs[components[3]]) ** 2
        ) / 90
    )
    dvv = (
        prefactor * np.pi * np.sqrt(
            cast(np.ndarray, rs[components[3]]) ** 2
        ) / 90
    )
    dww = np.zeros_like(cast(np.ndarray, rs[components[2]]))

    duv = (
        prefactor
        * np.pi
        * np.sqrt(
            (
                cast(np.ndarray, rs[components[0]])
                - cast(np.ndarray, rs[components[1]])
            )
            ** 2
        )
        / 180
    )
    duw = (
        prefactor
        * np.pi
        * np.sqrt(cast(np.ndarray, rs[components[5]]) ** 2)
        / 180
    )
    dvw = (
        prefactor
        * np.pi
        * np.sqrt(cast(np.ndarray, rs[components[4]]) ** 2)
        / 180
    )

    return duu, dvv, dww, duv, duw, dvw


def velocity_std_due_to_random(
    profile: ProfileData, n_eff: int, coordinate_system_type: str
):
    components = ["UU", "VV", "WW"]
    if coordinate_system_type == "Shear":
        components = ["UU_SS", "VV_SS", "WW_SS"]
    rs = cast(ProfileReynoldsStress, profile["reynolds_stress"])
    du = np.sqrt(cast(np.ndarray, rs[components[0]]) / n_eff)
    dv = np.sqrt(cast(np.ndarray, rs[components[1]]) / n_eff)
    dw = np.sqrt(cast(np.ndarray, rs[components[2]]) / n_eff)
    return du, dv, dw


def stress_std_due_to_random(
    profile: ProfileData, n_eff: int, coordinate_system_type: str
):
    components = ["UU", "VV", "WW", "UV", "UW", "VW"]
    if coordinate_system_type == "Shear":
        components = ["UU_SS", "VV_SS", "WW_SS", "UV_SS", "UW_SS", "VW_SS"]

    rs = cast(ProfileReynoldsStress, profile["reynolds_stress"])
    duu = np.sqrt(2 * cast(np.ndarray, rs[components[0]]) ** 2 / n_eff)
    dvv = np.sqrt(2 * cast(np.ndarray, rs[components[1]]) ** 2 / n_eff)
    dww = np.sqrt(2 * cast(np.ndarray, rs[components[2]]) ** 2 / n_eff)

    Ruv = (
        cast(np.ndarray, rs[components[3]])
        / np.sqrt(cast(np.ndarray, rs[components[0]]))
        / np.sqrt(cast(np.ndarray, rs[components[1]]))
    )
    Ruw = (
        cast(np.ndarray, rs[components[4]])
        / np.sqrt(cast(np.ndarray, rs[components[0]]))
        / np.sqrt(cast(np.ndarray, rs[components[2]]))
    )
    Rvw = (
        cast(np.ndarray, rs[components[5]])
        / np.sqrt(cast(np.ndarray, rs[components[1]]))
        / np.sqrt(cast(np.ndarray, rs[components[2]]))
    )

    duv = np.sqrt(
        (1 + Ruv**2)
        * cast(np.ndarray, rs[components[0]])
        * cast(np.ndarray, rs[components[1]])
        / n_eff
    )
    duw = np.sqrt(
        (1 + Ruw**2)
        * cast(np.ndarray, rs[components[0]])
        * cast(np.ndarray, rs[components[2]])
        / n_eff
    )
    dvw = np.sqrt(
        (1 + Rvw**2)
        * cast(np.ndarray, rs[components[1]])
        * cast(np.ndarray, rs[components[2]])
        / n_eff
    )

    return duu, dvv, dww, duv, duw, dvw
