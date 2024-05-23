"""This module contains functions for loading the BeVERLI Hill stereo PIV data."""
import sys

import scipy.io as scio

from .. import log, utility
from ..parser import InputFile
from ..piv import Piv


@log.log_process("Load BeVERLI Hill Stereo PIV data from Matlab files", "main")
def load_matlab_data() -> Piv:
    """
    Load the raw BeVERLI stereo PIV data from Matlab (.mat) files.

    :return: An instance of the :py:class:`datum.piv.Piv` class.
    """
    # Extract relevant input data
    input_data = InputFile().data

    general_input = input_data["general"]
    piv_input = input_data["piv_data"]
    system_input = input_data["system"]

    config = piv_input["configuration"]

    def get_path_to_flow_quantity(suffix: str) -> str:
        plane_number = piv_input["plane_number"]
        plane_type = piv_input["plane_type"]
        reynolds_number = general_input["reynolds_number"]

        dataset = (
            f"plane{plane_number}_"
            f"{int(reynolds_number * 1e-3)}k_"
            f"{plane_type.upper()}_{suffix}.mat"
        )

        return utility.construct_file_path(
            system_input["piv_case_data_folder"], [], dataset
        )

    # Load raw BeVERLI stereo PIV data sets
    mean_velocity = scio.loadmat(get_path_to_flow_quantity("mean_velocity"))
    reynolds_stress = scio.loadmat(get_path_to_flow_quantity("reynolds_stress"))
    instantaneous_velocity_frame = (
        scio.loadmat(get_path_to_flow_quantity("instantaneous_velocity_frame"))
        if config["instantaneous_velocity_frame_available"]
        else None
    )
    turbulence_dissipation = (
        scio.loadmat(get_path_to_flow_quantity("turbulence_dissipation"))
        if config["turbulence_dissipation_available"]
        else None
    )

    # Construct the dictionary holding the BeVERLI stereo PIV data
    flip_u_3 = config["flip_out_of_plane_component"]

    piv_data = {
        "coordinates": {"X": mean_velocity["X"], "Y": mean_velocity["Y"]},
        "mean_velocity": {
            key: (-val if key == "W" and flip_u_3 else val)
            for key, val in mean_velocity.items()
            if key in {"U", "V", "W"}
        },
        "reynolds_stress": {
            key: (-val if key in {"UW", "VW"} and flip_u_3 else val)
            for key, val in reynolds_stress.items()
            if key in {"UU", "VV", "WW", "UV", "UW", "VW"}
        },
        "turbulence_scales": {
            "TKE": 0.5
            * (reynolds_stress["UU"] + reynolds_stress["VV"] + reynolds_stress["WW"])
        },
    }

    if instantaneous_velocity_frame:
        piv_data["instantaneous_velocity_frame"] = {
            key: (-val if key == "W" and flip_u_3 else val)
            for key, val in instantaneous_velocity_frame.items()
            if key in {"U", "V", "W"}
        }

    if turbulence_dissipation:
        piv_data["turbulence_scales"]["epsilon"] = turbulence_dissipation["epsVals"]

    return Piv(data=piv_data)

def load_preprocessed_data() -> Piv:
    """Load the preprocessed BeVERLI stereo PIV data.

    :return: An instance of the :py:class:`datum.piv.Piv` class.
    """
    # Booleans
    input_data = InputFile().data
    plane_folder = input_data["system"]["piv_plane_data_folder"]

    file_name = (
        f"plane{input_data['piv_data']['plane_number']}_"
        f"{int(input_data['general']['reynolds_number'] / 1000.0)}k_"
        f"{input_data['piv_data']['plane_type']}_preprocessed.pkl"
    )

    file_path = utility.find_file(plane_folder, file_name)

    try:
        preprocessed_data = utility.load_pickle(file_path)
    except FileNotFoundError:
        print("No preprocessed data found!")
        sys.exit(1)

    return Piv(data=preprocessed_data)

def load_profiles():
    # Booleans
    input_data = InputFile().data
    plane_folder = input_data["system"]["piv_plane_data_folder"]

    file_name = (
        f"plane{input_data['piv_data']['plane_number']}_"
        f"{int(input_data['general']['reynolds_number'] / 1000.0)}k_"
        f"{input_data['piv_data']['plane_type']}_{input_data['profiles']['coordinate_system_type']}_profiles.pkl"
    )

    file_path = utility.find_file(plane_folder, file_name)

    try:
        profile_data = utility.load_pickle(file_path)
    except FileNotFoundError:
        print("No preprocessed data found!")
        sys.exit(1)

    return profile_data
