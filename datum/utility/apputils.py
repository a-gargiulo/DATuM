"""Define useful application utility functions."""
import json
import sys
import os
import pickle
from typing import Optional, Dict, Union
import scipy.io as scio
import numpy as np

from datum.utility.logging import logger
from datum.core.pose import Pose
from ..core.my_types import (
    NestedDict,
    RotationParameters,
    TranslationParameters,
    TransformationParameters,
    CalibrationPlateAngle,
    CalibrationPlateLocation,
    Triangulation,
    PoseMeasurement,
    PivData
)


# def construct_file_path(root_folder: str, subfolders: List[str], file_name: str) -> str:
#     """Construct a system specific path to a desired file.

#     :param root_folder: The path of the root folder containing the desired file.
#     :param subfolders: A list of subfolders separating the root folder from the desired
#         file.
#     :param file_name: The desired file's name.
#     :return: A string representing the constructed file path.
#     """
#     root_folder = os.path.normpath(root_folder)

#     file_path = os.path.join(root_folder, *subfolders, file_name)

#     return file_path

def search_nested_dict(dictionary: NestedDict, keyword: str) -> bool:
    """Search a keyword within a nested dictionary.

    :param dictionary: The nested dictionary.
    :param keyword: The keyword to be searched.

    :return: Boolean value indicating, whether the keyword was found or not.
    """
    if isinstance(dictionary, dict):
        if keyword in dictionary:
            return True

        for _, value in dictionary.items():
            if search_nested_dict(value, keyword):
                return True
    return False


def update_nested_dict(main_dict: NestedDict, update_dict: NestedDict) -> None:
    """Update arbitrary nested dictionaries.

    :param main_dict: Nested dictionary to update.
    :param update_dict: Updates to apply to the dictionary.
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in main_dict:
            update_nested_dict(main_dict[key], value)
        else:
            main_dict[key] = value


def find_file(root_folder: str, target_filename: str) -> Optional[str]:
    """Search for a file within all subfolders contained at a desired root folder
    location.

    :param root_folder: The desired root folder.
    :param target_filename: The file name to find.
    :return: An optional string representing the found file's system path.
    """
    root_folder = os.path.normpath(root_folder)
    result = None
    try:
        for foldername, _, filenames in os.walk(root_folder):
            if target_filename in filenames:
                result = os.path.join(foldername, target_filename)

        if result:
            return result
        raise FileNotFoundError(
            (
                f"The file `{target_filename}` was not found "
                "at the provided root location."
            )
        )
    except FileNotFoundError as err:
        print(f"-->ERROR: {err}")
        sys.exit(1)


def load_pose_measurement(path: str) -> PoseMeasurement:
    """Load pose measurement from .json file.

    :raises RuntimeError: If the pose measurement could not be loaded.
    """
    try:
        with open(path, "r") as f:
            raw = json.load(f)

        if not isinstance(raw, dict):
            raise ValueError("Top-level JSON must be a dictionary.")

        angle_raw = raw.get("calibration_plate_angle")
        loc_raw = raw.get("calibration_plate_location")

        if not isinstance(angle_raw, dict) or not isinstance(loc_raw, dict):
            raise ValueError(
                "Missing or malformed 'calibration_plate_angle' "
                "or 'calibration_plate_location'."
            )

        triang_raw = angle_raw.get("triangulation")
        if not isinstance(triang_raw, dict):
            raise ValueError("Missing or malformed 'triangulation' section.")

        # Build typed structures
        triangulation: Triangulation = {
            "upstream_plate_corner_arclength_position_m": float(
                triang_raw["upstream_plate_corner_arclength_position_m"]
            ),
            "downstream_plate_corner_arclength_position_m": float(
                triang_raw["downstream_plate_corner_arclength_position_m"]
            ),
        }

        calibration_plate_angle: CalibrationPlateAngle = {
            "direct_measurement_deg": float(
                angle_raw["direct_measurement_deg"]
            ),
            "triangulation": triangulation,
        }

        calibration_plate_location: CalibrationPlateLocation = {
            "x_1_m": float(loc_raw["x_1_m"]),
            "x_3_m": float(loc_raw["x_3_m"]),
        }

        return {
            "calibration_plate_angle": calibration_plate_angle,
            "calibration_plate_location": calibration_plate_location,
        }

    except Exception as e:
        raise RuntimeError(f"Failed to load pose measurement: {e}")


def load_transformation_parameters(
    path: str
) -> TransformationParameters:
    """Load transformation parameters from .json file.

    :param path: File path.

    :raises ValueError: If the json file contains invalid data.
    :return: PIV transformation parameters.
    :rtype: TransformationParameters
    """
    try:
        with open(path, "r") as f:
            raw = json.load(f)

        if not isinstance(raw, dict):
            raise ValueError("Expected a top-level dictionary.")

        rot_raw = raw.get("rotation")
        trans_raw = raw.get("translation")

        if not isinstance(rot_raw, dict) or not isinstance(trans_raw, dict):
            raise ValueError(
                "Missing or malformed 'rotation' or 'translation' sections."
            )

        rotation: RotationParameters = {
            "angle_1_deg": float(rot_raw["angle_1_deg"]),
            "angle_2_deg": float(rot_raw["angle_2_deg"]),
        }

        translation: TranslationParameters = {
            "x_1_glob_ref_m": float(trans_raw["x_1_glob_ref_m"]),
            "x_2_glob_ref_m": float(trans_raw["x_2_glob_ref_m"]),
            "x_3_glob_ref_m": float(trans_raw["x_3_glob_ref_m"]),
            "x_1_loc_ref_mm": float(trans_raw["x_1_loc_ref_mm"]),
            "x_2_loc_ref_mm": float(trans_raw["x_2_loc_ref_mm"]),
        }

        return {
            "rotation": rotation,
            "translation": translation,
        }

    except Exception as e:
        raise RuntimeError(f"Failed to load transformation parameters: {e}")


def make_pose_from_trans_params(tp: TransformationParameters) -> Pose:
    """Generate a Pose object from transformation parameters."""
    return Pose(
        angle1=tp["rotation"]["angle_1_deg"],
        angle2=tp["rotation"]["angle_2_deg"],
        loc=(
            tp["translation"]["x_1_loc_ref_mm"],
            tp["translation"]["x_2_loc_ref_mm"],
        ),
        glob=(
            tp["translation"]["x_1_glob_ref_m"],
            tp["translation"]["x_2_glob_ref_m"],
            tp["translation"]["x_3_glob_ref_m"],
        ),
    )


def read_json(file_path: str) -> Optional[NestedDict]:
    """Read json file.

    :param file_path: The path to the json file.

    :return: A nested dictionary containing the file's content.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} is not a valid JSON.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def write_json(file_path: str, dictionary: dict) -> None:
    """Write a dictionary to a json formatted file.

    :param file_path: System path to the json file.
    :param dictionary: Dictionary to write to a json file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(json.dumps(dictionary, indent=4))

    logger.info(f"File '{os.path.basename(file_path)}' created.")


def load_pickle(file_path: str) -> NestedDict:
    """Load a dictionary contained in a Pickle (.pkl) file.

    :param file_path: System path to the Pickle file.
    :return: A nested dictionary containing the file's content.
    """
    try:
        with open(file_path, "rb") as file:
            data = pickle.load(file)
        return data
    except Exception as e:
        raise RuntimeError(
                    f"Loading '{os.path.basename(file_path)}' failed: {e}"
                )

    logger.info(f"File '{os.path.basename(file_path)}' loaded.")



def write_pickle(file_path: str, dictionary: NestedDict) -> None:
    """Write a dictionary to a Pickle (.pkl) file.

    :param file_path: System path to the Pickle file.
    :param dictionary: Dictionary to write to the Pickle file.
    :raises RuntimeError: If writing to file fails.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(dictionary, file)
    except Exception as e:
        raise RuntimeError(
            f"Writing '{os.path.basename(file_path)}' failed: {e}"
        ) from e

    logger.info(f"File '{os.path.basename(file_path)}' created.")


def safe_loadmat(path: str) -> Dict[str, np.ndarray]:
    """Safely load a .mat file, raising a descriptive error if it fails.

    :param path: Path to the .mat file.

    :raises RuntimeError: If an exception occurs while loading the .mat file.
    :return: Content of the .mat file.
    :rtype: Dict[str, numpy.ndarray]
    """
    try:
        return scio.loadmat(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load .mat file at {path}: {e}") from e

# def get_output_file_path() -> str:
#     """Obtains the output file system path for the processed BeVERLI Hill stereo
#     PIV data.

#     :return: A string representing the output file system path.
#     """
#     input_data = parser.InputFile().data

#     outfile_dir = os.path.join(
#         input_data["system"]["piv_plane_data_folder"], "preprocessed"
#     )
#     os.makedirs(outfile_dir, exist_ok=True)

#     file_name = (
#         f"plane{input_data['piv_data']['plane_number']}_"
#         f"{int(input_data['general']['reynolds_number'] / 1000.0)}k_"
#         f"{input_data['piv_data']['plane_type']}_preprocessed.pkl"
#     )

#     return os.path.join(outfile_dir, file_name)
