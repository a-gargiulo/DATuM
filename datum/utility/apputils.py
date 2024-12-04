"""Define useful application utility functions."""
import json
import sys
import os
import pickle
from typing import Optional

# from . import parser
from ..core.my_types import NestedDict


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


def write_json(file_path: str, dictionary: NestedDict) -> None:
    """Write a dictionary to a json formatted file.

    :param file_path: System path to the json file.
    :param dictionary: Dictionary to write to a json file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(json.dumps(dictionary, indent=4))

    print(f"--> File '{os.path.basename(file_path)}' created.\n")


def load_pickle(file_path: str) -> NestedDict:
    """Load a dictionary contained in a Pickle (.pkl) file.

    :param file_path: System path to the Pickle file.
    :return: A nested dictionary containing the file's content.
    """
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data


def write_pickle(file_path: str, dictionary: NestedDict) -> None:
    """Write a dictionary to a Pickle (.pkl) file.

    :param file_path: System path to the Pickle file.
    :param dictionary: Dictionary to write to the Pickle file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(dictionary, file)
    print(f"--> File '{os.path.basename(file_path)}' created.\n")


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
