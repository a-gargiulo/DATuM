"""This module introduces two classes: :py:class:`datum.parsing.InputDataParser` and
:py:class:`datum.parsing.PoseDataParser`, responsible for the parsing and loading of
the input and pose measurement files, respectively."""
import re
import sys
from typing import Optional, TextIO, Tuple, Union

from . import utility
from .my_types import InputData, PoseMeasurement


class InputFile:
    """A class for the parsing and caching of the user input data, so that it is
    globally shared.

    :cvar _instance: Singleton instance of the
        :py:class:`datum.parsing.InputDataParser`.
    :cvar _loaded: Boolean tracking if the data has been loaded.
    """

    _instance: Optional["InputData"] = None
    _loaded: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.load_data()
        return cls._instance

    ################
    #    PUBLIC    #
    ################
    def load_data(self) -> None:
        """Load and cache input data."""
        if not InputFile._loaded:
            # pylint: disable=attribute-defined-outside-init
            # noinspection PyAttributeOutsideInit
            self.data = InputFile._parse_file("input.txt")
            self._expand_system_paths()
            InputFile._loaded = True

    ###################
    #    PROTECTED    #
    ###################
    @staticmethod
    def _parse_file(filename: str) -> InputData:
        """Read the input file.

        :param filename: String representing the system path to the input file.
        :return: Nested dictionary containing the input data.
        """
        try:
            with open(filename, "r", encoding="utf-8") as file:
                input_data = InputFile._parse_file_lines(file)
            return input_data
        except FileNotFoundError:
            print(f"--> The input file {filename} was not found!\n")
            sys.exit()

    @staticmethod
    def _parse_file_lines(file: TextIO) -> InputData:
        """Parse each line of the input file.

        :param file: The input file to parse.
        :return: A nested dictionary containing the parsed input data.
        """
        input_data = {}
        section_keys = [None, None, None]

        for line in file:
            line = line.strip()

            if InputFile._is_border(line):
                continue

            # Identify the sections of the dictionary
            if line.startswith("#"):
                section_keys[:3] = [
                    line.strip("# ").replace(" ", "_").lower(),
                    None,
                    None,
                ]
                input_data[section_keys[0]] = {}
            elif line.startswith("/"):
                section_keys[1:3] = [line.strip("/ ").replace(" ", "_").lower(), None]
                input_data[section_keys[0]][section_keys[1]] = {}
            elif line.startswith("|"):
                section_keys[2] = line.strip("| ").replace(" ", "_").lower()
                input_data[section_keys[0]][section_keys[1]][section_keys[2]] = {}
            elif line:
                line = line.strip(">")
                key, value = InputFile._parse_key_value(line)

                section_ref = input_data[section_keys[0]]
                for key_part in section_keys[1:]:
                    if key_part:
                        section_ref = section_ref[key_part]
                section_ref[key] = value

        return input_data

    @staticmethod
    def _parse_key_value(line: str) -> Tuple[str, Union[bool, float, int, str]]:
        """Parse key-value pair from a corresponding line of the input file.

        :param line: String representing the line of the file containing the key-value
            pair.
        :return: Tuple representing the key-value pair.
        """
        key, value = map(str.strip, line.split(":", 1))

        # Remove content within parentheses from key values
        key = re.sub(r"\([^)]*\)", "", key).strip()

        # Make keywords lower-case and connect words by an underscore
        key = key.replace(" ", "_").lower()

        # Determine the appropriate data type for the value
        value = InputFile._convert_string_to_value(value.strip())

        return key, value

    @staticmethod
    def _convert_string_to_value(value: str) -> Union[bool, float, int]:
        """Convert a string representation of a value into the corresponding data type.

        :param value: String representing a value.
        :return: The value in the correct data type.
        """
        if value.isdigit():
            value = int(value)
        elif value.replace(".", "", 1).isdigit():
            value = float(value)
        elif re.match(r"^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$", value):
            value = float(value)
        elif (value.lower() == "true") or (value.lower() == "false"):
            value = value.lower() == "true"
        return value

    @staticmethod
    def _is_border(line: str) -> bool:
        """Check if the parsed file line represents a decorative border.

        :param line: String representing the line to parse.
        :return: Boolean value indicating whether the line is a decorative border.
        """
        borders = [r"^#+$", r"^(\+|-)+$", r"^/+$"]
        if any(re.search(pattern, line) is not None for pattern in borders):
            return True
        return False

    def _expand_system_paths(self) -> None:
        """Add additional useful system paths to the nested dictionary containing the
        input data."""
        plane_number = self.data["piv_data"]["plane_number"]
        plane_type = self.data["piv_data"]["plane_type"]
        reynolds_number = self.data["general"]["reynolds_number"]

        piv_data_root = self.data["system"]["piv_data_root_folder"]
        case_subfolders = [
            f"plane{plane_number}",
            f"{int(reynolds_number * 1e-3)}k_{plane_type.upper()}",
        ]

        plane_data_path = utility.construct_file_path(
            piv_data_root, case_subfolders[:1], ""
        )
        case_data_path = utility.construct_file_path(piv_data_root, case_subfolders, "")

        system_paths_update = {
            "system": {
                "piv_plane_data_folder": plane_data_path,
                "piv_case_data_folder": case_data_path,
            }
        }

        utility.update_nested_dict(self.data, system_paths_update)


# pylint: disable=too-few-public-methods
class PoseDataParser:
    """A class for the loading of the pose data measurement collected for each BeVERLI
    stereo PIV plane during the corresponding wind tunnel entries.

    :cvar _instance: Singleton instance of the
        :py:class:`datum.parsing.PoseDataParser`.
    :cvar _loaded: Boolean tracking if the data has been loaded.
    """

    _instance: Optional["PoseDataParser"] = None
    _loaded: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.load_data()
        return cls._instance

    def load_data(self) -> None:
        """Load and cache input data."""
        input_data = InputFile().data
        posefile = utility.find_file(
            input_data["system"]["piv_plane_data_folder"],
            input_data["piv_data"]["pose_measurement"],
        )

        if not PoseDataParser._loaded:
            # pylint: disable=attribute-defined-outside-init
            # noinspection PyAttributeOutsideInit
            self.pose_measurement = PoseDataParser._parse_file(posefile)
            PoseDataParser._loaded = True

    @staticmethod
    def _parse_file(filename: str) -> PoseMeasurement:
        """Open and read the pose measurement file.

        :param filename: String representing the system path to the pose measurement
            file.
        :return: A nested dictionary containing the pose measurement data.
        """
        try:
            with open(filename, "r", encoding="utf-8") as file:
                pose_data = PoseDataParser._parse_file_line(file)
            return pose_data
        except FileNotFoundError:
            print(f"--> The file {filename} was not found!\n")
            sys.exit()

    @staticmethod
    def _parse_file_line(file: TextIO) -> PoseMeasurement:
        """Parse the specific DATuM input file.

        :param file: An object containing the file data to be parsed.
        :return: A dictionary containing the parsed pose measurement data.
        """
        pose_data = {}
        section_keys = [None, None]

        for line in file:
            line = line.strip()

            if line.startswith("-"):
                continue

            # Identify the sections of the dictionary
            if line.startswith("["):
                section_keys[:2] = [line.strip("[] ").replace(" ", "_").lower(), None]
                pose_data[section_keys[0]] = {}
            elif line.startswith("*"):
                section_keys[1] = line.strip("* ").replace(" ", "_").lower()
                pose_data[section_keys[0]][section_keys[1]] = {}
            elif line:
                key, value = PoseDataParser._parse_key_value(line)

                section_ref = pose_data[section_keys[0]]
                for key_part in section_keys[1:]:
                    if key_part:
                        section_ref = section_ref[key_part]
                section_ref[key] = value

        return pose_data

    @staticmethod
    def _parse_key_value(line: str) -> Tuple[str, Union[float, int, str]]:
        """Parse key-value pair from a corresponding line of the input file.

        :param line: String representing the line of the file containing the key-value
            pair.
        :return: Tuple representing the key-value pair.
        """
        key, value = map(str.strip, line.split(":", 1))

        # Remove content within angle brackets from key values
        key = re.sub(r"<[^>]*>", "", key).strip()

        # Make keywords lower-case and connect words by an underscore
        key = key.replace(" ", "_").lower()

        # Determine the appropriate data type for the value
        value = PoseDataParser._convert_string_to_value(value.strip())

        return key, value

    @staticmethod
    def _convert_string_to_value(value: str) -> float:
        """Convert a string representation of a value into the corresponding data type.

        :param value: A string representing a value.
        :return: The value in the correct data type.
        """
        if value.isdigit():
            value = int(value)
        elif value.replace(".", "", 1).isdigit():
            value = float(value)
        elif re.match(r"^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$", value):
            value = float(value)
        elif value.lower() == "nan":
            value = float("nan")
        return value
