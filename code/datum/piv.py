"""This module defines the :py:class:`datum.piv.Piv` base class, which encapsulates
the BeVERLI Hill stereo PIV data."""
import os
from typing import Optional

from . import parser
from . import utility
from .my_types import PivData, NestedDict


class Piv:
    """This base class encapsulates the BeVERLI Hill stereo PIV data.

    :ivar _data: The BeVERLI Hill stereo PIV data (nested dictionary).
    :ivar _transformation_parameters: The coordinate transformation parameters for the
        transformation of the BeVERLI Hill stereo PIV data from the local Cartesian PIV
        to the global Cartesian BeVERLI coordinate frame (nested dictionary).
    """

    def __init__(
        self,
        data: Optional[NestedDict] = None,
        transformation_parameters: Optional[NestedDict] = None,
    ) -> None:
        """:py:class:`datum.piv.Piv` class constructor."""
        self._data = data
        self._transformation_parameters = transformation_parameters

    @property
    def data(self) -> PivData:
        """Access the `data` property.

        :return: The BeVERLI Hill stereo PIV data.
        """
        return self._data

    @data.setter
    def data(self, new_data: NestedDict) -> None:
        """Setter of the `data` property.

        :param new_data: BeVERLI Hill stereo PIV data to be set.
        """
        self._data = new_data

    @property
    def transformation_parameters(self) -> Optional[NestedDict]:
        """Access the `transformation_parameters` property.

        :return: The coordinate transformation parameters for the transformation of the
        BeVERLI Hill stereo PIV data from the local Cartesian PIV to the global
        Cartesian BeVERLI coordinate frame.
        """
        if self._transformation_parameters is None:
            input_data = parser.InputFile().data
            parameters_file_path = utility.find_file(
                input_data["system"]["piv_plane_data_folder"],
                input_data["piv_data"]["coordinate_transformation"],
            )

            if not os.path.exists(parameters_file_path):
                self._transformation_parameters = {
                    "rotation": {"angle_deg": None}
                    if not input_data["piv_plane"]["plane_is_diagonal"]
                    else {"angle_1_deg": None, "angle_2_deg": None},
                    "translation": {
                        "x_1_glob_ref_m": None,
                        "x_2_glob_ref_m": None,
                        "x_1_loc_ref_mm": None,
                        "x_2_loc_ref_mm": None,
                    },
                }
                utility.write_json(
                    parameters_file_path, self._transformation_parameters
                )
            else:
                print(
                    f"--> File '{os.path.basename(parameters_file_path)}' already exists.\n"
                )
                print("--> Attempting to load existing file...", end=" ")
                self._transformation_parameters = utility.load_json(
                    parameters_file_path
                )
                print("--> Successfully loaded!\n")

        return self._transformation_parameters

    @transformation_parameters.setter
    def transformation_parameters(self, new_data: NestedDict) -> None:
        """Setter of the `transformation_parameters` property.

        :param new_data: Coordinate transformation parameters to be set.
        """
        self._transformation_parameters = new_data

    def has_quantity(self, data_type: str) -> bool:
        """Check if the BeVERLI Hill stereo PIV data contains a specified flow
        quantity.

        :param data_type:  The flow quantity to be searched.
        :return: Value indicating whether the specified quantity was found or not.
        """
        return utility.search_nested_dict(self.data, data_type)
