"""Defines the :py:class:`datum.piv.Piv` class, which encapsulates the BeVERLI Hill
stereo PIV data."""

import os
from typing import Optional

from . import parser, utility
from .my_types import NestedDict, PivData


class Piv:
    """Encapsulates the BeVERLI Hill stereo PIV data."""

    def __init__(
        self,
        data: Optional[PivData] = None,
        transformation_parameters: Optional[NestedDict] = None,
    ) -> None:
        """Constructs a :py:class:`datum.piv.Piv` object.

        :param data: The BeVERLI Hill stereo PIV data.
        :type data: :py:data:`datum.my_types.PivData`
        :param transformation_parameters: The coordinate transformation parameters for
            transforming the BeVERLI Hill stereo PIV data from their local Cartesian PIV
            coordinate system to the global Cartesian coordinate system of the BeVERLI
            experiment in the Virginia Tech Stability Wind Tunnel.
        :type transformation_parameters: :py:data:`datum.my_types.NestedDict`
        """
        self._data = data
        self._transformation_parameters = transformation_parameters

    @property
    def data(self):
        """The BeVERLI Hill stereo PIV data.

        :return: The BeVERLI Hill stereo PIV data.
        :rtype: :py:data:`datum.my_types.PivData`
        """
        return self._data

    @data.setter
    def data(self, new_data: PivData):
        """Sets the `data` property.

        :param new_data: The BeVERLI Hill stereo PIV data to be set.
        """
        self._data = new_data

    @property
    def transformation_parameters(self):
        """The coordinate transformation parameters for transforming the BeVERLI Hill
        stereo PIV data from their local Cartesian PIV coordinate system to the global
        Cartesian coordinate system of the BeVERLI experiment in the Virginia Tech
        Stability Wind Tunnel.

        :return: The transformation parameters.
        :rtype: :py:data:`datum.my_types.NestedDict`
        """
        if self._transformation_parameters is None:
            input_data = parser.InputFile().data
            parameters_file_path = utility.construct_file_path(
                input_data["system"]["piv_plane_data_folder"],
                [],
                input_data["piv_data"]["coordinate_transformation"],
            )

            if not os.path.exists(parameters_file_path):
                self._transformation_parameters = {
                    "rotation": (
                        {"angle_deg": None}
                        if not input_data["piv_plane"]["plane_is_diagonal"]
                        else {"angle_1_deg": None, "angle_2_deg": None}
                    ),
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
                    f"--> File '{os.path.basename(parameters_file_path)}' "
                    "already exists.\n"
                )
                print("--> Attempting to load existing file...", end=" ")
                self._transformation_parameters = utility.load_json(
                    parameters_file_path
                )
                print("--> Successfully loaded!\n")

        return self._transformation_parameters

    @transformation_parameters.setter
    def transformation_parameters(self, new_data: NestedDict) -> None:
        """Sets the `transformation_parameters` property.

        :param new_data: Coordinate transformation parameters to be set.
        """
        self._transformation_parameters = new_data

    def search(self, quantity: str) -> bool:
        """Searches the BeVERLI Hill stereo PIV data for a specified flow quantity.

        :param quantity: The flow quantity to be searched.

        :return: Boolean value indicating whether the specified quantity is contained in
            the BeVERLI Hill stereo PIV data or not.
        :rtype: :py:type:`bool`
        """
        return utility.search_nested_dict(self.data, quantity)
