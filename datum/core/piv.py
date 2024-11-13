"""The main class for the BeVERLI Hill stereo PIV data."""

from typing import Optional

from ..utility import apputils
from .my_types import NestedDict, PivData


class Piv:
    """Encapsulate the BeVERLI Hill stereo PIV data."""

    def __init__(self, data: Optional[PivData] = None, trans_params: Optional[NestedDict] = None):
        """Class constructor.

        :param data: The BeVERLI Hill stereo PIV data.
        :param trans_params: The coordinate transformation parameters (local -> global).
        """
        self._data = data
        self._trans_params = trans_params

    @property
    def data(self):
        """Get the BeVERLI Hill stereo PIV data.

        :return: The BeVERLI Hill stereo PIV data.
        """
        return self._data

    @data.setter
    def data(self, new_data: PivData):
        """Set the BeVERLI Hill stereo PIV data.

        :param new_data: The BeVERLI Hill stereo PIV data to be set.
        """
        self._data = new_data

    @property
    def trans_params(self):
        """Get the coordinate transformation parameters (local -> global).

        :return: The coordinate transformation parameters.
        """
        return self._trans_params

    @trans_params.setter
    def trans_params(self, new_data: NestedDict) -> None:
        """Set the transformation parameters.

        :param new_data: Coordinate transformation parameters to be set.
        """
        self._trans_params = new_data

    def search(self, quantity: str) -> bool:
        """Search for a specific flow quantity within the BeVERLI Hill stereo PIV data.

        :param quantity: The flow quantity to be searched.

        :return: Boolean value indicating whether the specified quantity was found or not.
        """
        if self.data:
            return apputils.search_nested_dict(self.data, quantity)
        else:
            return False
