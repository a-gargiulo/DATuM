"""This module defines the main PIV data container."""

from typing import Optional, TypedDict

from ..utility import apputils
from .my_types import PivData
from .pose import Pose


class Piv:
    """
    This class defines the main PIV data container.

    It contains the PIV data and its pose.
    """

    def __init__(self, d: Optional[PivData] = None, p: Optional[Pose] = None):
        """Initialize a PIV object.

        :param data: PIV data.
        :param pose: Pose data for the PIV plane.
        """
        self._data = d
        self.pose = p if p is not None else Pose()

    @property
    def data(self) -> PivData:
        """PIV data property."""
        if self._data is None:
            raise ValueError("PIV data is not initialized.")
        return self._data

    @data.setter
    def data(self, d: PivData):
        self._data = d

    def search(self, quantity: str) -> bool:
        """Search for a specific flow quantity.

        :param quantity: Quantity to be searched within the PIV data.
        """
        if self.data:
            return apputils.search_nested_dict(self.data, quantity)
        else:
            return False
