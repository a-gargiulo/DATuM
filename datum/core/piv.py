"""This module defines the main PIV data container."""

from typing import Optional

from ..utility import apputils
from .my_types import PivData
from .pose import Pose


class Piv:
    """
    This class defines the main PIV data container.

    It contains the PIV data and its pose.
    """

    def __init__(self, data: Optional[PivData] = None, pose: Optional[Pose] = None):
        """Initialize a PIV object."""
        self.data = data
        self.pose = pose if pose is not None else Pose()

    def search(self, quantity: str) -> bool:
        """Search for a specific flow quantity."""
        if self.data:
            return apputils.search_nested_dict(self.data, quantity)
        else:
            return False
