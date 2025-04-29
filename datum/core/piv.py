"""This module defines the main PIV data container."""

from typing import Optional

from datum.core.my_types import PivData
from datum.core.pose import Pose


class Piv:
    """BeVERLI Hill stereo PIV plane data container.

    Includes both the actual flow data and the geometrical pose.
    """

    def __init__(
        self, dd: Optional[PivData] = None, pp: Optional[Pose] = None
    ) -> None:
        """Initialize a BeVERLI Hill stereo PIV plane.

        :param data: Flow data.
        :param pose: Geometrical pose data.
        """
        self._data = dd
        self.pose = pp if pp is not None else Pose()

    @property
    def data(self) -> PivData:
        """Retrieve flow data.

        :raises ValueError: If flow data is not initialized.
        :return: The flow data.
        :rtype:
        """
        if self._data is None:
            raise ValueError("PIV data is not initialized.")
        return self._data

    @data.setter
    def data(self, dd: PivData):
        """Set the flow data."""
        self._data = dd
