"""This module defines the pose class."""

from typing import List, Optional


class Pose:
    """
    This class defines a container for the pose data.

    It further provides functionalities to calculate and manipulate the pose.
    """

    def __init__(
            self,
            orientation: float = 0.0,
            loc: List[float] = [0.0, 0.0],
            glob: List[float] = [0.0, 0.0]
    ):
        """Initialize a PIV plane pose."""
        self.orientation = orientation
        self.loc = loc
        self.glob = glob

    # def calculate_global_pose(hill_orientation_deg: float, measurement: str) -> Optional[Tuple[float, float, float]]:
