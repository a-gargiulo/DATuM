"""Transformation sub-module for translation operations."""
from typing import Tuple, TYPE_CHECKING
from ..my_types import Coordinates


if TYPE_CHECKING:
    from ..piv import Piv


def translate_all(piv: "Piv"):
    """Translate the stereo PIV data."""
    x_1_shift, x_2_shift = get_translation_vector(piv)
    translate_coordinates(piv.data["coordinates"], x_1_shift, x_2_shift)


def get_translation_vector(piv: "Piv") -> Tuple[float, float]:
    """Obtain the translation vector.

    :param piv: Instance of the :py:class:`datum.piv.Piv` class.
    :return: A tuple of floats, representing the components of the translation
        vector.
    """
    # THIS FUNCTION IS QUITE SPECIFIC TO THE BEVERLI PIV DATA, WHICH IS ASSUMED
    # TO HAVE COORDINATES MEASURED IN MM WHEN RAW
    x_1_shift = piv.pose.glob[0] * 1000 - piv.pose.loc[0]
    x_2_shift = piv.pose.glob[1] * 1000 - piv.pose.loc[1]

    return x_1_shift, x_2_shift


def translate_coordinates(
    coordinates: Coordinates,
    x_1_shift: float,
    x_2_shift: float,
):
    """Perform the coordinate translation.

    :param coordinates: Dictionary containing the stereo PIV coordinates.
    :param x_1_shift: Coordinate shift in the x_1 direction.
    :param x_2_shift: Coordinate shift in the x_2 direction.
    """
    coordinates["X"] += x_1_shift
    coordinates["Y"] += x_2_shift
