"""Utility functions for working with Tecplot files."""
import re
from typing import Tuple


def get_ijk(file_path: str) -> Tuple[int, ...]:
    """Obtain the dimensions of an ijk-ordered Tecplot data set.

    :param file_path: The path to the Tecplot file.

    :return: The dimensions of the data.
    """
    dimensions = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip().startswith("ZONE"):
                args = line.split(',')
                dimensions.extend(
                    int(re.sub(r"\D", "", arg)) for arg in args if arg.strip().startswith(("I=", "J=", "K="))
                )
                break
    return tuple(dimensions)
