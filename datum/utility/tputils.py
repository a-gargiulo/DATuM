"""Utility functions for working with Tecplot files."""
import os
import re
from typing import Dict, Tuple

import numpy as np
import tecplot as tp
from datum.utility import logging


def get_ijk(file_path: str) -> Tuple[int, ...]:
    """Obtain the dimensions from an ijk-ordered Tecplot dataset.

    :param file_path: Path to the Tecplot file.
    :raises FileNotFoundError: If the file does not exist.
    :raises ValueError: If the file contains non-numeric data or has
        inconsistent row lengths.
    :raises OSError: For other I/O related errors (e.g., permission issues).
    :return: ijk dimensions of the dataset.
    :rtype: Tuple[int, ...]
    """
    try:
        dimensions = []

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip().startswith("ZONE"):
                    args = line.split(",")
                    dimensions.extend(
                        int(re.sub(r"\D", "", arg))
                        for arg in args
                        if arg.strip().startswith(("I=", "J=", "K="))
                    )
                    break

        if len(dimensions) != 2:
            raise ValueError(
                "The calibration image is not 2d or the format is wrong."
            )
        return tuple(dimensions)
    except Exception as e:
        logging.logger.error(str(e))
        raise RuntimeError


def get_tecplot_derivatives(
    slice_path: str, zone_name: str, use_all: bool
) -> Dict[str, np.ndarray]:
    """Extract the non-computable components of the mean velocity gradient
    tensor from CFD data using Tecplot.

    :return: A dictionary containing the extracted mean velocity gradient tensor
        components as NumPy ndarrays of shape (m, n), where m and n represent
        the number of points in the x:sub:`1` and x:sub:`2 direction, respectively.
    """
    try:
        cfd_data = {}

        tp.new_layout()
        dataset = tp.data.load_tecplot(slice_path)
        zone = dataset.zone(zone_name)

        cfd_data["X"] = zone.values("X").as_numpy_array()
        cfd_data["Y"] = zone.values("Y").as_numpy_array()

        gradients = [
            ("dUdZ", True),
            ("dVdZ", True),
            ("dWdX", use_all),
            ("dWdY", use_all),
        ]
        for gradient, use in gradients:
            data = zone.values(gradient).as_numpy_array()
            if use:
                cfd_data[gradient] = data
            else:
                continue

        return cfd_data
    except Exception as e:
        raise RuntimeError(f"Tecplot processing failed: {e}") from e
