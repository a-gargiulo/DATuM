"""This module provides functions for computing the reference conditions for the
experimental data, provided a corresponding tunnel conditions (.stat) file."""
import os
import re
from typing import Dict, Union, Optional

import numpy as np
from scipy.optimize import fsolve

from . import utility
from .parser import InputFile


def get_all_plane_properties() -> Optional[Dict[str, Dict[str, float]]]:
    input_data = InputFile().data
    stat_file_exists = os.path.exists(
        os.path.join(
            input_data["system"]["piv_plane_data_folder"],
            input_data["general"]["tunnel_conditions"]
        )
    )
    properties_file = utility.find_file(
        input_data["system"]["piv_plane_data_folder"],
        input_data["general"]["fluid_and_flow_properties"],
    )

    properties = utility.load_json(properties_file)

    if stat_file_exists:
        tunnel_conditions = parse_stat_file()
        calculate_reference_conditions(tunnel_conditions, input_data["general"]["tunnel_entry"], properties)
        update_properties(properties, tunnel_conditions)
        return properties
    else:
        print("The .stat file is unavailable, the file containing the fluid and flow"
              "properties will be left unchanged.")
        return properties


def parse_stat_file() -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
    """Parse .stat file containing tunnel conditions.

    return: A nested dictionary containing the tunnel conditions.
    """
    in_hg_to_pa = 3386.39

    input_data = InputFile().data
    stat_file = utility.find_file(
        input_data["system"]["piv_plane_data_folder"],
        input_data["general"]["tunnel_conditions"],
    )

    tunnel_conditions = {}
    with open(stat_file, "r", encoding="utf-8") as file:
        pressure_data = []

        run_number = 0
        for line in file:
            if re.match(r"^\d", line):
                numeric_values = re.findall(r"[+-]?\d+\.\d+[eE][+-]?\d+", line)
                pressure_data.append([float(x) for x in numeric_values])
            else:
                match = re.search(r"Patm=(\d+(\.\d*)?).*T=(\d+(\.\d*)?)K", line)
                if match:
                    run_number += 1
                    tunnel_conditions[f"run{run_number}"] = {}
                    tunnel_conditions[f"run{run_number}"]["p_atm"] = (
                        float(match.group(1)) * in_hg_to_pa
                    )
                    tunnel_conditions[f"run{run_number}"]["T_0"] = float(match.group(3))
                else:
                    continue

    # Split data into runs
    pressure_data = np.array(pressure_data, dtype=np.float64)
    run_start_indices = np.squeeze(np.where(pressure_data[:, 0] == pressure_data[0, 0]))
    number_of_runs = run_start_indices.size
    if number_of_runs > 1:
        for run_number in range(number_of_runs):
            if run_number == number_of_runs - 1:
                tunnel_conditions[f"run{run_number+1}"]["pressure_data"] = pressure_data[
                    run_start_indices[run_number] :, :
                ]
            else:
                tunnel_conditions[f"run{run_number+1}"]["pressure_data"] = pressure_data[
                    run_start_indices[run_number] : run_start_indices[run_number + 1], :
                ]
    else:
        tunnel_conditions["run1"]["pressure_data"] = pressure_data

    return tunnel_conditions


def calculate_reference_conditions(
    tunnel_conditions: Dict[str, Dict[str, Union[float, np.ndarray]]],
    entry: int,
    properties: Dict[str, Dict[str, float]],
) -> None:
    """Calculate the reference conditions for the experimental data.

    :param tunnel_conditions: Data contained in the .stat file.
    :param entry: Integer identifier for the BeVERLI tunnel entry.
    :param properties: A nested dictionary containing the fluid's properties.
    """
    # pylint: disable=too-many-locals
    in_h2o_to_pa = 248.84

    heat_capacity_ratio = properties["fluid"]["heat_capacity_ratio"]
    gas_constant_air = properties["fluid"]["gas_constant_air"]

    reference_ports = None
    contraction_reference_ports = None
    settling_chamber_reference_ports = None
    cpc = None
    cps = None
    if entry == 2:
        reference_ports = [345, 361, 365, 376, 380, 810, 824]
        contraction_reference_ports = [269, 284, 300, 309, 312, 320]
        settling_chamber_reference_ports = 929

        cps = 0.9933
        cpc = 0.2536
    elif entry == 3:
        reference_ports = [370, 823, 930, 936, 947, 952, 956]
        contraction_reference_ports = [269, 284, 300, 309, 312, 320]
        settling_chamber_reference_ports = 1056

        cps = 0.9899
        cpc = 0.2294

    number_of_runs = len(tunnel_conditions.keys())
    p_contraction = None
    p_settling = None
    for run in range(number_of_runs):
        run_data = tunnel_conditions[f"run{run+1}"]

        contraction_idx = np.where(
            np.isin(run_data["pressure_data"][:, 0], contraction_reference_ports)
        )
        settling_idx = np.where(
            np.isin(run_data["pressure_data"][:, 0], settling_chamber_reference_ports)
        )
        reference_idx = np.where(
            np.isin(run_data["pressure_data"][:, 0], reference_ports)
        )

        p_contraction = np.mean(
            run_data["pressure_data"][contraction_idx, 8] * in_h2o_to_pa
            + run_data["p_atm"]
        )
        p_settling = float(
            np.squeeze(
                run_data["pressure_data"][settling_idx, 8] * in_h2o_to_pa
                + run_data["p_atm"]
            )
        )

        def fun(p_var):
            return [
                (p_contraction - p_var[0]) / (p_var[1] - p_var[0]) - cpc,
                (p_settling - p_var[0]) / (p_var[1] - p_var[0]) - cps,
            ]

        solution = fsolve(fun, x0=np.array([90000, 91000]))
        run_data["p_inf"], run_data["p_0"] = solution[0], solution[1]
        run_data["p_ref"] = np.mean(
            run_data["pressure_data"][reference_idx, 8] * in_h2o_to_pa
            + run_data["p_atm"]
        )

        run_data["M_ref"] = np.sqrt(
            (2 / (heat_capacity_ratio - 1))
            * (
                (run_data["p_0"] / run_data["p_ref"])
                ** ((heat_capacity_ratio - 1) / heat_capacity_ratio)
                - 1
            )
        )
        run_data["T_ref"] = run_data["T_0"] * (
            1 + (heat_capacity_ratio - 1) / 2 * run_data["M_ref"] ** 2
        ) ** (-1)
        run_data["U_ref"] = run_data["M_ref"] * np.sqrt(
            heat_capacity_ratio * gas_constant_air * run_data["T_ref"]
        )
        run_data["rho_ref"] = run_data["p_ref"] / (gas_constant_air * run_data["T_ref"])
        run_data["mu_ref"] = (
            1.716e-5
            * (run_data["T_ref"] / 273.15) ** (3 / 2)
            * (273.15 + 110.4)
            / (run_data["T_ref"] + 110.4)
        )


def update_properties(
    properties: Dict[str, Dict[str, float]],
    tunnel_conditions: Dict[str, Dict[str, Union[float, np.ndarray]]],
) -> None:
    """Update the properties dictionary with the experimental reference conditions.

    :param properties: Nested dictionary containing the available fluid and flow
        properties.
    :param tunnel_conditions: Dictionary containing the data extracted from an
        available .stat file for the analyzed PIV plane.
    """
    input_data = InputFile().data
    reynolds_number = input_data["general"]["reynolds_number"]
    for run in tunnel_conditions.keys():
        if (
            tunnel_conditions[run]["pressure_data"][0, 5] // 25000
        ) * 25000 == reynolds_number:
            updates_dict = {
                "fluid": {
                    "density_ref": tunnel_conditions[run]["rho_ref"],
                    "dynamic_viscosity_ref": tunnel_conditions[run]["mu_ref"],
                },
                "flow": {
                    "U_inf": tunnel_conditions[run]["pressure_data"][0, 4],
                    "p_0": tunnel_conditions[run]["p_0"],
                    "p_inf": tunnel_conditions[run]["p_inf"],
                    "p_atm": tunnel_conditions[run]["p_atm"],
                    "T_0": tunnel_conditions[run]["T_0"],
                    "T_ref": tunnel_conditions[run]["T_ref"],
                    "p_ref": tunnel_conditions[run]["p_ref"],
                    "U_ref": tunnel_conditions[run]["U_ref"],
                    "M_ref": tunnel_conditions[run]["M_ref"],
                },
            }
            utility.update_nested_dict(properties, updates_dict)
            break

    utility.write_json(
        utility.find_file(
            input_data["system"]["piv_plane_data_folder"],
            input_data["general"]["fluid_and_flow_properties"],
        ),
        properties,
    )
