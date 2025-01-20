"""Calculate the reference conditions for the experimental data."""
import re
from typing import cast, Dict, Optional, Union, List

import numpy as np
from scipy.optimize import fsolve

from ..utility import apputils
from .my_types import Properties, TunnelConditions


def get_properties(opts: Dict[str, Union[int, float, str, bool]]) -> Optional[Properties]:
    """Obtain the reference conditions for the experimental data.

    The reference conditions are calculated using a corresponding file containing the conditions measured in the VT SWT
    during the specific experimental run of intererst. Accepted files are in .stat format.
    """
    properties = apputils.read_json(cast(str, opts["properties_file"]))
    if properties is None:
        return None

    tunnel_conditions = _parse_stat_file(cast(str, opts["stat_file"]))
    if tunnel_conditions is None:
        return None

    _calculate_reference_conditions(
            cast(TunnelConditions, tunnel_conditions),
            cast(int, opts["tunnel_entry"]),
            cast(Properties, properties)
    )

    _update_properties(
            cast(Properties, properties),
            cast(str, opts["properties_file"]),
            cast(float, opts["reynolds_number"]),
            cast(TunnelConditions, tunnel_conditions)
    )

    return cast(Properties, properties)


def _parse_stat_file(stat_file: str) -> Optional[TunnelConditions]:
    in_hg_to_pa = 3386.39

    tunnel_conditions = {}

    try:
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
    except FileNotFoundError:
        print(f"Error: The file at {stat_file} was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    # Split data into runs
    pressure_data = np.array(pressure_data, dtype=np.float64)
    run_start_indices = np.squeeze(np.where(pressure_data[:, 0] == pressure_data[0, 0]))
    number_of_runs = run_start_indices.size
    if number_of_runs > 1:
        for run_number in range(number_of_runs):
            if run_number == number_of_runs - 1:
                tunnel_conditions[f"run{run_number+1}"]["pressure_data"] = pressure_data[
                    run_start_indices[run_number]:, :
                ]
            else:
                tunnel_conditions[f"run{run_number+1}"]["pressure_data"] = pressure_data[
                    run_start_indices[run_number]:run_start_indices[run_number + 1], :
                ]
    else:
        tunnel_conditions["run1"]["pressure_data"] = pressure_data

    return tunnel_conditions


def _calculate_reference_conditions(
    tunnel_conditions: TunnelConditions,
    entry: int,
    properties: Properties,
) -> None:
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
            np.isin(cast(dict, run_data)["pressure_data"][:, 0], cast(List[int], contraction_reference_ports))
        )
        settling_idx = np.where(
            np.isin(cast(dict, run_data)["pressure_data"][:, 0], cast(List[int], settling_chamber_reference_ports))
        )
        reference_idx = np.where(
            np.isin(cast(dict, run_data)["pressure_data"][:, 0], cast(List[int], reference_ports))
        )

        p_contraction = np.mean(
            cast(dict, run_data)["pressure_data"][contraction_idx, 8] * in_h2o_to_pa
            + cast(dict, run_data)["p_atm"]
        )
        p_settling = float(
            np.squeeze(
                cast(dict, run_data)["pressure_data"][settling_idx, 8] * in_h2o_to_pa
                + cast(dict, run_data)["p_atm"]
            )
        )

        def fun(p_var):
            return [
                (p_contraction - p_var[0]) / (p_var[1] - p_var[0]) - cpc,
                (p_settling - p_var[0]) / (p_var[1] - p_var[0]) - cps,
            ]

        solution = fsolve(fun, x0=np.array([90000, 91000]))
        cast(dict, run_data)["p_inf"], cast(dict, run_data)["p_0"] = solution[0], solution[1]
        cast(dict, run_data)["p_ref"] = np.mean(
            cast(dict, run_data)["pressure_data"][reference_idx, 8] * in_h2o_to_pa
            + cast(dict, run_data)["p_atm"]
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


def _update_properties(
    properties: Properties,
    properties_file: str,
    reynolds_number: float,
    tunnel_conditions: TunnelConditions,
) -> None:
    for run in tunnel_conditions.keys():
        if (
            cast(dict, tunnel_conditions[run])["pressure_data"][0, 5] // 25000
        ) * 25000 == reynolds_number:
            updates_dict = {
                "fluid": {
                    "density_ref": tunnel_conditions[run]["rho_ref"],
                    "dynamic_viscosity_ref": tunnel_conditions[run]["mu_ref"],
                },
                "flow": {
                    "U_inf": cast(dict, tunnel_conditions[run])["pressure_data"][0, 4],
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
            apputils.update_nested_dict(properties, updates_dict)
            break

    apputils.write_json(properties_file, properties)
