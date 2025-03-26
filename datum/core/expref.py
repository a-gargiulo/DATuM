"""Calculate the fluid, flow, and reference properties for a PIV plane."""

import re
from typing import Optional, cast

import numpy as np
from scipy.optimize import fsolve

from ..utility import apputils
from .my_types import Properties, TunnelConditions


def calculate_fluid_flow_reference_properties(
    stat_file_path: str,
    heat_capacity_ratio: float,
    gas_constant: float,
    reynolds_number: float,
    tunnel_entry_num: int,
) -> Optional[Properties]:
    """
    Calculate the PIV plane's fluid, flow, and reference properties. The properties are saved to a .json file.

    :param stat_file_path: Path to the tunnel conditions file in '.stat' format.
    :param heat_capacity_ratio: Heat capacity ratio of the working fluid [-].
    :param gas_constant: Gas constnat of the working fluid [J/mol/K].
    :param reynolds_number: Reynolds number [-].
    :param tunnel_entry_num: Number of the tunnel entry [2 or 3?].

    :return: A dictionary containing the properties or 'None', which indicates an error.
    :rtype: Optional[Properties]
    """
    properties = {}

    tunnel_conditions = _parse_stat_file(stat_file_path)
    if tunnel_conditions is None:
        return None

    if not _calculate_reference_conditions(tunnel_conditions, heat_capacity_ratio, gas_constant, tunnel_entry_num):
        return None

    _save_properties(tunnel_conditions, heat_capacity_ratio, gas_constant, reynolds_number, properties)

    return properties


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
                        tunnel_conditions[f"run{run_number}"]["p_atm"] = float(match.group(1)) * in_hg_to_pa
                        tunnel_conditions[f"run{run_number}"]["T_0"] = float(match.group(3))
                    else:
                        continue
    except FileNotFoundError:
        print(f"[ERROR]: The file {stat_file} was not found.")
        return None
    except Exception as e:
        print(f"[ERROR]: Something unexpected occurred during the parsing of {stat_file}: {e}")
        return None

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


def _calculate_reference_conditions(
    tunnel_conditions: TunnelConditions,
    heat_capacity_ratio: float,
    gas_constant: float,
    tunnel_entry: int,
) -> bool:
    in_h2o_to_pa = 248.84

    reference_ports = None
    contraction_reference_ports = None
    settling_chamber_reference_ports = None
    cpc = None
    cps = None
    if tunnel_entry == 2:
        reference_ports = [345, 361, 365, 376, 380, 810, 824]
        contraction_reference_ports = [269, 284, 300, 309, 312, 320]
        settling_chamber_reference_ports = 929

        cps = 0.9933
        cpc = 0.2536
    elif tunnel_entry == 3:
        reference_ports = [370, 823, 930, 936, 947, 952, 956]
        contraction_reference_ports = [269, 284, 300, 309, 312, 320]
        settling_chamber_reference_ports = 1056

        cps = 0.9899
        cpc = 0.2294

    if (
        reference_ports is None
        or contraction_reference_ports is None
        or settling_chamber_reference_ports is None
        or cpc is None
        or cps is None
    ):
        print("[Error]: Invalid tunnel entry for the computation of the experimental reference conditions.")
        return False

    number_of_runs = len(tunnel_conditions.keys())
    p_contraction = None
    p_settling = None
    for run in range(number_of_runs):
        run_data = cast(dict, tunnel_conditions[f"run{run+1}"])

        contraction_idx = np.where(np.isin(run_data["pressure_data"][:, 0], contraction_reference_ports))
        settling_idx = np.where(np.isin(run_data["pressure_data"][:, 0], settling_chamber_reference_ports))
        reference_idx = np.where(np.isin(run_data["pressure_data"][:, 0], reference_ports))

        p_contraction = np.mean(run_data["pressure_data"][contraction_idx, 8] * in_h2o_to_pa + run_data["p_atm"])
        p_settling = float(np.squeeze(run_data["pressure_data"][settling_idx, 8] * in_h2o_to_pa + run_data["p_atm"]))

        def fun(p_var):
            return [
                (p_contraction - p_var[0]) / (p_var[1] - p_var[0]) - cpc,
                (p_settling - p_var[0]) / (p_var[1] - p_var[0]) - cps,
            ]

        solution = fsolve(fun, x0=np.array([90000, 91000]))
        run_data["p_inf"], run_data["p_0"] = solution[0], solution[1]
        run_data["p_ref"] = np.mean(run_data["pressure_data"][reference_idx, 8] * in_h2o_to_pa + run_data["p_atm"])

        # Isentropic flow
        run_data["M_ref"] = np.sqrt(
            (2 / (heat_capacity_ratio - 1))
            * ((run_data["p_0"] / run_data["p_ref"]) ** ((heat_capacity_ratio - 1) / heat_capacity_ratio) - 1)
        )
        run_data["T_ref"] = run_data["T_0"] * (1 + (heat_capacity_ratio - 1) / 2 * run_data["M_ref"] ** 2) ** (-1)
        run_data["U_ref"] = run_data["M_ref"] * np.sqrt(heat_capacity_ratio * gas_constant * run_data["T_ref"])
        run_data["rho_ref"] = run_data["p_ref"] / (gas_constant * run_data["T_ref"])

        # Sutherland's Law
        run_data["mu_ref"] = (
            1.716e-5 * (run_data["T_ref"] / 273.15) ** (3 / 2) * (273.15 + 110.4) / (run_data["T_ref"] + 110.4)
        )

    return True


def _save_properties(
    tunnel_conditions: TunnelConditions,
    heat_capacity_ratio: float,
    gas_constant: float,
    reynolds_number: float,
    properties: Properties,
) -> None:
    for run in tunnel_conditions.keys():
        conditions = cast(dict, tunnel_conditions[run])
        if (conditions["pressure_data"][0, 5] // 25000) * 25000 == reynolds_number:
            properties["fluid"] = {
                "density": conditions["rho_ref"],  # incompressible
                "dynamic_viscosity": conditions["mu_ref"],  # incompressible
                "heat_capacity_ratio": heat_capacity_ratio,
                "gas_constant": gas_constant,
                "density_ref": conditions["rho_ref"],
                "dynamic_viscosity_ref": conditions["mu_ref"],
            }

            properties["flow"] = {
                "U_inf": conditions["pressure_data"][0, 4],
                "p_0": conditions["p_0"],
                "p_inf": conditions["p_inf"],
                "p_atm": conditions["p_atm"],
                "T_0": conditions["T_0"],
                "T_ref": conditions["T_ref"],
                "p_ref": conditions["p_ref"],
                "U_ref": conditions["U_ref"],
                "M_ref": conditions["M_ref"],
            }
            break

    # TODO: Catch possible errors during file writing.
    apputils.write_json("./outputs/fluid_and_flow_properties.json", properties)
