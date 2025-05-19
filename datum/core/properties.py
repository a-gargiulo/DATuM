"""Calculate fluid, flow, and reference properties for a given PIV plane."""

import os
import re
from typing import List, Tuple, cast

import numpy as np
from scipy.optimize import fsolve

from datum.core.my_types import STAT, PRInputs, Properties, StatFileData
from datum.utility import apputils

IN_HG_TO_PA = 3386.39
IN_H2O_TO_PA = 248.84
PROPERTIES_OUTFILE = "./outputs/fluid_and_flow_properties.json"


def calculate(ui: PRInputs) -> Properties:
    """
    Calculate the PIV plane's fluid, flow, and reference properties.

    The properties will be saved to a .json file.

    :param ui: User inputs from the GUI.

    :return: Fluid, flow, and reference properties.
    :rtype: Properties
    """
    try:
        run_conditions = _obtain_run_conditions(ui)
        return _extract_and_save_properties(run_conditions, ui)

    except Exception as e:
        raise RuntimeError(f"Properties calculation failed: {e}")


def _obtain_run_conditions(ui: PRInputs) -> StatFileData:
    sd: StatFileData = {}

    try:
        # Parse 'p_atm' and 'T_0'
        with open(ui["reference_stat_file"], "r", encoding="utf-8") as f:
            data = []

            r = 0
            for line in f:
                if re.match(r"^\d", line):
                    regex = r"[+-]?\d+\.\d+[eE][+-]?\d+"
                    values = re.findall(regex, line)
                    data.append([float(x) for x in values])
                else:
                    regex = r"Patm=(\d+(\.\d*)?).*T=(\d+(\.\d*)?)K"
                    match = re.search(regex, line)
                    if match:
                        r += 1
                        g1 = float(match.group(1))
                        g3 = float(match.group(3))
                        sd[f"run{r}"] = {
                            "data": np.empty((0,)),
                            "p_atm": 0.0,
                            "p_0": 0.0,
                            "p_inf": 0.0,
                            "T_0": 0.0,
                            "p_ref": 0.0,
                            "T_ref": 0.0,
                            "U_ref": 0.0,
                            "M_ref": 0.0,
                            "density_ref": 0.0,
                            "dynamic_viscosity_ref": 0.0,
                        }
                        sd[f"run{r}"]["p_atm"] = g1 * IN_HG_TO_PA
                        sd[f"run{r}"]["T_0"] = g3
                    else:
                        continue

        # Parse 'data'
        data = np.array(data)
        first_p_port_num = data[0, STAT["PORTNUM"]]
        runs_start = np.squeeze(
            np.where(data[:, STAT["PORTNUM"]] == first_p_port_num)
        )
        num_runs = runs_start.size
        if num_runs > 1:
            for run in range(num_runs):
                if run == num_runs - 1:
                    sd[f"run{run+1}"]["data"] = data[runs_start[run] :, :]
                else:
                    sd[f"run{run+1}"]["data"] = data[
                        runs_start[run] : runs_start[run + 1], :
                    ]
        else:
            sd["run1"]["data"] = data

        # Evaluate 'p_inf', 'p_ref', 'p_0', and reference conditions
        ref = _retrieve_ref_ports_and_cp(ui["tunnel_entry"])
        pp_ref, pp_con, pp_set, cps, cpc = ref

        for run in range(num_runs):
            sd_run = sd[f"run{run+1}"]

            ref_idx = np.where(
                np.isin(sd_run["data"][:, STAT["PORTNUM"]], pp_ref)
            )
            con_idx = np.where(
                np.isin(sd_run["data"][:, STAT["PORTNUM"]], pp_con)
            )
            set_idx = np.where(
                np.isin(sd_run["data"][:, STAT["PORTNUM"]], pp_set)
            )

            p_con = np.mean(
                sd_run["data"][con_idx, STAT["P_IN_H2O"]] * IN_H2O_TO_PA
                + sd_run["p_atm"]
            )
            p_set = float(
                np.squeeze(
                    sd_run["data"][set_idx, STAT["P_IN_H2O"]] * IN_H2O_TO_PA
                    + sd_run["p_atm"]
                )
            )

            def fun(p_var):
                return [
                    (p_con - p_var[0]) / (p_var[1] - p_var[0]) - cpc,
                    (p_set - p_var[0]) / (p_var[1] - p_var[0]) - cps,
                ]

            solution = fsolve(fun, x0=np.array([90000.0, 91000.0]))
            sd_run["p_inf"], sd_run["p_0"] = cast(
                Tuple[float, float], solution
            )

            sd_run["p_ref"] = float(
                np.mean(
                    sd_run["data"][ref_idx, STAT["PORTNUM"]] * IN_H2O_TO_PA
                    + sd_run["p_atm"]
                )
            )

            # Isentropic flow
            gamma = ui["gamma"] if ui["gamma"] is not None else 1.4
            gas_constant = (
                ui["gas_constant"] if ui["gas_constant"] is not None else 287
            )

            sd_run["M_ref"] = np.sqrt(
                (2 / (gamma - 1))
                * (
                    (sd_run["p_0"] / sd_run["p_ref"]) ** ((gamma - 1) / gamma)
                    - 1
                )
            )
            sd_run["T_ref"] = sd_run["T_0"] * (
                1 + (gamma - 1) / 2 * sd_run["M_ref"] ** 2
            ) ** (-1)
            sd_run["U_ref"] = sd_run["M_ref"] * np.sqrt(
                gamma * gas_constant * sd_run["T_ref"]
            )
            sd_run["density_ref"] = sd_run["p_ref"] / (
                gas_constant * sd_run["T_ref"]
            )

            # Sutherland's Law
            sd_run["dynamic_viscosity_ref"] = (
                1.716e-5
                * (sd_run["T_ref"] / 273.15) ** (3 / 2)
                * (273.15 + 110.4)
                / (sd_run["T_ref"] + 110.4)
            )

        return sd
    except Exception as e:
        raise RuntimeError(
            f"Parsing of {os.path.basename(ui['reference_stat_file'])} "
            f"failed: {e}"
        )


def _retrieve_ref_ports_and_cp(
    tunnel_entry: int,
) -> Tuple[List[int], List[int], int, float, float]:
    """Retrieve the reference pressure port numbers and Cp values.

    :param tunnel_entry: Tunnel entry number.
    :raises RuntimeError: If the tunnel entry number is invalid.
    :return: Reference pressure ports.
    :rtype: Tuple[List[int], List[int], int, float, float]
    """
    try:
        if tunnel_entry not in (2, 3):
            raise RuntimeError("The input tunnel entry number is invalid.")

        if tunnel_entry == 2:
            pp_ref = [345, 361, 365, 376, 380, 810, 824]
            pp_con = [269, 284, 300, 309, 312, 320]
            pp_set = 929
            cps = 0.9933
            cpc = 0.2536
        elif tunnel_entry == 3:
            pp_ref = [370, 823, 930, 936, 947, 952, 956]
            pp_con = [269, 284, 300, 309, 312, 320]
            pp_set = 1056
            cps = 0.9899
            cpc = 0.2294

        return (pp_ref, pp_con, pp_set, cps, cpc)
    except Exception as e:
        raise RuntimeError(
            f"An error occured while retrieving the "
            f"reference pressure port numbers: {e}"
        )


def _extract_and_save_properties(
    run_conditions: StatFileData, ui: PRInputs
) -> Properties:
    properties: Properties = {
        "fluid": {
            "density": 0.0,
            "dynamic_viscosity": 0.0,
            "heat_capacity_ratio": 0.0,
            "gas_constant": 0.0,
        },
        "flow": {
            "U_inf": 0.0,
            "p_0": 0.0,
            "p_inf": 0.0,
            "p_atm": 0.0,
            "T_0": 0.0,
        },
        "reference": {
            "T_ref": 0.0,
            "p_ref": 0.0,
            "U_ref": 0.0,
            "M_ref": 0.0,
            "density_ref": 0.0,
            "dynamic_viscosity_ref": 0.0,
        },
    }

    for run in run_conditions.keys():
        conditions = run_conditions[run]
        if (conditions["data"][0, 5] // 25000) * 25000 == ui[
            "reynolds_number"
        ]:
            properties["fluid"]["density"] = (
                ui["density"] if ui["density"] is not None else 1.103
            )
            properties["fluid"]["dynamic_viscosity"] = (
                ui["mu"] if ui["mu"] is not None else 1.8559405e-5
            )
            properties["fluid"]["heat_capacity_ratio"] = (
                ui["gamma"] if ui["gamma"] is not None else 1.4
            )
            properties["fluid"]["gas_constant"] = (
                ui["gas_constant"] if ui["gas_constant"] is not None else 287
            )

            properties["flow"]["U_inf"] = (
                ui["uinf"]
                if ui["uinf"] is not None
                else conditions["data"][0, 4]
            )
            properties["flow"]["p_0"] = conditions["p_0"]
            properties["flow"]["p_inf"] = conditions["p_inf"]
            properties["flow"]["p_atm"] = conditions["p_atm"]
            properties["flow"]["T_0"] = conditions["T_0"]

            properties["reference"]["T_ref"] = conditions["T_ref"]
            properties["reference"]["p_ref"] = conditions["p_ref"]
            properties["reference"]["U_ref"] = conditions["U_ref"]
            properties["reference"]["M_ref"] = conditions["M_ref"]
            properties["reference"]["density_ref"] = conditions["density_ref"]
            properties["reference"]["dynamic_viscosity_ref"] = conditions[
                "dynamic_viscosity_ref"
            ]

            break  # TODO: for now, only the first occurence/run is selected

    apputils.write_json(PROPERTIES_OUTFILE, cast(dict, properties))

    return properties
