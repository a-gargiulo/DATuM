"""Write the BeVERLI stereo PIV inflow reference_conditions in ASCII format"""
import re
import sys

import numpy as np
import scipy.io as scio
from scipy.optimize import fsolve

# Global Constants
HILL_HEIGHT = 0.186944
GAMMA = 1.4
RAIR = 287
IN_HG_TO_PA = 3386.39
IN_H2O_TO_PA = 248.84


def calculate_reference_conditions(entry: int, reynolds_number: float, stat_file_path):
    try:
        if entry == 2:
            CPS = 0.9933
            CPC = 0.2536

            reference_ports = [345, 361, 365, 376, 380, 810, 824]
            contraction_reference_ports = [269, 284, 300, 309, 312, 320]
            settling_chamber_reference_ports = 929
        elif entry == 3:
            CPS = 0.9899
            CPC = 0.2294

            reference_ports = [370, 823, 930, 936, 947, 952, 956]
            contraction_reference_ports = [269, 284, 300, 309, 312, 320]
            settling_chamber_reference_ports = 1056
        else:
            raise ValueError("Please select a valid tunnel entry.")
    except ValueError as err:
        print(f"ERROR: {err}")
        sys.exit(1)

    # Calculate reference conditions
    reference_conditions = {}

    with open(stat_file_path, "r", encoding="utf-8") as file:
        pressure_data = []

        run_number = 0
        for line in file:
            if re.match(r"^\d", line):
                numeric_values = re.findall(r"[+-]?\d+\.\d+[eE][+-]?\d+", line)
                pressure_data.append(list(map(lambda x: float(x), numeric_values)))
            else:
                match = re.search("Patm=(\d+(\.\d*)?).*T=(\d+(\.\d*)?)K", line)
                if match:
                    run_number += 1
                    reference_conditions[f"run{run_number}"] = {}
                    reference_conditions[f"run{run_number}"]["p_atm"] = (
                        float(match.group(1)) * IN_HG_TO_PA
                    )
                    reference_conditions[f"run{run_number}"]["T_0"] = float(match.group(3))
                else:
                    continue

# Split data into runs
pressure_data = np.array(pressure_data, dtype=np.float64)
run_start_indices = np.squeeze(np.where(pressure_data[:, 0] == pressure_data[0, 0]))
number_of_runs = len(run_start_indices)
for run_number in range(number_of_runs):
    if run_number == number_of_runs - 1:
        reference_conditions[f"run{run_number+1}"]["pressure_data"] = pressure_data[
            run_start_indices[run_number] :, :
        ]
    else:
        reference_conditions[f"run{run_number+1}"]["pressure_data"] = pressure_data[
            run_start_indices[run_number] : run_start_indices[run_number + 1], :
        ]

# Calculate all reference conditions
for run in range(number_of_runs):
    run_data = reference_conditions[f"run{run+1}"]

    contraction_idx = np.where(
        np.isin(run_data["pressure_data"][:, 0], contraction_reference_ports)
    )
    settling_idx = np.where(
        np.isin(run_data["pressure_data"][:, 0], settling_chamber_reference_ports)
    )
    reference_idx = np.where(np.isin(run_data["pressure_data"][:, 0], reference_ports))

    p_contraction = np.mean(
        run_data["pressure_data"][contraction_idx, 8] * IN_H2O_TO_PA + run_data["p_atm"]
    )
    p_settling = float(
        np.squeeze(
            run_data["pressure_data"][settling_idx, 8] * IN_H2O_TO_PA
            + run_data["p_atm"]
        )
    )

    def fun(p):
        return [
            (p_contraction - p[0]) / (p[1] - p[0]) - CPC,
            (p_settling - p[0]) / (p[1] - p[0]) - CPS,
        ]

    run_data["p_inf"], run_data["p_0"] = fsolve(fun, [90000, 91000])
    run_data["p_ref"] = np.mean(
        run_data["pressure_data"][reference_idx, 8] * IN_H2O_TO_PA + run_data["p_atm"]
    )

    run_data["M_ref"] = np.sqrt(
        (2 / (GAMMA - 1))
        * ((run_data["p_0"] / run_data["p_ref"]) ** ((GAMMA - 1) / GAMMA) - 1)
    )
    run_data["T_ref"] = run_data["T_0"] * (
        1 + (GAMMA - 1) / 2 * run_data["M_ref"] ** 2
    ) ** (-1)
    run_data["U_ref"] = run_data["M_ref"] * np.sqrt(GAMMA * RAIR * run_data["T_ref"])
    run_data["rho_ref"] = run_data["p_ref"] / (RAIR * run_data["T_ref"])
    run_data["mu_ref"] = (
        1.716e-5
        * (run_data["T_ref"] / 273.15) ** (3 / 2)
        * (273.15 + 110.4)
        / (run_data["T_ref"] + 110.4)
    )


# Load PIV data
piv = {}
runs = [2, 6, 12]
for number, run in zip(reynolds_number, runs):
    # Load data
    piv[f"{number}"] = scio.loadmat(f"Re{number}Prof.mat")[f"s{number}"][0, 0]
    file_name = f"PIV_Re{number}K_45Deg_Xneg1p83m_Z0m"
    with open(f"{file_name}.txt", "w", encoding="utf-8") as file:
        file.write(f"{file_name}\n")
        file.write("Conditions and information for this case.\n")
        file.write("\n")
        file.write(
            "Two sets of PIV were acquired at the same nominal conditions. "
            "The first run,\nlabeled as the `fast` set, was measured at a sampling "
            "frequency of 12.5 kHz.\nThe second run, designated as the `slow` set, was "
            "recorded at 1 kHz. Each PIV\nset comprises 24,000 double-frame images. "
            "This case represents the `fast` PIV\nset.\n"
        )
        file.write("\n")
        file.write(
            "For more information on the coordinate system and articles, "
            "please refer to the\noverall data description document or to\n\n"
            "\t***\n"
            "\tGargiulo et al. (2023). Strategies for Computational Fluid Dynamics\n"
            "\tValidation Experiments. Journal of Verification, Validation and\n"
            "\tUncertainty Quantification. Accepted Manuscript.\n"
            "\t***\n\n"
        )
        file.write(
            "X_1 is the streamwise coordinate, positive in the direction of the "
            "freestream.\nX_2 is the flat wall-normal coordinate.\n"
            "X_3 is the spanwise coordinate.\nAll distance units are in meters.\n\n"
            "A Spalding fit was used to adjust the X_2-coordinates to account for "
            "error in the nominal measurement location.\n\n"
        )
        file.write("Reference conditions:\n")
        file.write(f"Spalding X_2_correction = {piv[f'{number}']['y0'][0][0]:.4f} m\n")
        file.write(
            f"ReH = {reference_conditions[f'run{run}']['pressure_data'][0,5]:.1f}\n"
        )
        file.write(
            f"U_inf = {reference_conditions[f'run{run}']['pressure_data'][0,4]:.2f} m/s"
            "\n"
        )
        file.write(f"P_0 = {reference_conditions[f'run{run}']['p_0']:.2f} Pa\n")
        file.write(f"T_0 = {reference_conditions[f'run{run}']['T_0']:.2f} K\n")
        file.write(f"P_inf = {reference_conditions[f'run{run}']['p_inf']:.2f} Pa\n")
        file.write(f"P_ref = {reference_conditions[f'run{run}']['p_ref']:.2f} Pa\n")

        file.write(f"T_ref = {reference_conditions[f'run{run}']['T_ref']:.2f} K\n")
        file.write(f"U_ref = {reference_conditions[f'run{run}']['U_ref']:.2f} m/s\n")
        file.write(f"U_e = {piv[f'{number}']['Ue'][0][0]:.2f} m/s\n")
        file.write(f"delta = {piv[f'{number}']['delta'][0][0]:.4f} m\n")
        file.write(f"u_tau = {piv[f'{number}']['utau'][0][0]:.1f} m/s\n")
        file.write(f"nu = {piv[f'{number}']['nu'][0][0]:.4e} m^2/s\n")
        file.write("\n\n")
        headers = ["X_1/H", " X_2/H", " X_3/H", " U_1/U_inf", " U_1/U_ref"]

        column_width = 11

        formatted_header = "\t".join(
            "{:<{width}}".format(header, width=column_width) for header in headers
        )
        file.write(formatted_header + "\n")
        # file.write("X_1/H\t\tX_2/H\t\tX_3/H\t\tU_1/U_inf\t\tU_1/U_ref\n")
        for ii in range(len(piv[f"{number}"]["udat"][0])):
            file.write(
                f"{piv[f'{number}']['x'][0][0]/0.186944:11.4e}\t"
                f"{piv[f'{number}']['ydat'][0][int(f'{ii}')]/0.186944:11.4e}\t"
                f"{0:11.4e}\t"
                f"{piv[f'{number}']['udat'][0][int(f'{ii}')]/reference_conditions[f'run{run}']['pressure_data'][0,4]:11.4e}\t"
                f"{piv[f'{number}']['udat'][0][int(f'{ii}')]/reference_conditions[f'run{run}']['U_ref']:11.4e}\n"
            )
