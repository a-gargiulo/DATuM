"""Functions for integrating and handling ANSYS Fluent CFD data and Tecplot files in DATuM."""
import os
import re
from typing import cast, Dict, Tuple, Optional
from .my_types import Properties, CFDRefConditions, ProfileData, ProfileCoordinates, ProfileMeanVelocity, ProfileTurbulenceScales, ProfileProperties
from ..utility import mathutils
from datum.utility.logging import logger

import numpy as np
import tecplot as tp

HILL_HEIGHT_M = 0.186944

# def extract_dimensions_from_header_line(line: str) -> Tuple[int, ...]:
#     """Extract the ijk-indices from the line of the Tecplot file containing the
#     dimensions of the corresponding ijk-ordered data.

#     :param line: A string representing the relevant line containing the dimensional
#         information.
#     :return: A tuple of shape (3, ) containing the Cartesian dimensions.
#     """
#     dimensions = []
#     # Split the line into words
#     words = line.split()
#     for word in words:
#         if word.startswith("I=") or word.startswith("J=") or word.startswith("K="):
#             # Extract the integer from the word
#             integer = int(re.sub(r"\D", "", word))
#             dimensions.append(integer)

#     return tuple(dimensions)


def load_fluent_data(
    case_file: str, data_file: str, connected: bool
) -> None:
    """
    Initialize a Tecplot session and load a Fluent dataset.

    :param case_file: A string representing the system path to the Fluent case file.
    :param data_file: A string representing the system path to the Fluent data file.
    :param connected: Boolean value indicating whether to run the code in connected
        or batch mode. If the value is true, the code will run in connected mode.
    """
    try:
        # Connect to Tecplot session
        if connected:
            tp.session.connect()
        tp.new_layout()

        # Load Fluent dataset
        tp.data.load_fluent(
            case_filenames=os.path.normpath(case_file),
            data_filenames=os.path.normpath(data_file),
        )
    except Exception as e:
        raise RuntimeError(f"Loading of Fluent data failed: {e}")


def calculate_ref(
    reynolds_number: float,
    heat_capacity_ratio: float,
    gas_constant: float
) -> CFDRefConditions:
    """
    Calculate the reference conditions for the simulations and normalize the corresponding data.

    :param reynolds_number: The Reynolds number of the experiment.
    :param heat_capacity_ratio: The fluid's heat capacity ratio.
    :param gas_constant: The fluid's gas constant.

    :return: A dictionary containing the reference conditions for the simulations or 'None', which indicates and error.
    :rtype: Optional[Dict[str, float]]
    """
    dataset = tp.active_frame().dataset

    # Inlet boundary conditions
    if reynolds_number in [250, 250000]:
        stagnation_pressure = 94220
        stagnation_temperature = 297
    elif reynolds_number in [325, 325000]:
        stagnation_pressure = 94275
        stagnation_temperature = 297
    elif reynolds_number in [650, 650000]:
        stagnation_pressure = 94450
        stagnation_temperature = 297
    else:
        raise RuntimeError("Unsupported Reynolds number.")

    # Reference locations
    reference_locations = np.array(
        [
            [-2.228, 1.85, -0.6858],
            [-2.228, 1.85, -0.4572],
            [-2.228, 1.85, -0.2286],
            [-2.228, 1.85, 0],
            [-2.228, 1.85, 0.2286],
            [-2.228, 1.85, 0.4572],
            [-2.228, 1.85, 0.6858],
        ]
    )

    # Pressure variable index within dataset
    variables = [str(variable) for variable in dataset.variables()]
    pressure_variable_index = variables.index("Pressure")

    # Find the reference ports
    res = tp.data.query.probe_on_surface(reference_locations.transpose())
    values = np.array(res.data).reshape((-1, len(reference_locations))).transpose()

    # Compute the reference conditions
    static_pressure_ref = np.array(values[:, pressure_variable_index])
    static_pressure_ref = float(np.mean(static_pressure_ref))
    if stagnation_pressure is None:
        raise RuntimeError("Reference pressure calculation failed.")
    mach_ref = np.sqrt(
        (2 / (heat_capacity_ratio - 1))
        * (
            (stagnation_pressure / static_pressure_ref)
            ** ((heat_capacity_ratio - 1) / heat_capacity_ratio)
            - 1
        )
    )
    static_temperature_ref = stagnation_temperature * (
        1 + (heat_capacity_ratio - 1) / 2 * mach_ref**2
    ) ** (-1)
    velocity_ref = mach_ref * np.sqrt(
        heat_capacity_ratio * gas_constant * static_temperature_ref
    )
    density_ref = static_pressure_ref / (gas_constant * static_temperature_ref)
    dynamic_viscosity_ref = (
        1.716e-5
        * (static_temperature_ref / 273.15) ** (3 / 2)
        * (273.15 + 110.4)
        / (static_temperature_ref + 110.4)
    )

    reference_conditions: CFDRefConditions = {
        "p_0": stagnation_pressure,
        "T_0": stagnation_temperature,
        "p_ref": static_pressure_ref,
        "T_ref": static_temperature_ref,
        "density_ref": density_ref,
        "U_ref": velocity_ref,
        "dynamic_viscosity_ref": dynamic_viscosity_ref,
    }

    return reference_conditions


def calculate_qois_and_normalize(
    reference_conditions: CFDRefConditions,
    include_reynolds_stress: bool = False,
) -> None:
    """
    Calculate the quantities of interest from the numerical data and normalize by the reference conditions.

    :raises TecplotError: If tecplot commands fail.
    :param refernce_conditions: The reference conditions for the numerical data.
    :param include_reynolds_stress: A Boolean value that specifies whether the Reynolds stress should be computed.
    """
    # Cp
    # ---------------------------------------------------------------------------------
    tp.data.operate.execute_equation(
        f"{{C<sub>p</sub>}} = "
        f"({{Pressure}}-{reference_conditions['p_ref']}) / "
        f"(0.5 * {reference_conditions['density_ref']} * "
        f"{reference_conditions['U_ref']}**2)"
    )
    logger.info("Done computing Cp")
    # ---------------------------------------------------------------------------------

    # Cf
    # ---------------------------------------------------------------------------------
    tp.data.operate.execute_equation(
        "{<greek>t</greek><sub>w</sub>} = "
        "sqrt({Wall shear-1}**2+{Wall shear-2}**2+{Wall shear-3}**2)"
    )
    tp.data.operate.execute_equation(
        f"{{C<sub>f</sub>}} = "
        f"{{<greek>t</greek><sub>w</sub>}} / "
        f"(0.5 * {reference_conditions['density_ref']} * "
        f"{reference_conditions['U_ref']}**2)"
    )
    logger.info("Done computing Cf")
    # ---------------------------------------------------------------------------------

    # x_i/H
    # ---------------------------------------------------------------------------------
    for name, component in zip(
        ["X<sub>1</sub>", "X<sub>2</sub>", "X<sub>3</sub>"], ["X", "Y", "Z"]
    ):
        equation = f"{{{name}/H}}={{{component}}} / {HILL_HEIGHT_M}"
        tp.data.operate.execute_equation(equation)
    logger.info("Done computing xi/H")
    # ---------------------------------------------------------------------------------

    # U_i/U_ref
    # ---------------------------------------------------------------------------------
    velocity_list = [
        ("U<sub>1</sub>", "X Velocity"),
        ("U<sub>2</sub>", "Y Velocity"),
        ("U<sub>3</sub>", "Z Velocity"),
    ]
    for name, component in velocity_list:
        equation = (
            f"{{{name}/U<sub>ref</sub>}} = "
            f"{{{component}}} / {reference_conditions['U_ref']}"
        )
        tp.data.operate.execute_equation(equation)
    logger.info("Done computing Ui/Uref")
    # ---------------------------------------------------------------------------------

    # Shear stress vector (cell-centered and nodal)
    # ---------------------------------------------------------------------------------
    shear_stress_list = [
        ("<greek>t</greek><sub>1,w</sub>", 1),
        ("<greek>t</greek><sub>2,w</sub>", 2),
        ("<greek>t</greek><sub>3,w</sub>", 3),
    ]
    for name, component in shear_stress_list:
        equation_1 = f"{{{name}}} = -{{Wall shear-{component}}}"
        equation_2 = f"{{{name}<sup>Nodal</sup>}} = -{{Wall shear-{component}}}"
        tp.data.operate.execute_equation(equation_1)
        tp.data.operate.execute_equation(
            equation_2, value_location=tp.constant.ValueLocation.Nodal
        )
    logger.info("Done computing shear stress")
    # ---------------------------------------------------------------------------------

    # u_tau
    # ---------------------------------------------------------------------------------
    tp.data.operate.execute_equation(
        f"{{u<sub><greek>t</greek></sub>}} = "
        f"SQRT({{C<sub>f</sub>}}/2) * {reference_conditions['U_ref']}",
        value_location=tp.constant.ValueLocation.Nodal,
    )
    logger.info("Done computing utau")

    # Compute normal vectors
    tp.macro.execute_extended_command(
        command_processor_id="CFDAnalyzer4",
        command=(
            "Calculate Function='GRIDKUNITNORMAL' Normalization='None'"
            + " ValueLocation='Nodal' CalculateOnDemand='F'"
            + " UseMorePointsForFEGradientCalculations='F'"
        ),
    )
    logger.info("Done computing normal vectors")

    # Compute reynolds stresses and TKE production
    # (use with appropriate turbulence model)
    if include_reynolds_stress:
        tp.macro.execute_extended_command(
            command_processor_id="CFDAnalyzer4",
            command=(
                "Calculate Function='VELOCITYGRADIENT' Normalization='None' "
                + "ValueLocation='CELLCENTERED' CalculateOnDemand='F' "
                + "UseMorePointsForFEGradientCalculations='T'"
            ),
        )

        reynolds_stress_list = [
            ("U<sub>1</sub>U<sub>1</sub>", ["dUdX", "dUdX"], 1),
            ("U<sub>2</sub>U<sub>2</sub>", ["dVdY", "dVdY"], 1),
            ("U<sub>3</sub>U<sub>3</sub>", ["dWdZ", "dWdZ"], 1),
            ("U<sub>1</sub>U<sub>2</sub>", ["dUdY", "dVdX"], 0),
            ("U<sub>1</sub>U<sub>3</sub>", ["dUdZ", "dWdX"], 0),
            ("U<sub>2</sub>U<sub>3</sub>", ["dVdZ", "dWdY"], 0),
        ]

        for stress, gradient, kronecker in reynolds_stress_list:
            equation = (
                f"{{{stress}}}=-({{Turbulent Viscosity}}/"
                f"reference_conditions['density_ref'])*"
                f"({{{gradient[0]}}}+{{{gradient[1]}}})+"
                f"{kronecker}*(2/3)*{{Turbulent Kinetic Energy}}"
            )

            tp.data.operate.execute_equation(equation)

        tp.data.operate.execute_equation(
            "{P} = -("
            "U<sub>1</sub>U<sub>1</sub> * {dUdX} + "
            "U<sub>1</sub>U<sub>2</sub> * {dUdY} + "
            "U<sub>1</sub>U<sub>3</sub> * {dUdZ} + "
            "U<sub>1</sub>U<sub>2</sub> * {dVdX} + "
            "U<sub>2</sub>U<sub>2</sub> * {dVdY} + "
            "U<sub>2</sub>U<sub>3</sub> * {dVdZ} + "
            "U<sub>1</sub>U<sub>3</sub> * {dWdX} + "
            "U<sub>2</sub>U<sub>3</sub> * {dWdZ} + "
            "U<sub>3</sub>U<sub>3</sub> * {dWdZ}"
            ")"
        )

        logger.info("Done computing Reynolds stress, TKE, and production")


def extract_normal_profile(
    profile_location: Tuple[float, float, float],
    number_of_profile_points: int,
    profile_height: float,
    use_sigmoid: bool,
    system_type: str,
    reference_conditions: CFDRefConditions,
    reynolds_stress_available: bool = False,
) -> Tuple[ProfileData, tp.data.dataset.Dataset]:
    """
    Extract a hill surface or tunnel wall normal profile.

    :param profile_location: Profile origin coordinates.
    :param number_of_profile_points: Number of profile points to extract.
    :param profile_height: Profile height.
    :param use_sigmoid: A boolean value indicating whether to distribute the
        profile points using a sigmoid function or linearly. If the value is
        true a sigmoid function will be used to distribute the profile points.
    :param system_type: 'Shear' or 'Tunnel'
    :param reference_conditions: CFD reference conditions.
    :param reynolds_stress_available: A boolean value indicating whether the
        Reynolds stress tensor is available within the dataset. If the value
        is true, the Reynolds stress tensor is considered available.

    :return: CFD profile data and Tecplot dataset object.
    :rtype: ProfileData
    """
    def sigmoidspace(low, high, number_of_points, shape=1):
        raw = np.tanh(np.linspace(-shape, shape, number_of_points))
        return (raw - raw[0]) / (raw[-1] - raw[0]) * (high - low) + low

    # Fluid properties
    rho = reference_conditions["density_ref"]
    dyn_viscosity = reference_conditions["dynamic_viscosity_ref"]
    kin_viscosity = dyn_viscosity / rho

    dataset = tp.active_frame().dataset

    # Hill surface
    floor_zone = dataset.zone("floor")

    # Find hill surface mesh node `closest` to surface point of profile
    query_result = tp.data.query.probe_on_surface(
        profile_location,
        zones=floor_zone,
        probe_nearest=tp.constant.ProbeNearest.Node,
        tolerance=1e-12,
    )
    profile_node_nr = query_result[1][0] - 1

    # Extract surface-normal profile
    coordinates = ["X", "Y", "Z"]
    normal_vector_components = [
        "X Grid K Unit Normal",
        "Y Grid K Unit Normal",
        "Z Grid K Unit Normal",
    ]

    surface_point = [
        floor_zone.values(i)[profile_node_nr] for i in coordinates
    ]
    if system_type == "Shear":
        end_point = [
            i + floor_zone.values(j)[profile_node_nr] * profile_height
            for i, j in zip(surface_point, normal_vector_components)
        ]
    else:
        normal_vector = [0, 1, 0]
        end_point = [
            i + j * profile_height
            for i, j in zip(surface_point, normal_vector)
        ]

    line_points = np.zeros((3, number_of_profile_points))
    for i, (j, k) in enumerate(zip(surface_point, end_point)):
        if use_sigmoid:
            line_points[i] = sigmoidspace(j, k, number_of_profile_points, shape=3)
        else:
            line_points[i] = np.linspace(j, k, number_of_profile_points)

    line = tp.data.extract.extract_line(
        zip(line_points[0], line_points[1], line_points[2])
    )
    x_1_m_cfd = floor_zone.values("X")[profile_node_nr]
    x_3_m_cfd = floor_zone.values("Z")[profile_node_nr]
    dataset.zone(dataset.num_zones - 1).name = f"Line x = {x_1_m_cfd}, z = {x_3_m_cfd}"

    rotation_matrix = None
    u_tau = floor_zone.values("u<sub><greek>t</greek></sub>")[profile_node_nr]
    if system_type == "Shear":
        # Compute profile in local shear-stress coordinates
        # ---------------------------------------------------------------------------------
        tp.data.operate.execute_equation(
            equation=(
                f"{{x<sub>2,ss</sub>}}=SQRT("
                f"({{X}}-{surface_point[0]})**2+"
                f"({{Y}}-{surface_point[1]})**2+"
                f"({{Z}}-{surface_point[2]})**2"
                f")"
            ),
            zones=line,
        )

        # Find rotation matrix
        nvec = np.array(
            [
                floor_zone.values("X Grid K Unit Normal")[profile_node_nr],
                floor_zone.values("Y Grid K Unit Normal")[profile_node_nr],
                floor_zone.values("Z Grid K Unit Normal")[profile_node_nr],
            ]
        )

        tvec = np.array(
            [
                floor_zone.values("<greek>t</greek><sub>1,w</sub><sup>Nodal</sup>")[
                    profile_node_nr
                ],
                floor_zone.values("<greek>t</greek><sub>2,w</sub><sup>Nodal</sup>")[
                    profile_node_nr
                ],
                floor_zone.values("<greek>t</greek><sub>3,w</sub><sup>Nodal</sup>")[
                    profile_node_nr
                ],
            ]
        ) / np.sqrt(
            floor_zone.values("<greek>t</greek><sub>1,w</sub><sup>Nodal</sup>")[
                profile_node_nr
            ]
            ** 2
            + floor_zone.values("<greek>t</greek><sub>2,w</sub><sup>Nodal</sup>")[
                profile_node_nr
            ]
            ** 2
            + floor_zone.values("<greek>t</greek><sub>3,w</sub><sup>Nodal</sup>")[
                profile_node_nr
            ]
            ** 2
        )

        rvec = mathutils.cross(tvec, nvec)
        rvec = rvec / np.sqrt(rvec[0] ** 2 + rvec[1] ** 2 + rvec[2] ** 2)

        rotation_matrix = np.column_stack((tvec, nvec, rvec)).T

        # Rotate velocity
        velocity_list = [
            ("U<sub>1,ss</sub>", 0),
            ("U<sub>2,ss</sub>", 1),
            ("U<sub>3,ss</sub>", 2),
        ]
        for name, component in velocity_list:
            i = component
            equation = (
                f"{{{name}}}="
                f"{{X Velocity}}*{rotation_matrix[i, 0]}+"
                f"{{Y Velocity}}*{rotation_matrix[i, 1]}+"
                f"{{Z Velocity}}*{rotation_matrix[i, 2]}"
            )
            tp.data.operate.execute_equation(
                equation, zones=dataset.zone(f"Line x = {x_1_m_cfd}, z = {x_3_m_cfd}")
            )

        # Compute velocity in wall-units
        velocity_list = [
            ("U<sub>1,ss</sub><sup>+</sup>", 1),
            ("U<sub>2,ss</sub><sup>+</sup>", 2),
            ("U<sub>3,ss</sub><sup>+</sup>", 3),
        ]
        for name, component in velocity_list:
            equation = f"{{{name}}} = {{U<sub>{component},ss</sub>}} / {u_tau}"
            tp.data.operate.execute_equation(
                equation, zones=dataset.zone(f"Line x = {x_1_m_cfd}, z = {x_3_m_cfd}")
            )

        tp.data.operate.execute_equation(
            equation=(
                f"{{x<sub>2,ss</sub><sup>+</sup>}} = "
                f"{{x<sub>2,ss</sub>}} * {u_tau} / {kin_viscosity}"
            ),
            zones=dataset.zone(f"Line x = {x_1_m_cfd}, z = {x_3_m_cfd}"),
        )

    # Sort data in profile dictionary
    profile = cfd_profile_init()

    profile_zone = dataset.zone(f"Line x = {x_1_m_cfd}, z = {x_3_m_cfd}")
    profile_variables = [
        (
            "coordinates",
            ["X", "Y", "Y_SS", "Y_SS_PLUS"] if system_type == "Shear" else ["X", "Y"],
            ["X", "Y", "x<sub>2,ss</sub>", "x<sub>2,ss</sub><sup>+</sup>"]
            if system_type == "Shear"
            else ["X", "Y"],
        ),
        (
            "mean_velocity",
            [
                "U",
                "V",
                "W",
                "U_SS",
                "V_SS",
                "W_SS",
                "U_SS_PLUS",
                "V_SS_PLUS",
                "W_SS_PLUS",
            ]
            if system_type == "Shear"
            else ["U", "V", "W"],
            [
                "X Velocity",
                "Y Velocity",
                "Z Velocity",
                "U<sub>1,ss</sub>",
                "U<sub>2,ss</sub>",
                "U<sub>3,ss</sub>",
                "U<sub>1,ss</sub><sup>+</sup>",
                "U<sub>2,ss</sub><sup>+</sup>",
                "U<sub>3,ss</sub><sup>+</sup>",
            ]
            if system_type == "Shear"
            else ["X Velocity", "Y Velocity", "Z Velocity"],
        ),
        (
            "turbulence_scales",
            ["NUT"],
            ["Turbulent Viscosity"]
        )
    ]

    if reynolds_stress_available:
        if system_type == "Shear":
            reynolds_stress_list = [
                ("U<sub>1</sub>U<sub>1,ss</sub>", [0, 0]),
                ("U<sub>2</sub>U<sub>2,ss</sub>", [1, 1]),
                ("U<sub>3</sub>U<sub>3,ss</sub>", [2, 2]),
                ("U<sub>1</sub>U<sub>2,ss</sub>", [0, 1]),
                ("U<sub>1</sub>U<sub>3,ss</sub>", [0, 2]),
                ("U<sub>2</sub>U<sub>3,ss</sub>", [1, 2]),
            ]

            for name, indices in reynolds_stress_list:
                i, j = indices
                equation = (
                    f"{{{name}}}="
                    f"{rotation_matrix[i, 0]}*{{U<sub>1</sub>U<sub>1</sub>}}*{rotation_matrix[j, 0]}+"
                    f"{rotation_matrix[i, 0]}*{{U<sub>1</sub>U<sub>2</sub>}}*{rotation_matrix[j, 1]}+"
                    f"{rotation_matrix[i, 0]}*{{U<sub>1</sub>U<sub>3</sub>}}*{rotation_matrix[j, 2]}+"
                    f"{rotation_matrix[i, 1]}*{{U<sub>1</sub>U<sub>2</sub>}}*{rotation_matrix[j, 0]}+"
                    f"{rotation_matrix[i, 1]}*{{U<sub>2</sub>U<sub>2</sub>}}*{rotation_matrix[j, 1]}+"
                    f"{rotation_matrix[i, 1]}*{{U<sub>2</sub>U<sub>3</sub>}}*{rotation_matrix[j, 2]}+"
                    f"{rotation_matrix[i, 2]}*{{U<sub>1</sub>U<sub>3</sub>}}*{rotation_matrix[j, 0]}+"
                    f"{rotation_matrix[i, 2]}*{{U<sub>2</sub>U<sub>3</sub>}}*{rotation_matrix[j, 1]}+"
                    f"{rotation_matrix[i, 2]}*{{U<sub>3</sub>U<sub>3</sub>}}*{rotation_matrix[j, 2]}"
                )
                tp.data.operate.execute_equation(
                    equation=equation, zones=dataset.zone(f"Line x = {x_1_m_cfd}")
                )

            reynolds_stress_list = [
                ("U<sub>1</sub>U<sub>1,ss</sub><sup>+</sup>", [1, 1]),
                ("U<sub>2</sub>U<sub>2,ss</sub><sup>+</sup>", [2, 2]),
                ("U<sub>3</sub>U<sub>3,ss</sub><sup>+</sup>", [3, 3]),
                ("U<sub>1</sub>U<sub>2,ss</sub><sup>+</sup>", [1, 2]),
                ("U<sub>1</sub>U<sub>3,ss</sub><sup>+</sup>", [1, 3]),
                ("U<sub>2</sub>U<sub>3,ss</sub><sup>+</sup>", [2, 3]),
            ]
            for name, component in reynolds_stress_list:
                equation = (
                    f"{{{name}}} = "
                    f"{{U<sub>{component[0]}</sub>U<sub>{component[1]},ss</sub>}} / ({u_tau} ** 2)"
                )
                tp.data.operate.execute_equation(
                    equation,
                    zones=dataset.zone(f"Line x = {x_1_m_cfd}, z = {x_3_m_cfd}"),
                )

        profile_variables.append(
            (
                "reynolds_stress",
                [
                    "UU",
                    "VV",
                    "WW",
                    "UV",
                    "UW",
                    "VW",
                    "UU_SS",
                    "VV_SS",
                    "WW_SS",
                    "UV_SS",
                    "UW_SS",
                    "VW_SS",
                    "UU_SS_PLUS",
                    "VV_SS_PLUS",
                    "WW_SS_PLUS",
                    "UV_SS_PLUS",
                    "UW_SS_PLUS",
                    "VW_SS_PLUS",
                ]
                if system_type == "Shear"
                else ["UU", "VV", "WW", "UV", "UW", "VW"],
                [
                    "U<sub>1</sub>U<sub>1</sub>",
                    "U<sub>2</sub>U<sub>2</sub>",
                    "U<sub>3</sub>U<sub>3</sub>",
                    "U<sub>1</sub>U<sub>2</sub>",
                    "U<sub>1</sub>U<sub>3</sub>",
                    "U<sub>2</sub>U<sub>3</sub>",
                    "U<sub>1</sub>U<sub>1,ss</sub>",
                    "U<sub>2</sub>U<sub>2,ss</sub>",
                    "U<sub>3</sub>U<sub>3,ss</sub>",
                    "U<sub>1</sub>U<sub>2,ss</sub>",
                    "U<sub>1</sub>U<sub>3,ss</sub>",
                    "U<sub>2</sub>U<sub>3,ss</sub>",
                    "U<sub>1</sub>U<sub>1,ss</sub><sup>+</sup>",
                    "U<sub>2</sub>U<sub>2,ss</sub><sup>+</sup>",
                    "U<sub>3</sub>U<sub>3,ss</sub><sup>+</sup>",
                    "U<sub>1</sub>U<sub>2,ss</sub><sup>+</sup>",
                    "U<sub>1</sub>U<sub>3,ss</sub><sup>+</sup>",
                    "U<sub>2</sub>U<sub>3,ss</sub><sup>+</sup>",
                ]
                if system_type == "Shear"
                else [
                    "U<sub>1</sub>U<sub>1</sub>",
                    "U<sub>2</sub>U<sub>2</sub>",
                    "U<sub>3</sub>U<sub>3</sub>",
                    "U<sub>1</sub>U<sub>2</sub>",
                    "U<sub>1</sub>U<sub>3</sub>",
                    "U<sub>2</sub>U<sub>3</sub>",
                ],
            )
        )

    for variable, names, components in profile_variables:
        for name, component in zip(names, components):
            profile[variable][name] = profile_zone.values(component).as_numpy_array()

    profile["properties"]["NU"] = kin_viscosity
    profile["properties"]["RHO"] = rho
    profile["properties"]["U_REF"] = reference_conditions["U_ref"]
    profile["properties"]["U_TAU"] = u_tau
    rds = tp.data.Dataset, tp.active_frame().dataset
    return (profile, cast(tp.data.Dataset, rds))


def cfd_profile_init() -> ProfileData:
    """Initialize cfd profile."""
    def empty() -> np.ndarray:
        return np.empty((0,))

    def coordinates_init() -> ProfileCoordinates:
        return {
            "X": empty(),
            "Y": empty(),
            "Y_SS": None,
            "Y_SS_PLUS": None,
            "Y_SS_MODELED": None,
            "Y_W": None,
        }

    def mean_velocity_init() -> ProfileMeanVelocity:
        return {
            "U": empty(),
            "V": empty(),
            "W": empty(),
            "U_SS": None,
            "V_SS": None,
            "W_SS": None,
            "U_SS_PLUS": None,
            "V_SS_PLUS": None,
            "W_SS_PLUS": None,
            "U_SS_MODELED": None,
        }

    def turbulence_scales_init() -> ProfileTurbulenceScales:
        return {
            "NUT": empty(),
        }

    def properties_init() -> ProfileProperties:
        add_cfd = True
        return {
            "NU": 0.0,
            "RHO": 0.0,
            "U_REF": 0.0,
            "U_INF": None,
            "U_TAU": None,
            "X_CORRECTION": None,
            "Y_CORRECTION": None,
            "Y_SS_CORRECTION": None,
            "ANGLE_SS_DEG": None,
            "BL_PARAMS": {
                "GRIFFIN": {
                    "DELTA": 0.0,
                    "U_E": 0.0,
                    "DELTA_STAR": 0.0,
                    "THETA": 0.0,
                },
                "VINUESA": {
                    "DELTA": 0.0,
                    "U_E": 0.0,
                    "DELTA_STAR": 0.0,
                    "THETA": 0.0,
                },
            } if not add_cfd else None
        }
    return {
        "coordinates": coordinates_init(),
        "mean_velocity": mean_velocity_init(),
        "reynolds_stress": None,
        "strain_tensor": None,
        "rotation_tensor": None,
        "normalized_rotation_tensor": None,
        "turbulence_scales": turbulence_scales_init(),
        "uncertainty": None,
        "properties": properties_init(),
    }
