"""
This module defines functions for exporting the BeVERLI Hill PIV data to different formats.
"""
import os

import numpy as np
import tecplot as tp

from .. import utility
from ..parser import InputFile


# pylint: disable=too-many-locals
def export_data_to_tecplot_binary(piv_obj) -> int:
    """Export the PIV data as a binary (.plt) Tecplot file.

    :param piv_obj: Instance of the :py:class:`datum.piv.Piv` class.
    :return: An integer representing the functions exit code.
    """
    input_data = InputFile().data
    plane_number = input_data["piv_data"]["plane_number"]
    plane_type = input_data["piv_data"]["plane_type"]
    reynolds_number = input_data["general"]["reynolds_number"]

    variables = [
        ["coordinates", ["X", "Y", "Z"], ["X", "Y", "Z"]],
        ["mean_velocity", ["U", "V", "W"], ["X Velocity", "Y Velocity", "Z Velocity"]],
        [
            "reynolds_stress",
            ["UU", "VV", "WW", "UV", "UW", "VW"],
            ["UU", "VV", "WW", "UV", "UW", "VW"],
        ],
        ["turbulence_scales", ["TKE"], ["Turbulent Kinetic Energy"]],
    ]

    if utility.search_nested_dict(piv_obj.data, "epsilon"):
        variables[3][1].append("epsilon")
        variables[3][2].append("Turbulence Dissipation Rate")
    if utility.search_nested_dict(piv_obj.data, "instantaneous_velocity_frame"):
        variables.append(
            [
                "instantaneous_velocity_frame",
                ["U", "V", "W"],
                ["X Velocity Inst.", "Y Velocity Inst.", "Z Velocity Inst."],
            ]
        )
    if utility.search_nested_dict(piv_obj.data, "mean_velocity_gradient"):
        variables.append(
            [
                "mean_velocity_gradient",
                [
                    "dUdX",
                    "dUdY",
                    "dUdZ",
                    "dVdX",
                    "dVdY",
                    "dVdZ",
                    "dWdX",
                    "dWdY",
                    "dWdZ",
                ],
                [
                    "dUdX",
                    "dUdY",
                    "dUdZ",
                    "dVdX",
                    "dVdY",
                    "dVdZ",
                    "dWdX",
                    "dWdY",
                    "dWdZ",
                ],
            ]
        )
    if utility.search_nested_dict(piv_obj.data, "NUT"):
        variables[3][1].append("NUT")
        variables[3][2].append("<greek>n</greek><sub>t</sub>")
        variables.append(
            [
                "strain_tensor",
                [f"S_{i+1}{j+1}" for i in range(3) for j in range(3)],
                [f"S<sub>{i + 1}{j + 1}</sub>" for i in range(3) for j in range(3)],
            ]
        )
        variables.append(
            [
                "rotation_tensor",
                [f"W_{i + 1}{j + 1}" for i in range(3) for j in range(3)],
                [f"W<sub>{i + 1}{j + 1}</sub>" for i in range(3) for j in range(3)],
            ]
        )

    rows, cols = piv_obj.data["coordinates"]["X"].shape
    num_of_nodes = rows * cols

    dataset_title = f"Plane{plane_number}_{int(reynolds_number * 1e-3)}k_{plane_type}"
    cfd_components = [
        item
        for sublist in (inner_list for _, _, inner_list in variables)
        for item in sublist
    ]

    conn = []
    for i in range(rows * cols):
        if (i % rows < (rows - 1)) and (i // rows < (cols - 1)):
            v_1 = i
            v_2 = i + 1
            v_3 = i + rows + 1
            v_4 = i + rows
            quad = (v_1, v_2, v_3, v_4)
            conn.append(quad)
        else:
            continue
    conn = tuple(conn)

    tp.new_layout()
    dataset = tp.active_frame().create_dataset(dataset_title, cfd_components)
    plane_zone = dataset.add_fe_zone(
        tp.constant.ZoneType.FEQuad,
        dataset_title,
        num_of_nodes,
        len(conn),
        locations=tp.constant.ValueLocation.Nodal,
        dtypes=tp.constant.FieldDataType.Double,
    )

    for variable, components, cfd_components in variables:
        for component, cfd_component in zip(components, cfd_components):
            if variable == "coordinates":
                if component == "Z" and not input_data["piv_data"]["plane_is_diagonal"]:
                    zstring = input_data["cfd"]["zone_name"]
                    zpos = float(zstring[zstring.find("Z=") + 2 :])
                    plane_zone.values(cfd_component)[:] = np.ones((rows * cols,)) * zpos
                else:
                    plane_zone.values(cfd_component)[:] = piv_obj.data[variable][
                        component
                    ].ravel()
            else:
                plane_zone.values(cfd_component)[:] = piv_obj.data[variable][
                    component
                ].ravel()

    plane_zone.nodemap[:] = conn

    outdir = os.path.join(input_data["system"]["cfd_data_root_folder"], "Tecplot")

    os.makedirs(outdir, exist_ok=True)
    tp.data.save_tecplot_plt(os.path.join(outdir, dataset_title + ".plt"))

    return 0
