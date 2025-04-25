"""Configuration file for BeVERLI Hill PIV data transformations."""

DATA_TO_ROTATE = [
    ("coordinates", ["X", "Y"]),
    ("mean_velocity", ["U", "V", "W"]),
    (
        "reynolds_stress",
        ["UU", "UV", "UW", "UV", "VV", "VW", "UW", "VW", "WW"],
    ),
    ("instantaneous_velocity_frame", ["U", "V", "W"]),
]
