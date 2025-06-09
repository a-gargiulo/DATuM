"""Transform PIV 2D data to Tecplot."""
import numpy as np
import pickle
import json
import matlab.engine


# Inputs
PIV_FILE = "../../outputs/plane1_250k/plane1_pp_interp.pkl"
TRANS_PARAMS = "../../outputs/plane1_250k/plane1_tp.json"
ZONE_NAME = "Plane 1"
OUTNAME = "Plane1_250.plt"
DAT2LOAD = {
        "mean_velocity": ("U", "V", "W"),
        "reynolds_stress": ("UU", "VV", "WW", "UV", "UW", "VW"),
        "velocity_snapshot": ("U", "V", "W"),
        "turbulence_scales": ("TKE", "EPSILON", "NUT"),
        "mean_velocity_gradient": (
            "dUdX", "dUdY", "dUdZ",
            "dVdX", "dVdY", "dVdZ",
            "dWdX", "dWdY", "dWdZ",
        ),
        "strain_tensor": (
            "S_11", "S_12", "S_13",
            "S_21", "S_22", "S_23",
            "S_31", "S_32", "S_33",
        ),
        "rotation_tensor": (
            "W_11", "W_12", "W_13",
            "W_21", "W_22", "W_23",
            "W_31", "W_32", "W_33",
        ),
        "normalized_rotation_tensor": (
            "O_11", "O_12", "O_13",
            "O_21", "O_22", "O_23",
            "O_31", "O_32", "O_33",
        )
}

with open(PIV_FILE, "rb") as f:
    piv = pickle.load(f)

with open(TRANS_PARAMS, "r") as f:
    tp = json.load(f)

vrs = ["X", "Y", "Z"]
for quantity, variables in DAT2LOAD.items():
    for v in variables:
        if quantity == "velocity_snapshot":
            vrs.append(v + "inst")
        else:
            vrs.append(v)

m, n = piv["coordinates"]["X"].shape
nvars = len(vrs)
Z = np.ones_like(piv["coordinates"]["X"]) * tp["translation"]["x_3_glob_ref_m"]
tdata = {
    "Nvar": nvars,
    "varnames": vrs,
    "vformat": matlab.double([2] * nvars),
    "surfaces": {
        "zonename": ZONE_NAME,
        "order": 3,
        "varloc": 0,
        "x": matlab.double(piv["coordinates"]["X"].tolist()),
        "y": matlab.double(piv["coordinates"]["Y"].tolist()),
        "z": matlab.double(Z.tolist()),
        'v': matlab.double(np.zeros((nvars - 3, m, n)).tolist())
    }
}
c = 0
for quantity, variables in DAT2LOAD.items():
    for v in variables:
        tdata["surfaces"]["v"][c][:][:] = matlab.double(
            piv[quantity][v].tolist()
        )
        c += 1

eng = matlab.engine.start_matlab()
eng.workspace["tdata"] = tdata
eng.mat2tecplot(eng.workspace["tdata"], OUTNAME, nargout=1)
