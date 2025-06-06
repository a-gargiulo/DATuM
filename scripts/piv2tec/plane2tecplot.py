import numpy as np
import pickle
import json
import scipy.io as scio
import matlab.engine
import sys
import os


# Constants
IN2M = 0.0254
H = 0.186944
W = 36.8 * IN2M
S = 3.68 * IN2M


# Inputs
PIV_FILE = "../../outputs/plane1_650k/plane1_pp_interp.pkl"
TRANS_PARAMS = "../../outputs/plane1_250k/plane1_tp.pkl"

DAT2LOAD = {
        "coordinates": ("X", "Y", "Z"),
        "mean_velocity": ("U", "V", "W"),
        "reynolds_stress": ("UU", "VV", "WW", "UV", "UW", "VW"),
        # "velocity_snapshot": ("U", "V", "W"),
        # "turbulence_scales": ("TKE", "NUT"),
        "turbulence_scales": ("TKE",),
        "mean_velocity_gradient": (
            "dUdX", "dUdY", "dUdZ",
            "dVdX", "dVdY", "dVdZ",
            "dWdX", "dWdY", "dWdZ",
        ),
        "strain_tensor": (
            "S11", "S12", "S13",
            "S21", "S22", "S23",
            "S31", "S32", "S33",
        ),
        "rotation_tensor": (
            "W11", "W12", "W13",
            "W21", "W22", "W23",
            "W31", "W32", "W33",
        ),
        "normalized_tensor": (
            "O11", "O12", "O13",
            "O21", "O22", "O23",
            "O31", "O32", "O33",
        )
}

with open(PIV_FILE, "rb") as f:
    piv = pickle.load(f)

with open(TRANS_PARAMS, "r") as f:
    tp = json.load(f)

vrs = []
for quantity, variables in DAT2LOAD.items():
    for v in variables:
        if quantity == "velocity_snapshot":
            vrs.append(v + "inst")
        else:
            vrs.append(v)

tdata = {
    "Nvar": len(vrs),
    "varnames": vrs,
    "vformat": matlab.double([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
         'surfaces': {'zonename': 'Plane'+str(pivPln),
                      'order': 3,
                      'varloc': 0,
                      'x': matlab.double(x.tolist()),
                      'y': matlab.double(y.tolist()),
                      'z': matlab.double(z.tolist()),
                      'v': matlab.double(np.zeros((12,pivData['coordinates']['X'].shape[0],pivData['coordinates']['X'].shape[1])).tolist())
                     }
         }



tdata['surfaces']['v'][0][:][:] = matlab.double(u.tolist())
tdata['surfaces']['v'][1][:][:] = matlab.double(v.tolist())
tdata['surfaces']['v'][2][:][:] = matlab.double(w.tolist())
tdata['surfaces']['v'][3][:][:] = matlab.double(uu.tolist())
tdata['surfaces']['v'][4][:][:] = matlab.double(vv.tolist())
tdata['surfaces']['v'][5][:][:] = matlab.double(ww.tolist())
tdata['surfaces']['v'][6][:][:] = matlab.double(uv.tolist())
tdata['surfaces']['v'][7][:][:] = matlab.double(uw.tolist())
tdata['surfaces']['v'][8][:][:] = matlab.double(vw.tolist())
tdata['surfaces']['v'][9][:][:] = matlab.double(uInst.tolist())
tdata['surfaces']['v'][10][:][:] = matlab.double(vInst.tolist())
tdata['surfaces']['v'][11][:][:] = matlab.double(wInst.tolist())
# tdata['surfaces']['v'][12][:][:] = matlab.double(epsilon.tolist())

eng = matlab.engine.start_matlab()
eng.workspace['tdata'] = tdata
eng.mat2tecplot(eng.workspace['tdata'],'Plane'+str(pivPln)+'_CAD.plt',nargout=1)
