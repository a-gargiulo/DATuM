import pickle
import numpy as np
import matplotlib.pyplot as plt

data_path = "/Users/galdo/Desktop/direct-analysis-turbulence-models/data/piv/experiment/plane3/preprocessed/plane3_250k_FS_profiles.pkl"

with open(data_path, "rb") as file:
    profiles = pickle.load(file)

prof_num = 3

y = profiles[f"profile_{prof_num}"]["exp"]["coordinates"]["Y_SS"]
uv = profiles[f"profile_{prof_num}"]["exp"]["reynolds_stress"]["UV_SS"]

uv_bous = (
    -2
    * profiles[f"profile_{prof_num}"]["exp"]["turbulence_scales"]["NUT_BOUS"]
    * profiles[f"profile_{prof_num}"]["exp"]["strain_tensor"]["S_12_SS"]
)
#
# for keys in profiles[f"profile_{prof_num}"]["exp"]["properties"]["integral_parameters"]["griffin"]:
#     print(keys)

uv_qcr = 0.1*2*profiles[f"profile_{prof_num}"]["exp"]["turbulence_scales"]["NUT_QCR"] * (
    profiles[f"profile_{prof_num}"]["exp"]["strain_tensor"]["S_12_SS"]
    + 0.3
    * (
        profiles[f"profile_{prof_num}"]["exp"]["strain_tensor"]["S_11_SS"]
        * profiles[f"profile_{prof_num}"]["exp"]["normalized_rotation_tensor"]["O_12_SS"]
        + profiles[f"profile_{prof_num}"]["exp"]["strain_tensor"]["S_12_SS"]
        * profiles[f"profile_{prof_num}"]["exp"]["normalized_rotation_tensor"]["O_22_SS"]
        + profiles[f"profile_{prof_num}"]["exp"]["strain_tensor"]["S_13_SS"]
        * profiles[f"profile_{prof_num}"]["exp"]["normalized_rotation_tensor"]["O_32_SS"]
        + profiles[f"profile_{prof_num}"]["exp"]["normalized_rotation_tensor"]["O_11_SS"]
        * profiles[f"profile_{prof_num}"]["exp"]["strain_tensor"]["S_21_SS"]
        + profiles[f"profile_{prof_num}"]["exp"]["normalized_rotation_tensor"]["O_12_SS"]
        * profiles[f"profile_{prof_num}"]["exp"]["strain_tensor"]["S_22_SS"]
        + profiles[f"profile_{prof_num}"]["exp"]["normalized_rotation_tensor"]["O_13_SS"]
        * profiles[f"profile_{prof_num}"]["exp"]["strain_tensor"]["S_32_SS"]
    )
)

# deriv = 0.28 #-0.28,-0.26,-0.29 plane 3
# deriv = 0.16 #0.16,0.14,0.19 plane 1
# deriv = 0.17 #-0.17,-0.24.-0.12 plane 4
#
# umax = np.nanmax(profiles[f"profile_{prof_num}"]["exp"]["mean_velocity"]["U_SS"])
# umin = np.nanmin(profiles[f"profile_{prof_num}"]["exp"]["mean_velocity"]["U_SS"])
# idx = np.nanargmax(profiles[f"profile_{prof_num}"]["exp"]["mean_velocity"]["U_SS"])
# bmax = y[idx]
#
rho = 1.103
# deriv = (profiles["profile_3"]["exp"]["properties"]["integral_parameters"]["griffin"]["DELTA"] - profiles["profile_1"]["exp"]["properties"]["integral_parameters"]["griffin"]["DELTA"])/(profiles["profile_3"]["exp"]["coordinates"]["X"][0] - profiles["profile_1"]["exp"]["coordinates"]["X"][0])
# # deriv = -0.18
# delta = profiles[f"profile_{prof_num}"]["exp"]["properties"]["integral_parameters"]["griffin"]["DELTA"]
# uv_egolf = rho * y * deriv * profiles[f"profile_{prof_num}"]["exp"]["mean_velocity"]["U_SS"] * (umax-profiles[f"profile_{prof_num}"]["exp"]["mean_velocity"]["U_SS"])/(0.35-y)
#






plt.figure()
plt.semilogx(y, -rho*uv)
plt.semilogx(y, -rho*uv_bous)
plt.semilogx(y, -rho*uv_qcr)
# plt.semilogx(y, -uv_egolf)
plt.xlim([1e-4, 1e-1])
plt.show()
