"""This module defines the main routine."""
import datum

datum.print_title()

piv = datum.load_matlab_data()
datum.preprocess_data(piv)

piv_no_interpolation = datum.load_matlab_data()
datum.extract_profile_data(piv_no_interpolation, piv)

# Check Contour
datum.plotting.plot_contour(
     piv.data["coordinates"],
     piv.data["mean_velocity"]["U"],
     properties={
         "colormap": "jet",
         "contour_range": {"start": 0, "end": 25, "num_of_contours": 100},
         "zpos": 0,
         "xlim": [-0.5, -0.4],
         "ylim": [0.0, 0.1],
         "xmajor_locator": 0.05,
         "ymajor_locator": None,
         "cbar_range": {"start": 0, "end": 25, "num_of_ticks": 6},
         "cbar_label": r"$\overline{U}_1$ (m/s)",
     },
     outname="contour_mean_U1"
 )
# datum.print_title()
# piv = datum.load_matlab_data()
# piv.preprocess_data()
#
# # Export to Tecplot
# # datum.export_data_to_tecplot_binary(piv)
#
# # Extract profile data (from non-interpolated data)
# piv_no_intrp = datum.load_matlab_data()
# datum.Piv.extract_profile_data(piv_no_intrp, piv)

# base_tensors = datum.analysis.get_base_tensors(piv)
# rho = datum.analysis.qcr_alignment(base_tensors,2000)
# rho = datum.analysis.gatski_and_speziale_alignment(base_tensors,piv.data["turbulence_scales"]["epsilon"],"LRR",3)
# Check Contour
# datum.plotting.plot_contour(
#     piv.data["coordinates"],
#     piv.data["reynolds_stress"]["UU"],
#     properties={
#         "colormap": "jet",
#         "contour_range": {"start": 0, "end": 25, "num_of_contours": 100},
#         "zpos": 0,
#         "xlim": [0.2, 0.3],
#         "ylim": [0.12, 0.25],
#         "xmajor_locator": 0.05,
#         "ymajor_locator": None,
#         "cbar_range": {"start": 0, "end": 25, "num_of_ticks": 6},
#         "cbar_label": r"$\overline{u_1^2}$ (m$^2$/$s^2$)",
#     },
#     outname="contour_mean_UU"
# )

# datum.plotting.plot_contour(
#     piv.data["coordinates"],
#     rho,
#     properties={
#         "colormap": "jet",
#         "contour_range": {"start": 0, "end": 1, "num_of_contours": 100},
#         "zpos": 0,
#         "xlim": [-0.475, -0.410],
#         "ylim": [0, 0.080],
#         "xmajor_locator": 0.025,
#         "ymajor_locator": None,
#         "cbar_range": {"start": 0, "end": 1, "num_of_ticks": 11},
#         "cbar_label": r"$\rho_{RS}$ (m/s)",
#     },
#     outname="contour_rho_gatski_3rd"
# )


# datum.plotting.plot_contour(
#     piv.data["coordinates"],
#     piv.data["turbulence_scales"]["NUT_BOUS"],
#     properties={
#         "colormap": "jet",
#         "contour_range": {"start": 0, "end": 0.005, "num_of_contours": 100},
#         "zpos": 0,
#         "xlim": [-0.475, -0.410],
#         "ylim": [0, 0.080],
#         "xmajor_locator": 0.025,
#         "ymajor_locator": None,
#         "cbar_range": {"start": 0, "end": 0.005, "num_of_ticks": 6},
#         "cbar_label": r"$\nu_t$ (m$^2$/s)",
#     },
#     outname="nut_bous"
# )
