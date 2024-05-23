"""This module provides a small GUI routine to Spalding fit velocity profiles."""
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider


def spalding_profile(u_plus: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """The Spalding profile function.

    :param u_plus: Boundary layer velocity normalized by wall units as NumPy ndarray of
        shape (n,), where n represents the number of profile points, or a float value.
    :return: The boundary layer height y:sup:`+` normalized by wall units as NumPy
        ndarray of shape (n,), where n represents the number of profile points, or a
        float value.
    """
    kappa = 0.41
    b_const = 5
    return u_plus + np.exp(-kappa * b_const) * (
        np.exp(kappa * u_plus)
        - 1
        - (kappa * u_plus)
        - (kappa * u_plus) ** 2 / 2
        - (kappa * u_plus) ** 3 / 6
    )


def spalding_fit_profile(
    profile: Dict[str, Dict[str, Dict[str, Union[np.ndarray, float]]]], add_cfd: bool
) -> None:
    """Spalding fit routine with GUI.

    :param profile: The profile to Spalding fit as NumPy ndarray of shape (n,), where n
        represents the number of profile points.
    :param add_cfd: A boolean value indicating whether to add reference CFD data to the
        Spalding fit or not. If the value is true, CFD data will be added.
    """
    in_to_m = 0.0254
    hill_top_width_in = 3.68

    # Spalding profile parameters
    kappa_1 = 0.41
    kappa_2 = 0.384
    number_of_spalding_profile_points = 1000
    u_plus_spalding = np.linspace(0, 25, number_of_spalding_profile_points)

    # GUI Initialization
    x_2_ss_m = profile["exp"]["coordinates"]["Y_SS"]
    x_1_m = profile["exp"]["coordinates"]["X"][0]
    u_1_ss = profile["exp"]["mean_velocity"]["U_SS"]
    kinematic_viscosity = profile["exp"]["properties"]["NU"]

    u_tau_init = 1.5
    x_2_ss_0_init = 0
    x_2_ss_plus_init = (x_2_ss_m - x_2_ss_0_init) * u_tau_init / kinematic_viscosity
    u_1_ss_plus_init = u_1_ss / u_tau_init

    plt.close()

    # Appearance
    plt.rcParams["font.family"] = "Avenir"
    plt.rcParams["font.size"] = "18"
    plt.rcParams["lines.linewidth"] = "2"
    plt.rcParams["axes.linewidth"] = "2"

    # Initialize Figure 1: `Velocity Profile`
    fig_1 = plt.figure(figsize=(10, 6))
    axs_1 = fig_1.add_axes([0.08, 0.2, 0.6, 0.73])
    axs_1.semilogx(
        spalding_profile(u_plus_spalding), u_plus_spalding, color="blue"
    )  # Spalding profile
    (line_1,) = axs_1.semilogx(
        x_2_ss_plus_init,
        u_1_ss_plus_init,
        color="black",
        linestyle="--",
        label=f"{x_1_m} m",
    )
    if add_cfd:
        x_2_ss_cfd = profile["cfd"]["coordinates"]["Y_SS"]
        x_1_m_cfd = profile["cfd"]["coordinates"]["X"][0]
        u_1_ss_cfd = profile["cfd"]["mean_velocity"]["U_SS"]
        u_tau_cfd = profile["cfd"]["properties"]["U_TAU"]
        nu_cfd = profile["cfd"]["properties"]["NU"]
        x_2_ss_plus_cfd = x_2_ss_cfd * u_tau_cfd / nu_cfd
        u_1_ss_plus_cfd = u_1_ss_cfd / u_tau_cfd
        # x_2_ss_plus_cfd = profile["cfd"]["coordinates"]["Y_SS_PLUS"]
        # u_1_ss_plus_cfd = profile["cfd"]["mean_velocity"]["U_SS_PLUS"]
        axs_1.semilogx(
            x_2_ss_plus_cfd,
            u_1_ss_plus_cfd,
            color="r",
            linestyle="--",
            label=f"{x_1_m_cfd} m",
        )

    # Labels and annotations
    axs_1.set_xlabel(r"$x_2^{+}$", labelpad=10)
    axs_1.set_ylabel(r"$u_1^{+}$", labelpad=10)
    axs_1.legend(loc="upper left")

    # Add sliders and buttons to GUI
    axs_u_tau_slider = fig_1.add_axes([0.75, 0.2, 0.03, 0.73])
    u_tau_slider = Slider(
        ax=axs_u_tau_slider,
        label=r"$u_\tau$ [m/s]",
        valmin=0,
        valmax=3,
        valinit=1.5,
        orientation="vertical",
    )

    axs_y_0_slider = fig_1.add_axes([0.9, 0.2, 0.03, 0.73])
    x_2_ss_0_slider = Slider(
        ax=axs_y_0_slider,
        label=r"$x_{2,0}$ [m]",
        valmin=-0.01,
        valmax=0.01,
        valinit=0,
        orientation="vertical",
    )

    axs_submit_button = fig_1.add_axes([0.79, 0.05, 0.1, 0.06])
    submit_button = Button(axs_submit_button, "Submit", hovercolor="0.975")

    # Initialize Figure 2: `Profile Diagnostics`
    fig_2 = plt.figure(figsize=(6, 6))
    axs_2 = fig_2.add_axes([0.2, 0.15, 0.7, 0.75])
    zeta = x_2_ss_plus_init * np.gradient(u_1_ss_plus_init, x_2_ss_plus_init)
    (line_2,) = axs_2.semilogx(x_2_ss_plus_init, zeta)
    axs_2.semilogx(
        x_2_ss_plus_init,
        np.ones_like(x_2_ss_plus_init) * (1 / kappa_1),
        color="k",
        linestyle="--",
        label=f"$\\kappa={kappa_1}$",
    )
    axs_2.semilogx(
        x_2_ss_plus_init,
        np.ones_like(x_2_ss_plus_init) * (1 / kappa_2),
        color="b",
        linestyle="--",
        label=f"$\\kappa={kappa_2}$",
    )

    # Labels and annotations
    axs_2.set_xlabel(r"$x_2^{+}$", labelpad=10)
    axs_2.set_ylabel(r"$x_2^{+}\cdot\frac{du_1^{+}}{dx_2^{+}}$", labelpad=10)
    axs_2.xaxis.set_tick_params(which="major", size=8, width=2, direction="in")
    axs_2.xaxis.set_tick_params(which="minor", size=5, width=1.5, direction="in")
    axs_2.yaxis.set_tick_params(which="major", size=8, width=2, direction="in")
    axs_2.yaxis.set_tick_params(which="minor", size=5, width=1.5, direction="in")
    axs_2.set_xlim(1e-2, 1e4)
    axs_2.set_ylim(0, 15)
    axs_2.legend(loc="upper left")

    # Event handler functions
    def update(val):
        y_plus_current = (
            (x_2_ss_m - x_2_ss_0_slider.val) * u_tau_slider.val / kinematic_viscosity
        )
        u_plus_current = u_1_ss / u_tau_slider.val
        line_1.set_xdata(y_plus_current)
        line_1.set_ydata(u_plus_current)
        line_2.set_xdata(y_plus_current)
        line_2.set_ydata(y_plus_current * np.gradient(u_plus_current, y_plus_current))
        fig_1.canvas.draw_idle()
        fig_2.canvas.draw_idle()

    # register the update function with each slider
    u_tau_slider.on_changed(update)
    x_2_ss_0_slider.on_changed(update)

    # Appearance Figure 1 (to make sure the plot doesn't keep changing appearance)
    axs_1.xaxis.set_tick_params(which="major", size=8, width=2, direction="in")
    axs_1.xaxis.set_tick_params(which="minor", size=5, width=1.5, direction="in")
    axs_1.yaxis.set_tick_params(which="major", size=8, width=2, direction="in")
    axs_1.yaxis.set_tick_params(which="minor", size=5, width=1.5, direction="in")
    axs_1.set_xlim(1e-2, 1e4)
    axs_1.set_ylim(0, 25)

    def submit(event):
        plt.close()

    submit_button.on_clicked(submit)

    plt.show()
    fig_1.clear()
    fig_2.clear()

    # Set values
    profile["exp"]["properties"]["U_TAU"] = u_tau_slider.val
    profile["exp"]["properties"]["Y_SS_CORRECTION"] = x_2_ss_0_slider.val

    # Correction in tunnel coordinates
    if x_1_m < hill_top_width_in * in_to_m / 2:
        profile["exp"]["properties"]["X_CORRECTION"] = x_2_ss_0_slider.val * np.cos(
            np.pi / 2 - np.deg2rad(profile["exp"]["properties"]["ANGLE_SS_DEG"])
        )
        profile["exp"]["properties"]["Y_CORRECTION"] = x_2_ss_0_slider.val * np.sin(
            np.pi / 2 - np.deg2rad(profile["exp"]["properties"]["ANGLE_SS_DEG"])
        )
    elif x_1_m > hill_top_width_in * in_to_m / 2:
        profile["exp"]["properties"]["X_CORRECTION"] = x_2_ss_0_slider.val * np.cos(
            np.pi / 2 + np.deg2rad(profile["exp"]["properties"]["ANGLE_SS_DEG"])
        )
        profile["exp"]["properties"]["Y_CORRECTION"] = x_2_ss_0_slider.val * np.sin(
            np.pi / 2 + np.deg2rad(profile["exp"]["properties"]["ANGLE_SS_DEG"])
        )
    else:
        profile["exp"]["properties"]["X_CORRECTION"] = 0
        profile["exp"]["properties"]["Y_CORRECTION"] = x_2_ss_0_slider.val

