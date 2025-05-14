"""Pose calculator application window."""

import sys
import tkinter as tk
from tkinter import messagebox
from typing import Tuple, cast

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Cursor

from datum.core import transform
from datum.core.beverli import Beverli
from datum.core.my_types import TransformationParameters, SecParams
from datum.core.piv import Piv
from datum.gui.widgets import (
    Button,
    Checkbutton,
    Entry,
    FileLoader,
    Frame,
    Label,
    ScrollableCanvas,
    Section,
)
from datum.utility import apputils, tputils
from datum.utility.logging import logger
from datum.utility.configure import STYLES

# Constants
WINDOW_TITLE = "Pose Calculator"
WINDOW_SIZE = (1000, 600)
CALCULATION_MODES = (
    ("none", "LOAD all transformation parameters"),
    ("local", "LOAD global / CALCULATE local"),
    ("all", "CALCULATE global / CALCULATE local"),
)
CALIBRATION_IMG_SKIPROWS = 4
PAD_S = STYLES["pad"]["small"]


class PoseWindow:
    """Pose calculator window."""

    def __init__(
        self,
        master: tk.Toplevel,
        piv: Piv,
        geometry: Beverli,
        calculation_status: tk.BooleanVar,
    ) -> None:
        """Construct the window.

        :param master: Parent window handle.
        :param piv: PIV data.
        :param geometry: BeVERLI Hill geometry.
        :param calculation_status: Monitor variable for calculation status.
        """
        # Resources
        self.piv = piv
        self.geometry = geometry
        self.status = calculation_status

        # GUI
        self.master = master
        self.root = tk.Toplevel(master)
        self.configure_root()
        self.create_widgets()
        self.layout_widgets(calculation_mode="none")
        self.scrollable_canvas.configure_frame()
        logger.info("Pose window opened successfully.")

    def on_closing(self) -> None:
        """Free resources after closing the pose calculator."""
        if hasattr(self, "cal_fig"):
            plt.close(self.cal_fig)
        if hasattr(self, "cal_canvas"):
            self.cal_canvas.get_tk_widget().destroy()
        if hasattr(self, "glob_fig"):
            plt.close(self.glob_fig)
        if hasattr(self, "glob_canvas"):
            self.glob_canvas.get_tk_widget().destroy()
        self.root.destroy()
        logger.info("Pose window closed successfully.")

    def configure_root(self) -> None:
        """Configure the window."""
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}")
        self.root.resizable(False, False)
        self.root.configure(bg=STYLES["color"]["base"])
        self.root.option_add(
            "*Font", (STYLES["font"], STYLES["font_size"]["regular"])
        )
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self) -> None:
        """Create all widget entities."""
        self.scrollable_canvas = ScrollableCanvas(self.root, True, False)
        self.main_frame = self.scrollable_canvas.frame
        self.options_section = Section(self.main_frame, "Settings", 1)
        self.checkbox_diagonal = Checkbutton(
            self.options_section.content, 1, text="Plane is Diagonal"
        )
        self.mode_selector_label = Label(
            self.options_section.content, "Select Calculation Mode:", 1
        )
        self.mode_selector_var = tk.StringVar()
        self.mode_selector_var.set(CALCULATION_MODES[0][1])
        self.mode_selector_var.trace("w", self.on_mode_selection)
        self.mode_selector = tk.OptionMenu(
            self.options_section.content,
            self.mode_selector_var,
            *[x for _, x in CALCULATION_MODES],
        )
        self.submit_button = Button(
            self.main_frame, "Submit Transformation File", self.submit_file
        )
        self.parameters_loader = FileLoader(
            self.options_section.content,
            title="Transformation Parameters:",
            filetypes=[
                ("Transformation Parameters", "*.json"),
                ("All Files", "*.*"),
            ],
            category=1,
            isCheckable=False,
        )
        self.calplate_loader = FileLoader(
            self.options_section.content,
            title="Calibration Image:",
            filetypes=[("Calibration Image", "*.dat"), ("All Files", "*.*")],
            category=1,
            isCheckable=False,
        )
        self.calplate_loader.status_label_var.trace(
            "w",
            lambda *args: self.root.after(
                10,
                (
                    lambda: (
                        self.create_local_pose_selector(
                            self.mode_selector_var.get(), *args
                        )
                        if self.calplate_loader.status_label_var.get()
                        == "File Loaded"
                        and self.global_loader.status_label_var.get()
                        == "File Loaded"
                        else None
                    )
                ),
            ),
        )
        self.global_loader = FileLoader(
            self.options_section.content,
            title="Global parameters:",
            filetypes=[
                ("Transformation Parameters", "*.json"),
                ("All Files", "*.*"),
            ],
            category=1,
            isCheckable=False,
        )
        self.global_loader.status_label_var.trace(
            "w",
            lambda *args: self.root.after(
                10,
                (
                    lambda: (
                        self.create_local_pose_selector(
                            self.mode_selector_var.get(), *args
                        )
                        if self.calplate_loader.status_label_var.get()
                        == "File Loaded"
                        and self.global_loader.status_label_var.get()
                        == "File Loaded"
                        else None
                    )
                ),
            ),
        )
        self.measurement_loader = FileLoader(
            self.options_section.content,
            title="Pose Measurement:",
            filetypes=[("Pose Measurement", "*.json"), ("All Files", "*.*")],
            category=1,
            isCheckable=False,
        )
        self.measurement_loader.status_label_var.trace(
            "w",
            lambda *args: self.root.after(
                10,
                (
                    lambda: (
                        self.create_global_pose_calculator(
                            self.mode_selector_var.get(), *args
                        )
                        if self.calplate_loader.status_label_var.get()
                        == "File Loaded"
                        and self.measurement_loader.status_label_var.get()
                        == "File Loaded"
                        else None
                    )
                ),
            ),
        )
        self.calplate_loader.status_label_var.trace(
            "w",
            lambda *args: self.root.after(
                10,
                (
                    lambda: (
                        self.create_global_pose_calculator(
                            self.mode_selector_var.get(), *args
                        )
                        if self.calplate_loader.status_label_var.get()
                        == "File Loaded"
                        and self.measurement_loader.status_label_var.get()
                        == "File Loaded"
                        else None
                    )
                ),
            ),
        )
        self.local_section = Section(self.main_frame, "Local Pose", 2)
        self.calplate_plot = Frame(
            self.local_section.content, 2, bd=2, relief="solid"
        )
        self.picker_frame = Frame(self.local_section.content, 2)
        self.picker_button = Button(
            self.picker_frame, "Pick Location", self.pick_location
        )
        self.picker_monitors = Frame(self.picker_frame, 2)
        self.xpick_var = tk.StringVar()
        self.xpick_entry_label = Label(self.picker_monitors, "x1 [mm]:", 2)
        self.xpick_entry = Entry(
            self.picker_monitors,
            2,
            textvariable=self.xpick_var,
            state="readonly",
        )
        self.ypick_entry_label = Label(self.picker_monitors, "x2 [mm]:", 2)
        self.ypick_var = tk.StringVar()
        self.ypick_entry = Entry(
            self.picker_monitors,
            2,
            textvariable=self.ypick_var,
            state="readonly",
        )
        self.global_section = Section(self.main_frame, "Global Pose", 2)
        self.global_plot = Frame(
            self.global_section.content, 2, bd=2, relief="solid"
        )
        self.is_convex_option = Checkbutton(
            self.global_section.content,
            2,
            text="Use Convex Hill Curvature Correction",
        )
        self.use_measured_angle_option = Checkbutton(
            self.global_section.content, 2, text="Use measured angle"
        )
        self.calculate_global_button = Button(
            self.global_section.content,
            "Calculate Global",
            self.calculate_global,
        )

    def layout_widgets(self, calculation_mode: str) -> None:
        """Layout all widgets on the window for a specific calcualtion mode.

        :param calculation_mode: Identifier for the specific calculation mode.
        """
        if calculation_mode == "none":
            self.layout_widgets_default()
        elif calculation_mode == "local":
            self.layout_widgets_local()
        elif calculation_mode == "all":
            self.layout_widgets_global()

    def reset_layout(self) -> None:
        """Restore the default layout."""
        self.calplate_loader.reset()
        self.calplate_loader.grid_forget()
        self.global_loader.reset()
        self.global_loader.grid_forget()
        self.parameters_loader.reset()
        self.parameters_loader.grid_forget()
        self.xpick_var.set("")
        self.ypick_var.set("")
        self.local_section.grid_forget()
        self.measurement_loader.reset()
        self.measurement_loader.grid_forget()
        self.global_plot.grid_forget()
        self.global_section.grid_forget()

    def layout_widgets_default(self) -> None:
        """Generate the default layout."""
        self.reset_layout()
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.options_section.grid(
            row=0, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew"
        )
        self.options_section.content.grid_columnconfigure(0, weight=1)
        self.options_section.content.grid_columnconfigure(1, weight=1)
        self.checkbox_diagonal.grid(
            row=0, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew"
        )
        self.mode_selector_label.grid(
            row=1, column=0, padx=PAD_S, pady=PAD_S, sticky="w"
        )
        self.mode_selector.grid(
            row=1, column=1, padx=PAD_S, pady=PAD_S, sticky="ew"
        )
        self.parameters_loader.grid(
            row=2,
            column=0,
            columnspan=2,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.submit_button.grid(row=1, column=0, padx=PAD_S, pady=PAD_S)
        self.scrollable_canvas.configure_frame()

    def layout_widgets_local(self) -> None:
        """Generate the layout for the local mode window."""
        self.layout_widgets_default()
        self.checkbox_diagonal.grid_forget()
        self.mode_selector.grid(row=0)
        self.mode_selector_label.grid(row=0)
        self.calplate_loader.grid(
            row=1,
            column=0,
            columnspan=2,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.global_loader.grid(
            row=2,
            column=0,
            columnspan=2,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.submit_button.grid(row=2, column=0, padx=PAD_S, pady=PAD_S)
        self.scrollable_canvas.configure_frame()

    def layout_widgets_global(self) -> None:
        """Generate the layout for the global mode window."""
        self.layout_widgets_default()
        self.checkbox_diagonal.grid_forget()
        self.mode_selector.grid(row=0)
        self.mode_selector_label.grid(row=0)
        self.calplate_loader.grid(
            row=1,
            column=0,
            columnspan=2,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.measurement_loader.grid(
            row=2,
            column=0,
            columnspan=2,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.submit_button.grid(row=2, column=0, padx=PAD_S, pady=PAD_S)
        self.scrollable_canvas.configure_frame()

    def on_mode_selection(self, *args) -> None:
        """Activate the mode-specific layout based on the user's selection."""
        selected_option = self.mode_selector_var.get()
        if selected_option == CALCULATION_MODES[0][1]:
            lvl = CALCULATION_MODES[0][0]
        elif selected_option == CALCULATION_MODES[1][1]:
            lvl = CALCULATION_MODES[1][0]
        else:
            lvl = CALCULATION_MODES[2][0]
        self.layout_widgets(lvl)

    def plot_calplate(self, case: str) -> None:
        """Plot the calibration plate image.

        :param case: Calculation mode identifier.
        :raises RuntimeError: If an error occurs at any step.
        """
        if hasattr(self, "cal_fig") and hasattr(self, "cal_canvas"):
            self.cal_ax.clear()
        else:
            self.cal_fig = plt.figure(figsize=(7, 6))
            self.cal_ax = self.cal_fig.add_axes((0.12, 0.12, 0.85, 0.87))
            self.cal_canvas = FigureCanvasTkAgg(
                self.cal_fig, master=self.calplate_plot
            )
            self.cal_canvas.get_tk_widget().grid(
                row=0, column=0, sticky="nsew"
            )

        cal_img_path = self.calplate_loader.get_listbox_content()

        try:
            cal_img = np.loadtxt(
                cal_img_path, skiprows=CALIBRATION_IMG_SKIPROWS
            )
            dims = tputils.get_ijk(cal_img_path)

            img_coords_mm = np.array(
                [
                    np.reshape(cal_img[:, i], (dims[1], dims[0]))
                    for i in range(2)
                ]
            )
            img_vals = np.reshape(cal_img[:, 2], (dims[1], dims[0]))

            rotation_angle_deg = 0.0
            if case == CALCULATION_MODES[1][1]:
                tp_path = self.global_loader.get_listbox_content()
                tp = apputils.load_transformation_parameters(tp_path)
                rotation_angle_deg = tp["rotation"]["angle_1_deg"]
            elif case == CALCULATION_MODES[2][1]:
                # NOTE: this path is triggered when the global calculator
                # button is pressed, at which point we should already know
                # if global_pose is None.
                rotation_angle_deg = cast(SecParams, self.global_pose)[6]
            rotation_matrix = transform.rotation.get_rotation_matrix(
                rotation_angle_deg, "z"
            )
            img_coords_mm = transform.rotation.rotate_vector_planar(
                img_coords_mm, rotation_matrix
            )

            cmap = plt.get_cmap("gray")
            self.cal_ax.pcolormesh(
                img_coords_mm[0],
                img_coords_mm[1],
                img_vals,
                cmap=cmap,
                vmin=0,
                vmax=4000,
            )
            self.cal_ax.set_xlabel(r"$x_1$ (mm)", labelpad=10)
            self.cal_ax.set_ylabel(r"$x_2$ (mm)", labelpad=10)
            self.cal_canvas.draw()
            self.scrollable_canvas.configure_frame()
        except Exception as e:
            raise RuntimeError(
                f"An error occured while plotting the calibration image: {e}"
            )

    def pick_location(self):
        """Pick a location on the calibration plate image."""
        if not hasattr(self, "picker_cursor") or self.picker_cursor is None:
            self.picker_cursor = Cursor(
                self.cal_ax, useblit=True, color="gray", linewidth=1
            )
        self.cal_canvas.draw_idle()
        pts = plt.ginput(n=1, timeout=0, show_clicks=True)
        if not hasattr(self, "picker_mark"):
            self.picker_mark = self.cal_ax.plot(
                pts[0][0], pts[0][1], "rx", markersize=10
            )[0]
        else:
            self.picker_mark.remove()
            self.picker_mark = self.cal_ax.plot(
                pts[0][0], pts[0][1], "rx", markersize=10
            )[0]
        self.picker_cursor = None
        self.cal_canvas.draw_idle()
        self.xpick_var.set(f"{pts[0][0]}")
        self.ypick_var.set(f"{pts[0][1]}")
        self.local_pose = [
            float(self.xpick_var.get()),
            float(self.ypick_var.get()),
        ]

    def create_local_pose_selector(self, case: str, *args) -> None:
        """
        Create selector to identify the local pose from the calibration image.

        :param case: Calculation mode identifier.
        :param *args: Additional arguments.
        :raises RuntimeError: If calibration image is not properly loaded.
        """
        try:
            if not args:
                row = args[0]
            else:
                row = 3

            self.local_section.grid(
                row=row, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew"
            )
            self.calplate_plot.grid(
                row=0, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew"
            )
            self.local_section.content.grid_columnconfigure(1, weight=1)
            self.local_section.content.grid_rowconfigure(0, weight=1)
            self.submit_button.grid(row=row + 1)
            self.plot_calplate(case)
            self.picker_frame.grid(row=0, column=1, padx=PAD_S, pady=PAD_S)
            self.picker_frame.grid_columnconfigure(0, weight=1)
            self.picker_button.grid(row=0, column=0, padx=PAD_S, pady=PAD_S)
            self.picker_monitors.grid(row=1, column=0, padx=PAD_S, pady=PAD_S)
            self.picker_monitors.grid_columnconfigure(0, weight=1)
            self.picker_monitors.grid_columnconfigure(1, weight=1)
            self.xpick_entry_label.grid(
                row=0, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew"
            )
            self.ypick_entry_label.grid(
                row=0, column=1, padx=PAD_S, pady=PAD_S, sticky="nsew"
            )
            self.xpick_entry.grid(
                row=1, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew"
            )
            self.ypick_entry.grid(
                row=1, column=1, padx=PAD_S, pady=PAD_S, sticky="nsew"
            )
        except Exception as e:
            if case == CALCULATION_MODES[1][1]:
                self.layout_widgets("local")
                logger.error(str(e))
                messagebox.showerror(
                    "ERROR!",
                    "Calibration plate image or transformation parameters "
                    "could not be loaded. Check the log, fix the issue, and "
                    "try again.",
                )
                return
            elif case == CALCULATION_MODES[2][1]:
                self.layout_widgets("all")
                logger.error(str(e))
                messagebox.showerror(
                    "ERROR!",
                    "Calibration plate image "
                    "could not be loaded. Check the log, fix the issue, and "
                    "try again.",
                )
                return

    def create_global_pose_calculator(self, case: str, *args):
        """
        Create calculator for the global pose.

        :param case: Calculation mode identifier.
        :param *args: Additional arguments.
        """
        self.global_section.grid(
            row=2, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew"
        )
        self.global_section.content.grid_columnconfigure(0, weight=1)
        self.global_section.content.grid_columnconfigure(1, weight=1)
        self.global_section.content.grid_columnconfigure(2, weight=1)
        self.is_convex_option.grid(
            row=0, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew"
        )
        self.use_measured_angle_option.grid(
            row=0, column=1, padx=PAD_S, pady=PAD_S, sticky="nsew"
        )
        self.calculate_global_button.grid(
            row=0, column=2, padx=PAD_S, pady=PAD_S, sticky="nsew"
        )
        self.submit_button.grid(row=3)

    def plot_global(
        self,
        secant: Tuple[float, float, float, float, float, float, float, float],
    ):
        """
        Generate a plot that visualizes the global pose.

        :param secant: Global pose parameters obtained from the global pose
            calculation.
        """
        if hasattr(self, "glob_fig") and hasattr(self, "glob_canvas"):
            self.glob_ax.clear()
        else:
            self.glob_fig = plt.figure(figsize=(10, 4))
            self.glob_ax = self.glob_fig.add_axes((0.12, 0.12, 0.85, 0.87))
            self.glob_canvas = FigureCanvasTkAgg(
                self.glob_fig, master=self.global_plot
            )
            self.glob_canvas.get_tk_widget().grid(
                row=0, column=0, sticky="nsew"
            )

        x1_prof, x2_prof = self.geometry.calculate_x1_x2(secant[7])
        self.glob_ax.plot(x1_prof, x2_prof, color="blue")
        self.glob_ax.plot(
            [secant[2], secant[0]],
            [secant[3], secant[1]],
            color="red",
            marker="o",
            markersize=1,
            linestyle="dashed",
        )
        self.glob_ax.scatter(
            secant[4], secant[5], s=2, marker="o", c="black", zorder=500
        )
        rect = patches.Rectangle(
            (secant[0], secant[1]),
            np.sqrt(
                (secant[0] - secant[2]) ** 2 + (secant[1] - secant[3]) ** 2
            ),
            np.sqrt(
                (secant[0] - secant[2]) ** 2 + (secant[1] - secant[3]) ** 2
            ),
            angle=secant[6],
            color="red",
            alpha=0.5,
        )
        self.glob_ax.add_patch(rect)
        self.glob_ax.set_xlabel(r"$x_1$ [m]", labelpad=10)
        self.glob_ax.set_ylabel(r"$x_2$ [m]", labelpad=10)
        self.glob_ax.xaxis.set_tick_params(
            which="major", size=8, width=2, direction="in"
        )
        self.glob_ax.xaxis.set_tick_params(
            which="minor", size=5, width=1.5, direction="in"
        )
        self.glob_ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
        self.glob_ax.yaxis.set_tick_params(
            which="major", size=8, width=2, direction="in"
        )
        self.glob_ax.yaxis.set_tick_params(
            which="minor", size=5, width=1.5, direction="in"
        )
        if secant[4] < 0:
            self.glob_ax.axis((-0.55, 0, -0.025, 0.186944 + 0.15))
            self.glob_ax.set_aspect("equal", adjustable="box")
        else:
            self.glob_ax.axis((0, 0.55, -0.025, 0.186944 + 0.15))
            self.glob_ax.set_aspect("equal", adjustable="box")

        self.glob_canvas.draw()
        self.scrollable_canvas.configure_frame()

    def calculate_global(self):
        """Calculate the global pose parameters."""
        try:
            self.global_plot.grid(
                row=1,
                column=0,
                columnspan=3,
                padx=PAD_S,
                pady=PAD_S,
                sticky="nsew",
            )
            opts = {
                "apply_convex_curvature_correction": bool(
                    self.is_convex_option.var.get()
                ),
                "use_measured_rotation_angle": bool(
                    self.use_measured_angle_option.var.get()
                ),
            }
            self.global_pose = self.piv.pose.calculate_global_pose(
                self.geometry, self.measurement_loader.get_listbox_content(), opts
            )
            self.plot_global(self.global_pose)
            self.create_local_pose_selector(self.mode_selector_var.get(), 3)
        except RuntimeError as e:
            logger.error(
                f"An error occured while calculating the transformation "
                f"parameters: {e}"
            )
            messagebox.showerror(
                "ERROR!",
                "Calibration plate image or pose parameters "
                "could not be loaded. Check the log, fix the issue, and "
                "try again.",
            )
            self.layout_widgets("all")
            return

    def submit_file(self):
        """Generate the pose parameters file."""
        case = self.mode_selector_var.get()
        if case == CALCULATION_MODES[0][1]:
            tp_path = self.parameters_loader.get_listbox_content()
            try:
                tp = apputils.load_transformation_parameters(tp_path)
            except RuntimeError as e:
                logger.error(str(e))
                messagebox.showerror(
                    "ERROR!",
                    "Transformation parameters could not be loaded."
                    "Check the log, fix the issue, and "
                    "try again.",
                )
                self.layout_widgets("none")
                return
            if self.checkbox_diagonal and bool(
                self.checkbox_diagonal.var.get()
            ):
                self.piv.pose.angle1 = tp["rotation"]["angle_1_deg"]
                self.piv.pose.angle2 = tp["rotation"]["angle_2_deg"]
            else:
                self.piv.pose.angle1 = tp["rotation"]["angle_1_deg"]
                self.piv.pose.angle2 = 0.0
            self.piv.pose.loc = (
                tp["translation"]["x_1_loc_ref_mm"],
                tp["translation"]["x_2_loc_ref_mm"],
            )
            self.piv.pose.glob = (
                tp["translation"]["x_1_glob_ref_m"],
                tp["translation"]["x_2_glob_ref_m"],
                tp["translation"]["x_3_glob_ref_m"],
            )
            self.status.set(True)
            messagebox.showinfo(
                "INFO",
                "Transformation parameters successfully submitted.",
            )
            logger.info("Transformation parameters successfully submitted.")
            self.on_closing()
        elif case == CALCULATION_MODES[1][1]:
            tp_path = self.global_loader.get_listbox_content()
            try:
                tp = apputils.load_transformation_parameters(tp_path)
            except RuntimeError:
                logger.error(str(e))
                messagebox.showerror(
                    "ERROR!",
                    "Transformation parameters could not be loaded."
                    "Check the log, fix the issue, and "
                    "try again.",
                )
                self.layout_widgets("local")
                return
            self.piv.pose.angle1 = tp["rotation"]["angle_1_deg"]
            self.piv.pose.angle2 = 0.0
            self.piv.pose.glob = (
                tp["translation"]["x_1_glob_ref_m"],
                tp["translation"]["x_2_glob_ref_m"],
                tp["translation"]["x_3_glob_ref_m"],
            )
            self.piv.pose.loc = (
                float(self.xpick_var.get()),
                float(self.ypick_var.get()),
            )
            parameters: TransformationParameters = {
                "rotation": {
                    "angle_1_deg": self.piv.pose.angle1,
                    "angle_2_deg": self.piv.pose.angle2,
                },
                "translation": {
                    "x_1_glob_ref_m": self.piv.pose.glob[0],
                    "x_2_glob_ref_m": self.piv.pose.glob[1],
                    "x_3_glob_ref_m": self.piv.pose.glob[2],
                    "x_1_loc_ref_mm": self.piv.pose.loc[0],
                    "x_2_loc_ref_mm": self.piv.pose.loc[1],
                },
            }
            apputils.write_json(
                "./outputs/transformation_parameters.json",
                cast(dict, parameters),
            )
            self.status.set(True)
            messagebox.showinfo(
                "INFO",
                "Transformation parameters successfully submitted.",
            )
            logger.info("Transformation parameters successfully submitted.")
            self.on_closing()
        else:
            self.piv.pose.angle1 = self.global_pose[6]
            self.piv.pose.angle2 = 0.0
            self.piv.pose.glob = (
                self.global_pose[4],
                self.global_pose[5],
                self.global_pose[7],
            )
            self.piv.pose.loc = (
                float(self.xpick_var.get()),
                float(self.ypick_var.get()),
            )
            parameters = {
                "rotation": {
                    "angle_1_deg": self.piv.pose.angle1,
                    "angle_2_deg": self.piv.pose.angle2,
                },
                "translation": {
                    "x_1_glob_ref_m": self.piv.pose.glob[0],
                    "x_2_glob_ref_m": self.piv.pose.glob[1],
                    "x_3_glob_ref_m": self.piv.pose.glob[2],
                    "x_1_loc_ref_mm": self.piv.pose.loc[0],
                    "x_2_loc_ref_mm": self.piv.pose.loc[1],
                },
            }
            apputils.write_json(
                "./outputs/transformation_parameters.json",
                cast(dict, parameters),
            )
            self.status.set(True)
            messagebox.showinfo(
                "INFO",
                "Transformation parameters successfully submitted.",
            )
            logger.info("Transformation parameters successfully submitted.")
            self.on_closing()
