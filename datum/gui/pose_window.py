"""Pose Calculator."""

import sys
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import patches
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Cursor

from ..core import transform
from ..core.piv import Piv
from ..core.beverli import Beverli
from ..utility import apputils, tputils
from ..utility.configure import STYLES
from .widgets import Button, Checkbutton, Entry, FileLoader, Frame, Label, ScrollableCanvas, Section

from typing import cast, List

# Constants
WINDOW_TITLE = "Pose Calculator"
WINDOW_SIZE = (1000, 600)
CALC_MODES = [
    "LOAD ALL transformation parameters",
    "LOAD global / CALCULATE local",
    "CALCULATE global / CALCULATE local",
]


class PoseWindow:
    """The pose calculator."""

    def __init__(self, master: tk.Toplevel, piv: Piv, geometry: Beverli, param_status: tk.BooleanVar):
        """
        Set up GUI and resources.

        :param master: The parent window.
        """
        # GUI
        self.root = tk.Toplevel(master)
        self._configure_root()
        self._create_widgets()
        self._layout_widgets("default")

        # Resources
        self.status = param_status
        self.piv = piv
        self.geometry = geometry

    def _configure_root(self):
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}")
        self.root.resizable(False, False)
        self.root.configure(bg=STYLES["color"]["base"])
        self.root.option_add("*Font", (STYLES["font"], STYLES["font_size"]["regular"]))
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _on_closing(self):
        if hasattr(self, "cal_fig"):
            plt.close(self.cal_fig)
        if hasattr(self, "cal_canvas"):
            self.cal_canvas.get_tk_widget().destroy()
        if hasattr(self, "glob_fig"):
            plt.close(self.glob_fig)
        if hasattr(self, "glob_canvas"):
            self.glob_canvas.get_tk_widget().destroy()
        self.root.destroy()

    def _create_widgets(self):
        self.scrollable_canvas = ScrollableCanvas(self.root, True, False)
        self.main_frame = self.scrollable_canvas.get_frame()
        self.opt_sect = Section(self.main_frame, "Settings", 1)
        self.mode_selector_label = Label(self.opt_sect.content, "Select Calculation Mode:", 1)
        self.mode_selector_var = tk.StringVar()
        self.mode_selector_var.set(CALC_MODES[0])
        self.mode_selector_var.trace("w", self._on_mode_selection)
        self.mode_selector = tk.OptionMenu(self.opt_sect.content, self.mode_selector_var, *CALC_MODES)
        self.submit_button = Button(self.main_frame, "Submit Transformation File", command=self._submit_file)
        self.parameters_loader = FileLoader(
            self.opt_sect.content,
            "Transformation Parameters:",
            [("Transformation Parameters", "*.json"), ("All Files", "*.*")],
            1,
            False,
        )
        self.calplate_loader = FileLoader(
            self.opt_sect.content,
            "Calibration Image:",
            [("Calibration Image", "*.dat"), ("All Files", "*.*")],
            1,
            False,
        )
        self.calplate_loader.status_label_var.trace(
            "w",
            lambda *args: (
                self._create_local_pose_selector(self.mode_selector_var.get(), *args)
                if self.calplate_loader.status_label_var.get() == "File Loaded"
                and self.global_loader.status_label_var.get() == "File Loaded"
                else None
            ),
        )
        self.global_loader = FileLoader(
            self.opt_sect.content,
            "Global parameters:",
            [("Transformation Parameters", "*.json"), ("All Files", "*.*")],
            1,
            False,
        )
        self.global_loader.status_label_var.trace(
            "w",
            lambda *args: (
                self._create_local_pose_selector(self.mode_selector_var.get(), *args)
                if self.calplate_loader.status_label_var.get() == "File Loaded"
                and self.global_loader.status_label_var.get() == "File Loaded"
                else None
            ),
        )
        self.meas_loader = FileLoader(
            self.opt_sect.content,
            "Pose Measurement:",
            [("Pose Measurement", "*.json"), ("All Files", "*.*")],
            1,
            False,
        )
        self.meas_loader.status_label_var.trace(
            "w",
            lambda *args: (
                self._create_global_pose_calculator(self.mode_selector_var.get(), *args)
                if self.calplate_loader.status_label_var.get() == "File Loaded"
                and self.meas_loader.status_label_var.get() == "File Loaded"
                else None
            ),
        )
        self.calplate_loader.status_label_var.trace(
            "w",
            lambda *args: (
                self._create_global_pose_calculator(self.mode_selector_var.get(), *args)
                if self.calplate_loader.status_label_var.get() == "File Loaded"
                and self.meas_loader.status_label_var.get() == "File Loaded"
                else None
            ),
        )
        self.local_sect = Section(self.main_frame, "Local Pose", 2)
        self.calplate_plt = Frame(self.local_sect.content, 2, bd=2, relief="solid")
        self.picker_frame = Frame(self.local_sect.content, 2)
        self.picker_button = Button(self.picker_frame, "Pick Location", command=self._pick_location)
        self.picker_monitors = Frame(self.picker_frame, 2)
        self.xpick_var = tk.StringVar()
        self.xpick_entry_label = Label(self.picker_monitors, "x1 [mm]:", 2)
        self.xpick_entry = Entry(self.picker_monitors, 2, textvariable=self.xpick_var, state="readonly")
        self.ypick_entry_label = Label(self.picker_monitors, "x2 [mm]:", 2)
        self.ypick_var = tk.StringVar()
        self.ypick_entry = Entry(self.picker_monitors, 2, textvariable=self.ypick_var, state="readonly")
        self.global_sect = Section(self.main_frame, "Global Pose", 2)
        self.global_plt = Frame(self.global_sect.content, 2, bd=2, relief="solid")
        self.is_convex_opt = Checkbutton(self.global_sect.content, 2, text="Use Convex Hill Curvature Correction")
        self.use_meas_angle_opt = Checkbutton(self.global_sect.content, 2, text="Use measured angle")
        self.calc_glob_button = Button(self.global_sect.content, "Calculate Global", command=self._calculate_global)

    def _layout_widgets(self, calc_lvl: str):
        lvl = calc_lvl.lower()
        if lvl == "default":
            self._layout_widgets_default()
        elif lvl == "global":
            self._layout_widgets_global()
        elif lvl == "local":
            self._layout_widgets_local()

    def _reset_layout(self):
        self.calplate_loader.reset()
        self.calplate_loader.grid_forget()
        self.global_loader.reset()
        self.global_loader.grid_forget()
        self.parameters_loader.reset()
        self.parameters_loader.grid_forget()
        self.xpick_var.set("")
        self.ypick_var.set("")
        self.local_sect.grid_forget()
        self.meas_loader.reset()
        self.meas_loader.grid_forget()
        self.global_plt.grid_forget()
        self.global_sect.grid_forget()

    def _layout_widgets_default(self):
        self._reset_layout()
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.opt_sect.grid(row=0, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew")
        self.opt_sect.content.grid_columnconfigure(0, weight=1)
        self.opt_sect.content.grid_columnconfigure(1, weight=1)
        self.mode_selector_label.grid(
            row=0,
            column=0,
            padx=STYLES["pad"]["small"],
            pady=STYLES["pad"]["small"],
            sticky="w"
        )
        self.mode_selector.grid(row=0, column=1, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="ew")
        self.parameters_loader.grid(
            row=1,
            column=0,
            columnspan=2,
            padx=STYLES["pad"]["small"],
            pady=STYLES["pad"]["small"],
            sticky="nsew",
        )
        self.submit_button.grid(row=1, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"])
        self.scrollable_canvas.configure_frame()

    def _layout_widgets_local(self):
        self._layout_widgets_default()
        self.calplate_loader.grid(
            row=1,
            column=0,
            columnspan=2,
            padx=STYLES["pad"]["small"],
            pady=STYLES["pad"]["small"],
            sticky="nsew",
        )
        self.global_loader.grid(
            row=2,
            column=0,
            columnspan=2,
            padx=STYLES["pad"]["small"],
            pady=STYLES["pad"]["small"],
            sticky="nsew",
        )
        self.submit_button.grid(row=2, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"])
        self.scrollable_canvas.configure_frame()

    def _layout_widgets_global(self):
        self._layout_widgets_default()
        self.calplate_loader.grid(
            row=1,
            column=0,
            columnspan=2,
            padx=STYLES["pad"]["small"],
            pady=STYLES["pad"]["small"],
            sticky="nsew",

        )
        self.meas_loader.grid(
            row=2,
            column=0,
            columnspan=2,
            padx=STYLES["pad"]["small"],
            pady=STYLES["pad"]["small"],
            sticky="nsew",
        )
        self.submit_button.grid(row=2, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"])
        self.scrollable_canvas.configure_frame()

    def _on_mode_selection(self, *args):
        selected_option = self.mode_selector_var.get()
        if selected_option == CALC_MODES[0]:
            lvl = "default"
        elif selected_option == CALC_MODES[1]:
            lvl = "local"
        else:
            lvl = "global"
        self._layout_widgets(lvl)

    def _plot_calplate(self, case: str):
        if hasattr(self, "cal_fig") and hasattr(self, "cal_canvas"):
            self.cal_ax.clear()
        else:
            self.cal_fig = plt.figure(figsize=(7, 6))
            self.cal_ax = self.cal_fig.add_axes((0.12, 0.12, 0.85, 0.87))
            self.cal_canvas = FigureCanvasTkAgg(self.cal_fig, master=self.calplate_plt)
            self.cal_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        cal_img_path = self.calplate_loader.get_listbox_content()[0]

        # TODO: a fixed value for skiprows is too specific. Generalize later.
        cal_img = np.loadtxt(cal_img_path, skiprows=4)
        dims = tputils.get_ijk(cal_img_path)
        img_coords_mm = np.array([np.reshape(cal_img[:, i], (dims[1], dims[0])) for i in range(2)])
        img_vals = np.reshape(cal_img[:, 2], (dims[1], dims[0]))

        # TODO: Handle errors more neatly
        if case == CALC_MODES[1]:
            trans_params_path = self.global_loader.get_listbox_content()[0]
            trans_params = apputils.read_json(trans_params_path)
            if trans_params is None:
                sys.exit(1)
            rotation_angle_deg = cast(dict, trans_params["rotation"])["angle_deg"]
        elif case == CALC_MODES[2]:
            rotation_angle_deg = cast(list, self.global_pose)[6]
        else:
            print("ERROR: Invalid case for pose calculator.")
            sys.exit(1)

        rotation_matrix = transform.get_rotation_matrix(rotation_angle_deg, (0, 0, 1))

        img_coords_mm = transform.rotate_planar_vector_field(img_coords_mm, rotation_matrix)
        cmap = plt.get_cmap("gray")
        self.cal_ax.pcolormesh(img_coords_mm[0], img_coords_mm[1], img_vals, cmap=cmap, vmin=0, vmax=4000)
        self.cal_ax.set_xlabel(r"$x_1$ (mm)", labelpad=10)
        self.cal_ax.set_ylabel(r"$x_2$ (mm)", labelpad=10)
        self.cal_canvas.draw()
        self.scrollable_canvas.configure_frame()

    def _pick_location(self):
        if not hasattr(self, "picker_cursor") or self.picker_cursor is None:
            self.picker_cursor = Cursor(self.cal_ax, useblit=True, color="gray", linewidth=1)
        self.cal_canvas.draw_idle()
        pts = plt.ginput(n=1, timeout=0, show_clicks=True)
        if not hasattr(self, "picker_mark"):
            self.picker_mark = self.cal_ax.plot(pts[0][0], pts[0][1], "rx", markersize=10)[0]
        else:
            self.picker_mark.remove()
            self.picker_mark = self.cal_ax.plot(pts[0][0], pts[0][1], "rx", markersize=10)[0]
        self.picker_cursor = None
        self.cal_canvas.draw_idle()
        self.xpick_var.set(f"{pts[0][0]}")
        self.ypick_var.set(f"{pts[0][1]}")
        self.local_pose = [float(self.xpick_var.get()), float(self.ypick_var.get())]

    def _create_local_pose_selector(self, case: str, *args):
        if not args:
            row = args[0]
        else:
            row = 3

        self.local_sect.grid(row=row, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew")
        self.calplate_plt.grid(row=0, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew")
        self.local_sect.content.grid_columnconfigure(1, weight=1)
        self.local_sect.content.grid_rowconfigure(0, weight=1)
        self.submit_button.grid(row=row+1)
        self._plot_calplate(case)
        self.picker_frame.grid(row=0, column=1, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"])
        self.picker_frame.grid_columnconfigure(0, weight=1)
        self.picker_button.grid(row=0, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"])
        self.picker_monitors.grid(row=1, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"])
        self.picker_monitors.grid_columnconfigure(0, weight=1)
        self.picker_monitors.grid_columnconfigure(1, weight=1)
        self.xpick_entry_label.grid(
            row=0, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.ypick_entry_label.grid(
            row=0, column=1, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.xpick_entry.grid(row=1, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew")
        self.ypick_entry.grid(row=1, column=1, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew")

    def _create_global_pose_calculator(self, case: str, *args):
        self.global_sect.grid(row=2, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew")
        self.global_sect.content.grid_columnconfigure(0, weight=1)
        self.global_sect.content.grid_columnconfigure(1, weight=1)
        self.global_sect.content.grid_columnconfigure(2, weight=1)
        self.is_convex_opt.grid(
            row=0, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.use_meas_angle_opt.grid(
            row=0, column=1, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.calc_glob_button.grid(
            row=0, column=2, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.submit_button.grid(row=3)

    def _plot_global(self, secant: List[float]):
        if hasattr(self, "glob_fig") and hasattr(self, "glob_canvas"):
            self.glob_ax.clear()
        else:
            self.glob_fig = plt.figure(figsize=(10, 4))
            self.glob_ax = self.glob_fig.add_axes((0.12, 0.12, 0.85, 0.87))
            self.glob_canvas = FigureCanvasTkAgg(self.glob_fig, master=self.global_plt)
            self.glob_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

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
        self.glob_ax.scatter(secant[4], secant[5], s=2, marker="o", c="black", zorder=500)
        rect = patches.Rectangle(
            (secant[0], secant[1]),
            np.sqrt((secant[0] - secant[2]) ** 2 + (secant[1] - secant[3]) ** 2),
            np.sqrt((secant[0] - secant[2]) ** 2 + (secant[1] - secant[3]) ** 2),
            angle=secant[6],
            color="red",
            alpha=0.5,
         )
        self.glob_ax.add_patch(rect)
        self.glob_ax.set_xlabel(r"$x_1$ [m]", labelpad=10)
        self.glob_ax.set_ylabel(r"$x_2$ [m]", labelpad=10)
        self.glob_ax.xaxis.set_tick_params(which="major", size=8, width=2, direction="in")
        self.glob_ax.xaxis.set_tick_params(which="minor", size=5, width=1.5, direction="in")
        self.glob_ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
        self.glob_ax.yaxis.set_tick_params(which="major", size=8, width=2, direction="in")
        self.glob_ax.yaxis.set_tick_params(which="minor", size=5, width=1.5, direction="in")
        if secant[4] < 0:
            self.glob_ax.axis((-0.55, 0, -0.025, 0.186944 + 0.15))
            self.glob_ax.set_aspect("equal", adjustable="box")
        else:
            self.glob_ax.axis((0, 0.55, -0.025, 0.186944 + 0.15))
            self.glob_ax.set_aspect("equal", adjustable="box")

        self.glob_canvas.draw()
        self.scrollable_canvas.configure_frame()

    def _calculate_global(self):
        self.global_plt.grid(
            row=1, column=0, columnspan=3, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        opts = {
            "apply_convex_curvature_correction":  bool(self.is_convex_opt.get_var().get()),
            "use_measured_rotation_angle": bool(self.use_meas_angle_opt.get_var().get())
        }
        self.global_pose = self.piv.pose.calculate_global_pose(self.geometry, self.meas_loader.get_listbox_content()[0], opts)
        if self.global_pose is None:
            sys.exit(-1)
        self._plot_global(self.global_pose)
        self._create_local_pose_selector(self.mode_selector_var.get(), 3)

    def _submit_file(self):
        case = self.mode_selector_var.get()
        if case == CALC_MODES[0]:
            trans_params_path = self.parameters_loader.get_listbox_content()[0]
            trans_params = apputils.read_json(trans_params_path)
            if trans_params is None:
                sys.exit(-1)
            self.piv.pose.angle = cast(float, trans_params["rotation"]["angle_deg"])
            self.piv.pose.loc[0] = cast(float, trans_params["translation"]["x_1_loc_ref_mm"])
            self.piv.pose.loc[1] = cast(float, trans_params["translation"]["x_2_loc_ref_mm"])
            self.piv.pose.glob[0] = cast(float, trans_params["translation"]["x_1_glob_ref_m"])
            self.piv.pose.glob[1] = cast(float, trans_params["translation"]["x_2_glob_ref_m"])
            self.piv.pose.glob[2] = cast(float, trans_params["translation"]["x_3_glob_ref_m"])
            self.status.set(True)
            self._on_closing()
        elif case == CALC_MODES[1]:
            trans_params_path = self.global_loader.get_listbox_content()[0]
            trans_params = apputils.read_json(trans_params_path)
            if trans_params is None:
                sys.exit(-1)
            self.piv.pose.angle = cast(float, trans_params["rotation"]["angle_deg"])
            self.piv.pose.glob[0] = cast(float, trans_params["translation"]["x_1_glob_ref_m"])
            self.piv.pose.glob[1] = cast(float, trans_params["translation"]["x_2_glob_ref_m"])
            self.piv.pose.glob[2] = cast(float, trans_params["translation"]["x_3_glob_ref_m"])
            self.piv.pose.loc[0] = float(self.xpick_var.get())
            self.piv.pose.loc[1] = float(self.ypick_var.get())
            parameters = {
                    "rotation": {
                        "angle_deg": self.piv.pose.angle
                    },
                    "translation": {
                        "x_1_glob_ref_m": self.piv.pose.glob[0],
                        "x_2_glob_ref_m": self.piv.pose.glob[1],
                        "x_3_glob_ref_m": self.piv.pose.glob[2],
                        "x_1_loc_ref_mm": self.piv.pose.loc[0],
                        "x_2_loc_ref_mm": self.piv.pose.loc[1]
                    }
            }
            apputils.write_json("./outputs/test.json", parameters)
            self.status.set(True)
            self._on_closing()
        else:
            self.piv.pose.angle = cast(list, self.global_pose)[6]
            self.piv.pose.glob[0] = cast(list, self.global_pose)[4]
            self.piv.pose.glob[1] = cast(list, self.global_pose)[5]
            self.piv.pose.glob[2] = cast(list, self.global_pose)[7]
            self.piv.pose.loc[0] = float(self.xpick_var.get())
            self.piv.pose.loc[1] = float(self.ypick_var.get())
            parameters = {
                    "rotation": {
                        "angle_deg": self.piv.pose.angle
                    },
                    "translation": {
                        "x_1_glob_ref_m": self.piv.pose.glob[0],
                        "x_2_glob_ref_m": self.piv.pose.glob[1],
                        "x_3_glob_ref_m": self.piv.pose.glob[2],
                        "x_1_loc_ref_mm": self.piv.pose.loc[0],
                        "x_2_loc_ref_mm": self.piv.pose.loc[1]
                    }
            }
            apputils.write_json("./outputs/test.json", parameters)
            self.status.set(True)
            self._on_closing()
