"""Pose Calculator."""

import sys
import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Cursor

from ..core import transform
from ..core.piv import Piv
from ..utility import apputils, tputils
from ..utility.configure import STYLES
from .widgets import Button, Entry, FileLoader, Frame, Label, ScrollableCanvas, Section


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

    def __init__(self, master: tk.Toplevel, piv: Piv):
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
        self.piv = piv

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
            [("Pose Measurement", "*.txt"), ("All Files", "*.*")],
            1,
            False,
        )
        self.meas_loader.status_label_var.trace(
            "w",
            lambda *args: (
                self._run_global_pose_calculator(self.mode_selector_var.get(), *args)
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
        self.xpick_var.set("")
        self.ypick_var.set("")
        self.local_sect.grid_forget()

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
        else:
            print("ERROR: Invalid case for pose calculator.")
            sys.exit(1)

        rotation_angle_deg = trans_params["rotation"]["angle_deg"]
        if isinstance(rotation_angle_deg, float):
            rotation_matrix = transform.get_rotation_matrix(rotation_angle_deg, (0, 0, 1))
        else:
            print("ERROR: Invalid rotation angle")
            sys.exit(1)

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

    def _create_local_pose_selector(self, case: str, *args):
        self.local_sect.grid(row=2, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew")
        self.calplate_plt.grid(row=0, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew")
        self.local_sect.content.grid_columnconfigure(1, weight=1)
        self.local_sect.content.grid_rowconfigure(0, weight=1)
        self.submit_button.grid(row=3)
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

    def _run_global_pose_calculator(self, case: str, *args):
        self.global_sect.grid(row=2, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew")
        self.global_plt.grid(
            row=0, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.global_sect.content.grid_columnconfigure(0, weight=1)

    def _submit_file(self):
        pass
