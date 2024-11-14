"""Pose Calculator."""
import sys
import tkinter as tk
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Cursor

from ..core.piv import Piv
from ..core import transform
from ..utility.configure import STYLES
from ..utility import apputils, tputils
from .widgets import Button, Entry, FileLoader, Frame, Label, ScrollableCanvas, Section

# Constants
WINDOW_TITLE = "Pose Calculator"
WINDOW_SIZE = (1000, 600)
CASES = [
    "Load transformation file.",
    "Calculate local and load global pose.",
    "Calculate local and global pose.",
]


class PoseWindow:
    """The pose calculator applet."""

    def __init__(self, master: tk.Toplevel, piv: Piv):
        """
        Set up the GUI.

        :param master: The master window calling the applet.
        """
        self.root = tk.Toplevel(master)
        self._configure_root()
        # self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self._create_widgets()
        self._layout_widgets_default()
        self.scrollable_canvas.configure_frame()

        self.piv = piv

    def _configure_root(self):
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}")
        self.root.resizable(False, False)
        self.root.configure(bg=STYLES["color"]["base"])
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.option_add("*Font", (STYLES["font"], STYLES["font_size"]["regular"]))

    def _on_closing(self):
        if hasattr(self, "cal_fig"):
            plt.close(self.cal_fig)

        if hasattr(self, "cal_canvas"):
            self.cal_canvas.get_tk_widget().grid_forget()
            self.cal_canvas.get_tk_widget().destroy()

        self.root.destroy()

    def _create_widgets(self):
        self.scrollable_canvas = ScrollableCanvas(self.root, True, False)
        self.main_frame = self.scrollable_canvas.get_frame()

        self.selection_sect = Section(self.main_frame, "Settings", 1)
        self.mode_selector_label = Label(
            self.selection_sect.content, "Select Calculation Mode:", 1
        )
        self.mode_selector_var = tk.StringVar()
        self.mode_selector_var.set(CASES[0])
        self.mode_selector_var.trace("w", self._on_selection)
        self.mode_selector = tk.OptionMenu(
            self.selection_sect.content, self.mode_selector_var, *CASES
        )

        self.submit_button = Button(
            self.main_frame,
            "Submit Transformation File",
            command=self._submit_file,
        )

        self.calplate_loader = FileLoader(
            self.selection_sect.content,
            "Calibration Image:",
            [("Calibration Image", "*.dat"), ("All Files", "*.*")],
            1,
            False,
        )
        self.calplate_status_var = self.calplate_loader.get_status_var()
        self.global_loader = FileLoader(
            self.selection_sect.content,
            "Global parameters:",
            [("Transformation Parameters", "*.json"), ("All Files", "*.*")],
            1,
            False,
        )
        self.global_status_var = self.global_loader.get_status_var()
        self.calplate_status_var.trace("w", lambda *args: self.create_local_pose_selector(self.mode_selector_var.get(), *args) if self.calplate_status_var.get() == "File Loaded" and self.global_status_var.get() == "File Loaded" else None)
        self.global_status_var.trace("w", lambda *args: self.create_local_pose_selector(self.mode_selector_var.get(), *args) if self.calplate_status_var.get() == "File Loaded" and self.global_status_var.get() == "File Loaded" else None)
        self.local_sect = Section(self.main_frame, "Local Pose", 2)
        self.calplate_plt = Frame(self.local_sect.content, 2, bd=2, relief="solid")
        self.pickers = Frame(self.local_sect.content, 2)
        self.pick_button = Button(
            self.pickers,
            "Pick Location",
            command=self._pick_location,
        )
        self.pick_monitors = Frame(self.pickers, 2)
        self.xpick = tk.StringVar()
        self.xentry_label = Label(self.pick_monitors, "x1 [mm]:", 2)
        self.xentry = Entry(self.pick_monitors, 2, textvariable=self.xpick, state="readonly")
        self.yentry_label = Label(self.pick_monitors, "x2 [mm]:", 2)
        self.ypick = tk.StringVar()
        self.yentry = Entry(self.pick_monitors, 2, textvariable=self.ypick, state="readonly")
        self.cursor = None
        self.marker_pt = None

    def _layout_widgets_default(self):
        self.calplate_loader.grid_forget()
        self.global_loader.grid_forget()
        self.local_sect.grid_forget()

        self.main_frame.grid_columnconfigure(0, weight=1)

        self.selection_sect.grid(
            row=0,
            column=0,
            padx=STYLES["pad"]["small"],
            pady=STYLES["pad"]["small"],
            sticky="nsew",
        )
        self.selection_sect.content.grid_columnconfigure(0, weight=1)
        self.selection_sect.content.grid_columnconfigure(1, weight=1)

        self.mode_selector_label.grid(
            row=0,
            column=0,
            padx=STYLES["pad"]["small"],
            pady=STYLES["pad"]["small"],
            sticky="w",
        )
        self.mode_selector.grid(
            row=0,
            column=1,
            padx=STYLES["pad"]["small"],
            pady=STYLES["pad"]["small"],
            sticky="ew",
        )

        self.submit_button.grid(
            row=1, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"]
        )

    def _layout_widgets_local_only(self):
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

        self.submit_button.grid(
            row=2, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"]
        )

    def _layout_widgets_local_global(self):
        self._layout_widgets_default()
        # self.load_transform_button.grid_forget()
        # self.main_frame.grid_rowconfigure(1, weight=1)

        # self.local_sect.grid(
        #     row=1,
        #     column=0,
        #     padx=STYLES["pad"]["small"],
        #     pady=STYLES["pad"]["small"],
        #     sticky="nsew",
        # )

    def _on_selection(self, *args):
        selected_option = self.mode_selector_var.get()
        if selected_option == CASES[0]:
            self._layout_widgets_default()
            self.scrollable_canvas.configure_frame()
        elif selected_option == CASES[1]:
            self._layout_widgets_local_only()
            self.scrollable_canvas.configure_frame()
        else:
            self._layout_widgets_local_global()
            self.scrollable_canvas.configure_frame()

    def _submit_file(self):
        pass

    def plot_calplate(self, case: str):
        """Plot the calibration plate image."""
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
        if case == CASES[1]:
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

        # extent = (img_coords_mm[0].min(), img_coords_mm[0].max(), img_coords_mm[1].min(), img_coords_mm[1].max())
        # self.cal_ax.imshow(img_vals, extent=extent)
        cmap = plt.get_cmap("gray")
        self.cal_ax.pcolormesh(img_coords_mm[0], img_coords_mm[1], img_vals, cmap=cmap, vmin=0, vmax=4000)
        self.cal_ax.set_xlabel(r"$x_1$ (mm)", labelpad=10)
        self.cal_ax.set_ylabel(r"$x_2$ (mm)", labelpad=10)
        # self.cal_ax.set_xlim(-0.65, 0.65)
        # self.cal_ax.set_ylim(0.65, -0.65)
        # self.cal_ax.set_aspect("equal")
        self.cal_canvas.draw()
        self.scrollable_canvas.configure_frame()

    def _pick_location(self):
        if self.cursor is None:
            self.cursor = Cursor(self.cal_ax, useblit=True, color="gray", linewidth=1)
        if self.marker_pt:
            self.marker_pt.remove()
        self.cal_canvas.draw_idle()
        pts = plt.ginput(n=1, timeout=0, show_clicks=True)

        self.marker_pt = self.cal_ax.plot(pts[0][0], pts[0][1], 'rx', markersize=10)[0]
        self.cal_canvas.draw_idle()

        self.cursor=None
        self.xpick.set(f"{pts[0][0]}")
        self.ypick.set(f"{pts[0][1]}")

    def create_local_pose_selector(self, case: str, *args):
        self.local_sect.grid(
            row=2,
            column=0,
            padx=STYLES["pad"]["small"],
            pady=STYLES["pad"]["small"],
            sticky="nsew",
        )
        self.calplate_plt.grid(
            row=0,
            rowspan=1,
            column=0,
            padx=STYLES["pad"]["small"],
            pady=STYLES["pad"]["small"],
            sticky="nsew"
        )
        self.local_sect.content.grid_columnconfigure(1, weight=1)
        self.local_sect.content.grid_rowconfigure(0, weight=1)
        self.submit_button.grid(row=3)

        self.plot_calplate(case)
        self.pickers.grid(
            row=0, column=1, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"]
        )
        self.pickers.grid_columnconfigure(0, weight=1)
        self.pick_button.grid(
            row=0, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"]
        )
        self.pick_monitors.grid(
            row=1, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"]
        )
        self.pick_monitors.grid_columnconfigure(0, weight=1)
        self.pick_monitors.grid_columnconfigure(1, weight=1)
        self.xentry_label.grid(
            row=0, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.yentry_label.grid(
            row=0, column=1, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.xentry.grid(
            row=1, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.yentry.grid(
            row=1, column=1, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )

    # pts = plotting.local_reference_selector(
    #     img_coords_mm[0], img_coords_mm[1], img_vals, pose_measurement
    # )
    # x_1_loc_ref_mm = pts[0][0]
    # x_2_loc_ref_mm = pts[0][1]

    # transform_params_updates = {
    #     "translation": {
    #         "x_1_loc_ref_mm": x_1_loc_ref_mm,
    #         "x_2_loc_ref_mm": x_2_loc_ref_mm,
    #     }
    # }
    # utility.update_nested_dict(
    #     piv_obj.transformation_parameters, transform_params_updates
    # )
    # utility.write_json(transform_params_file_path, piv_obj.transformation_parameters)



    #     bev = Beverli(self.bump_orientation, "cad")
    #     px, pz = bev.compute_perimeter(self.bump_orientation)
    #     self.bump_ax.plot(px, pz, color="blue")
    #     self.bump_ax.set_xlabel(r"$x_1$ (m)", labelpad=10)
    #     self.bump_ax.set_ylabel(r"$x_3$ (m)", labelpad=10)
    #     self.bump_ax.set_xlim(-0.65, 0.65)
    #     self.bump_ax.set_ylim(0.65, -0.65)
    #     self.bump_ax.set_aspect("equal")
    #     self.bump_canvas.draw()
