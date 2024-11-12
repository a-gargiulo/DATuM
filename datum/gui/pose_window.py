"""Pose Calculator."""

import tkinter as tk
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ..utility.configure import STYLES
from ..utility import tputils
from .widgets import Button, FileLoader, Frame, Label, ScrollableCanvas, Section

# Constants
WINDOW_TITLE = "Pose Calculator"
WINDOW_SIZE = (600, 600)
CASES = [
    "Load transformation file.",
    "Calculate local and load global pose.",
    "Calculate local and global pose.",
]


class PoseWindow:
    """The pose calculator applet."""

    def __init__(self, master: tk.Tk):
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

    def _configure_root(self):
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}")
        self.root.resizable(False, False)
        self.root.configure(bg=STYLES["color"]["base"])
        self.root.option_add("*Font", (STYLES["font"], STYLES["font_size"]["regular"]))

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
        self.global_loader = FileLoader(
            self.selection_sect.content,
            "Global parameters:",
            [("Transformation Parameters", "*.json"), ("All Files", "*.*")],
            1,
            False,
        )
        self.local_sect = Section(self.main_frame, "Local Pose", 2)
        self.calplate_plt = Frame(self.local_sect.content, 2, bd=2, relief="solid")

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

        self.local_sect.grid(
            row=2,
            column=0,
            padx=STYLES["pad"]["small"],
            pady=STYLES["pad"]["small"],
            sticky="nsew",
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

    def plot_calplate(self):
        """Plot the calibration plate image."""
        if hasattr(self, "cal_fig") and hasattr(self, "cal_canvas"):
            self.cal_ax.clear()
        else:
            self.cal_fig = plt.figure(figsize=(4, 4))
            self.cal_ax = self.cal_fig.add_axes((0.28, 0.3, 0.75, 0.65))
            self.cal_canvas = FigureCanvasTkAgg(self.cal_fig, master=self.calplate_plt)
            self.cal_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # To-Do: a fixed value for skiprows is too specific. Generalize later.
        cal_img_path = self.calplate_loader.get_listbox_content()

        cal_img = np.loadtxt(cal_img_path, skiprows=4)
        dims = tputils.get_ijk(cal_img_path)

        img_coords_mm = np.array([np.reshape(cal_img[:, i], (dims[1], dims[0])) for i in range(2)])
        img_vals = np.reshape(cal_img[:, 2], (dims[1], dims[0]))


    # rotation_angle_deg = piv_obj.transformation_parameters["rotation"]["angle_deg"]
    # rotation_matrix = get_rotation_matrix(rotation_angle_deg, (0, 0, 1))
    # img_coords_mm = rotate_vector_quantity(img_coords_mm, rotation_matrix)

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
