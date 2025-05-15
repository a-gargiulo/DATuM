"""Create the profiler application window."""

import sys
import tkinter as tk
from tkinter import messagebox
from typing import Optional, cast

from datum.core import profiles  # Deactivate to run without errors
from datum.core.beverli import Beverli
from datum.core.my_types import PivData
from datum.core.piv import Piv
from datum.core.pose import Pose
from datum.gui.widgets import (
    Button,
    Checkbutton,
    Entry,
    FileLoader,
    Label,
    ScrollableCanvas,
    Section,
)
from datum.utility import apputils
from datum.utility.configure import STYLES
from datum.utility.logging import logger

# Constants
WINDOW_TITLE = "Profiler"
WINDOW_SIZE = (600, 600)
PAD_S = STYLES["pad"]["small"]


class ProfilerWindow:
    """Generate the GUI for the profiler window."""

    def __init__(self, master: tk.Tk):
        """Initialize GUI.

        :param master: Parent window handle.
        """
        self.root = tk.Toplevel(master)
        self.configure_root()
        self.create_widgets()
        self.layout_widgets()
        self.scrollable_canvas.configure_frame()
        logger.info("Profiler window opened successfully.")

    def configure_root(self):
        """Configure window settings."""
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}")
        self.root.resizable(False, False)
        self.root.configure(bg=STYLES["color"]["base"])
        self.root.option_add(
            "*Font", (STYLES["font"], STYLES["font_size"]["regular"])
        )
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        """Create all widget entities."""
        self.scrollable_canvas = ScrollableCanvas(self.root, True, False)
        self.main_frame = self.scrollable_canvas.frame

        self.general_section = Section(self.main_frame, "General", category=1)
        self.hill_orientation_label = Label(
            self.general_section.content, "Hill Orientation:", 1
        )
        self.hill_orientation = Entry(self.general_section.content, 1)
        self.data_loader = FileLoader(
            self.general_section.content,
            title="Piv Data (no interp)",
            filetypes=[("Pickle File", "*.pkl"), ("All Files", "*.*")],
            category=1,
            isCheckable=False,
        )
        self.data_loader_interp = FileLoader(
            self.general_section.content,
            title="Piv Data (interp)",
            filetypes=[("Pickle File", "*.pkl"), ("All Files", "*.*")],
            category=1,
            isCheckable=False,
        )
        self.pose_loader = FileLoader(
            self.general_section.content,
            title="Transformation Parameters",
            filetypes=[("Pose File", "*.json"), ("All Files", "*.*")],
            category=1,
            isCheckable=False,
        )
        # self.checkbox_diagonal = Checkbutton(
        #     self.general_section.content,
        #     category=1,
        #     text="Plane is Diagonal",
        # )
        self.properties_loader = FileLoader(
            self.general_section.content,
            title="Fluid and Flow Properties",
            filetypes=[("Properties File", "*.json"), ("All Files", "*.*")],
            category=1,
            isCheckable=False,
        )
        self.ref_conditions_loader = FileLoader(
            self.general_section.content,
            title="Reference Conditions",
            filetypes=[
                ("Reference Conditions File", "*.stat"),
                ("All Files", "*.*"),
            ],
            category=1,
            isCheckable=False,
        )

        self.profiler_section = Section(self.main_frame, "Profiles", 1)
        self.num_profile_label = Label(
            self.profiler_section.content, "Number of Profiles:", 1
        )
        self.num_profile = Entry(self.profiler_section.content, 1)
        self.num_profile_pts_label = Label(
            self.profiler_section.content, "Number of Profile Points:", 1
        )
        self.num_profile_pts = Entry(self.profiler_section.content, 1)
        self.profile_height_label = Label(
            self.profiler_section.content, "Profile Height:", 1
        )
        self.profile_height = Entry(self.profiler_section.content, 1)
        self.coordinates_selector_label = Label(
            self.profiler_section.content, "Coordinate System:", 1
        )
        self.coordinates_selector_var = tk.StringVar()
        self.coordinates_selector_var.set("Tunnel")
        self.coordinates_selector = tk.OptionMenu(
            self.profiler_section.content,
            self.coordinates_selector_var,
            "Tunnel",
            "Shear",
        )
        self.coordinates_selector_var.trace("w", self.on_coordinates_selection)
        self.reconstruction_checkbox = Checkbutton(
            self.profiler_section.content,
            category=1,
            text="Add Reconstruction Points",
            command=self.toggle_reconstruction,
        )
        self.num_reconstruction_pts_label = Label(
            self.profiler_section.content,
            text="Number of Reconstruction Points:",
            category=1,
        )
        self.num_reconstruction_pts = Entry(self.profiler_section.content, 1)
        self.checkbox_cfd = Checkbutton(
            self.profiler_section.content,
            category=1,
            text="Extract CFD Profiles (expensive)",
            command=self.toggle_cfd,
        )
        self.fluent_case_loader = FileLoader(
            self.profiler_section.content,
            title="Fluent Case",
            filetypes=[("Fluent Case", "*.cas"), ("All Files", ".*.")],
            category=1,
            isCheckable=False,
        )
        self.fluent_data_loader = FileLoader(
            self.profiler_section.content,
            title="Fluent Data",
            filetypes=[("Fluent Data", "*.dat"), ("All Files", ".*.")],
            category=1,
            isCheckable=False,
        )
        self.calculate_button = Button(
            self.main_frame, text="Submit", command=self.calculate
        )

        self.pressure_section = Section(self.profiler_section.content, "Pressure Data", 2)
        self.port_loader = FileLoader(
            self.pressure_section.content,
            title="Port Wall Pressure",
            filetypes=[
                ("Port Wall Pressure File", "*.stat"),
                ("All Files", "*.*"),
            ],
            category=2,
            isCheckable=False,
        )
        self.hill_loader = FileLoader(
            self.pressure_section.content,
            title="Hill Surface Pressure",
            filetypes=[
                ("Hill Surface Pressure File", "*.stat"),
                ("All Files", "*.*"),
            ],
            category=2,
            isCheckable=False,
        )
        self.info_loader = FileLoader(
            self.pressure_section.content,
            title="Pressure Data Info File",
            filetypes=[
                ("Pressure Data Info File", "*.stat"),
                ("All Files", "*.*"),
            ],
            category=2,
            isCheckable=False,
        )

    def layout_widgets(self):
        """Layout all widgets on the window."""
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.general_section.grid(
            row=0, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew"
        )
        self.general_section.content.grid(columnspan=3)
        self.general_section.content.grid_columnconfigure(0, weight=1)
        self.general_section.content.grid_columnconfigure(1, weight=1)
        self.general_section.content.grid_columnconfigure(2, weight=1)
        self.hill_orientation_label.grid(
            row=0, column=0, padx=(2 * PAD_S, 0), pady=PAD_S, sticky="nsw"
        )
        self.hill_orientation.grid(
            row=0, column=1, columnspan=2, padx=0, pady=PAD_S, sticky="nsew"
        )
        self.data_loader.grid(
            row=1,
            column=0,
            columnspan=3,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.data_loader_interp.grid(
            row=2,
            column=0,
            columnspan=3,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.pose_loader.grid(
            row=3,
            column=0,
            columnspan=3,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        # self.checkbox_diagonal.grid(
        #     row=4, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew"
        # )
        self.properties_loader.grid(
            row=4,
            column=0,
            columnspan=3,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.ref_conditions_loader.grid(
            row=5,
            column=0,
            columnspan=3,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.pressure_section.grid(
            row=9, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew", columnspan=3
        )
        self.pressure_section.content.grid(columnspan=3)
        self.pressure_section.content.grid_columnconfigure(0, weight=1)
        self.pressure_section.content.grid_columnconfigure(1, weight=1)
        self.pressure_section.content.grid_columnconfigure(2, weight=1)
        self.port_loader.grid(
            row=0,
            column=0,
            columnspan=3,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.hill_loader.grid(
            row=1,
            column=0,
            columnspan=3,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.info_loader.grid(
            row=2,
            column=0,
            columnspan=3,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.profiler_section.grid(
            row=2, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew"
        )
        self.profiler_section.content.grid(columnspan=3)
        self.profiler_section.content.grid_columnconfigure(0, weight=1)
        self.profiler_section.content.grid_columnconfigure(1, weight=1)
        self.profiler_section.content.grid_columnconfigure(2, weight=1)
        self.num_profile_label.grid(
            row=0, column=0, padx=PAD_S, pady=PAD_S, sticky="nsw"
        )
        self.num_profile.grid(
            row=0,
            column=1,
            columnspan=2,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.num_profile_pts_label.grid(
            row=1, column=0, padx=PAD_S, pady=PAD_S, sticky="nsw"
        )
        self.num_profile_pts.grid(
            row=1,
            column=1,
            columnspan=2,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.profile_height_label.grid(
            row=2, column=0, padx=PAD_S, pady=PAD_S, sticky="nsw"
        )
        self.profile_height.grid(
            row=2,
            column=1,
            columnspan=2,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.coordinates_selector_label.grid(
            row=3, column=0, padx=PAD_S, pady=PAD_S, sticky="nsw"
        )
        self.coordinates_selector.grid(
            row=3,
            column=1,
            columnspan=2,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.reconstruction_checkbox.grid(
            row=4, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew"
        )
        self.num_reconstruction_pts_label.grid(
            row=5, column=0, padx=PAD_S, pady=PAD_S, sticky="nsw"
        )
        self.num_reconstruction_pts.grid(
            row=5,
            column=1,
            columnspan=2,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.checkbox_cfd.grid(
            row=6, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew"
        )
        self.fluent_case_loader.grid(
            row=7,
            column=0,
            columnspan=3,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.fluent_data_loader.grid(
            row=8,
            column=0,
            columnspan=3,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.calculate_button.grid(
            row=3, column=0, padx=PAD_S, pady=PAD_S, sticky="ns"
        )

    def on_closing(self):
        """Free resources and clean up when closing the window."""
        logger.info("Profiler window closed successfully.")
        self.root.destroy()

    def toggle_reconstruction(self):
        """Activate/deactivate the profile reconstruction option."""
        pass

    def on_coordinates_selection(self, *args):
        """Perform an action when the coordinate system is selected."""
        pass

    def toggle_cfd(self):
        """Activate/deactivate the option to extract equivalent profiles from CFD data."""
        pass

    def load_pose(self, pose_path: str, is_diagonal: bool) -> Optional[Pose]:
        """
        Load the pose of the PIV plane from the specified file.

        :param pose_path: Filepath.
        :param is_diagonal: Boolean indicating whether the PIV plane is diagonal.

        :return: A pose object.
        :rtype: Pose
        """
        pose_data = apputils.read_json(pose_path)
        if pose_data is None:
            return None
        pose = Pose(
            angle1=(
                float(cast(dict, pose_data["rotation"])["angle_1_deg"])
                if is_diagonal
                else float(cast(dict, pose_data["rotation"])["angle_deg"])
            ),
            angle2=(
                float(cast(dict, pose_data["rotation"])["angle_2_deg"])
                if is_diagonal
                else 0.0
            ),
            loc=[
                float(cast(dict, pose_data["translation"])["x_1_loc_ref_mm"]),
                float(cast(dict, pose_data["translation"])["x_2_loc_ref_mm"]),
            ],
            glob=[
                float(cast(dict, pose_data["translation"])["x_1_glob_ref_m"]),
                float(cast(dict, pose_data["translation"])["x_2_glob_ref_m"]),
                float(cast(dict, pose_data["translation"])["x_3_glob_ref_m"]),
            ],
        )
        return pose

    def calculate(self):
        """Extract 1D profile data from the 2D plane data."""
        try:
            hill_orientation = float(self.hill_orientation.get())
            tp_path = self.pose_loader.get_listbox_content()
            pth_interp = self.data_loader_interp.get_listbox_content()
            pth_no_interp = self.data_loader.get_listbox_content()

            # TODO: add a check for the orientation and the shear mode

            trans_params = apputils.load_transformation_parameters(tp_path)
            pose = apputils.make_pose_from_trans_params(trans_params)
            data_interp = cast(PivData, apputils.load_pickle(pth_interp))
            data_no_interp = cast(PivData, apputils.load_pickle(pth_no_interp))

            self.geometry = Beverli(hill_orientation, use_cad=True)
            self.piv_intrp = Piv(data_interp, pose)
            self.piv_no_intrp = Piv(data_no_interp, pose)

            # profiles.extract_data(self.piv_no_intrp, self.piv_intrp, self.geometry, opts)
        except Exception as e:
            logger.error(
                f"An error occured during the profile extraction: {e}"
            )
            messagebox.showerror(
                "ERROR!",
                "An error occured during the profile extraction. "
                "Read the log and try again."
            )
            return
