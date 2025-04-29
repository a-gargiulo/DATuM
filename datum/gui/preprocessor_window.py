"""Preprocessor application window."""

import tkinter as tk
from tkinter import messagebox
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from datum.utility.logging import logger
from ..core import preprocessing
from ..core.beverli import Beverli
from ..core.my_types import PPInputs
from ..core.piv import Piv
from ..utility.configure import STYLES, system
from .cfd_section import CfdSection
from .data_section import DataSection
from .geometry_section import GeometrySection
from .widgets import Button, ScrollableCanvas

# CONSTANTS
WINDOW_TITLE = "Preprocessor"
WINDOW_SIZE = (800, 600)
PAD_S = STYLES["pad"]["small"]
PAD_M = STYLES["pad"]["medium"]
MSG_SUCCESS = "PREPROCESSING SUCCESSFUL! Data saved in the output folder."


class PreprocessorWindow:
    """Preprocessor window."""

    def __init__(self, master: tk.Tk):
        """
        Construct the preprocessor window.

        :param master: Parent window handle.
        """
        # Shared resources
        self.hill = Beverli(use_cad=True)
        self.piv = Piv()

        self.hill_orientation = 0.0
        self.hill_orientation_confirmed = False

        # GUI
        self.root = tk.Toplevel(master)
        self.configure_root()
        self.create_widgets()
        self.layout_widgets()
        self.plot_hill()
        self.scrollable_canvas.configure_frame()

    def configure_root(self):
        """Configure the window."""
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}")
        self.root.resizable(False, False)
        self.root.configure(bg=STYLES["color"]["base"])
        self.root.option_add(
            "*Font", (STYLES["font"], STYLES["font_size"]["regular"])
        )
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.vfcmd = self.root.register(self.validate_float)

    def create_widgets(self):
        """Create widget entities."""
        self.scrollable_canvas = ScrollableCanvas(self.root, True, False)
        self.main_frame = self.scrollable_canvas.frame

        self.geometry_section = GeometrySection(self.main_frame, self)
        self.data_section = DataSection(self.main_frame, self)
        self.cfd_section = CfdSection(self.main_frame, self)

        self.process_button = Button(
            self.main_frame,
            "Preprocess Data",
            command=self.preprocess_data,
        )
        self.process_button.config(width=200 if system == "Darwin" else 20)

    def layout_widgets(self):
        """Layout all widgets on the window."""
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)

        self.geometry_section.layout()
        self.data_section.layout()
        self.cfd_section.layout()

        self.process_button.grid(
            row=4, column=0, columnspan=2, pady=PAD_S, padx=PAD_M
        )

    def validate_float(self, input_value: str) -> bool:
        """
        Check the user input hill orientation.

        :param input_value: User input value.
        """
        if input_value == "":
            self.hill.rotate(-self.hill_orientation)
            self.hill_orientation = 0.0
            self.hill.orientation = self.hill_orientation
            self.orientation_is_confirmed = False
            self.plot_hill()
            return True
        try:
            float(input_value)
            self.hill.rotate(float(input_value) - self.hill_orientation)
            self.hill_orientation = float(input_value)
            self.hill.orientation = self.hill_orientation
            self.orientation_is_confirmed = False
            self.plot_hill()
            return True
        except ValueError:
            self.on_invalid_input()
            self.hill.rotate(-self.hill_orientation)
            self.hill_orientation = 0.0
            self.hill.orientation = self.hill_orientation
            self.orientation_is_confirmed = False
            return False

    def on_invalid_input(self):
        """Inform the user when the hill orientation input is wrong."""
        messagebox.showerror("Invalid Input", "Please enter a valid float.")

    def on_closing(self):
        """Free the resources after closing the window."""
        if hasattr(self, "hill_fig"):
            plt.close(self.hill_fig)

        if hasattr(self, "hill_canvas"):
            self.hill_canvas.get_tk_widget().grid_forget()
            self.hill_canvas.get_tk_widget().destroy()

        self.root.destroy()

    def plot_hill(self):
        """Plot the hill contour."""
        if hasattr(self, "hill_fig") and hasattr(self, "hill_canvas"):
            self.hill_ax.clear()
        else:
            self.hill_fig = plt.figure(figsize=(2.5, 2.4))
            self.hill_ax = self.hill_fig.add_axes((0.28, 0.3, 0.75, 0.65))
            self.hill_canvas = FigureCanvasTkAgg(
                self.hill_fig, master=self.geometry_section.hill_plot
            )
            self.hill_canvas.get_tk_widget().grid(
                row=0, column=0, sticky="nsew"
            )

        px, pz = self.hill.calculate_perimeter()
        self.hill_ax.plot(px, pz, color="blue")
        self.hill_ax.set_xlabel(r"$x_1$ (m)", labelpad=10)
        self.hill_ax.set_ylabel(r"$x_3$ (m)", labelpad=10)
        self.hill_ax.set_xlim(-0.65, 0.65)
        self.hill_ax.set_ylim(0.65, -0.65)
        self.hill_ax.set_aspect("equal")
        self.hill_canvas.draw()

    def ready_to_preprocess(self) -> bool:
        """
        Check the status of all required input fields.

        :return: Boolean indicating whether all inputs are valid or not.
        """
        if not self.geometry_section.pose_status_var.get():
            messagebox.showwarning(
                "Warning",
                "Pose data has not been loaded! Please load the pose data.",
            )
            return False

        loaders = [
            self.data_section.velocity_loader,
            self.data_section.stress_loader,
            self.data_section.dissipation_loader,
            self.data_section.inst_velocity_loader,
            self.cfd_section.slice_loader,
        ]
        for loader in loaders:
            if loader.load_button.cget("state") == "normal":
                if loader.status_label_var.get() == "Nothing Loaded":
                    messagebox.showwarning(
                        "Warning",
                        (
                            r"Not all data has been loaded! "
                            r"Please load all raw data."
                        ),
                    )
                    return False

        return True

    def fetch_raw_data_paths(self) -> Dict[str, str]:
        """
        Fetch the paths for the raw PIV data to be loaded.

        :return: Raw data file paths.
        :rtype: Dict[str, str]
        """
        data_paths = {}

        loaders = [
            (self.data_section.velocity_loader, "mean_velocity"),
            (self.data_section.stress_loader, "reynolds_stress"),
            (self.data_section.dissipation_loader, "turbulence_dissipation"),
            (
                self.data_section.inst_velocity_loader,
                "velocity_snapshot",
            ),
        ]

        for loader in loaders:
            if loader[0].load_button.cget("state") == "normal":
                data_paths[loader[1]] = loader[0].get_listbox_content()

        return data_paths

    def collect_user_inputs(self) -> PPInputs:
        """Collect required user inputs to run the preprocessor."""
        ui: PPInputs = {
            "piv_data_paths": self.fetch_raw_data_paths(),
            "load_set": {
                "reynolds_stress": bool(
                    self.data_section.stress_loader.checkbox_var.get()
                ),
                "turbulence_dissipation": bool(
                    self.data_section.dissipation_loader.checkbox_var.get()
                ),
                "velocity_snapshot": bool(
                    self.data_section.inst_velocity_loader.checkbox_var.get()
                ),
            },
            "flip_u3": bool(self.data_section.checkbox_flip_u3_var.get()),
            "interpolate_data": bool(
                self.geometry_section.checkbox_interpolation_var.get()
            ),
            "num_interpolation_pts": int(
                self.geometry_section.interpolation_pts_entry.get()
            ),
            "compute_gradients": bool(
                self.cfd_section.checkbox_gradient_var.get()
            ),
            "use_cfd_dwdx_and_dwdy": bool(
                self.cfd_section.checkbox_gradient_opt_var.get()
            ),
            "slice_path": self.cfd_section.slice_loader.get_listbox_content(),
            "slice_name": self.cfd_section.slice_zone_name.get(),
        }

        return ui

    def preprocess_data(self):
        """Preprocess the PIV data."""
        if not self.ready_to_preprocess():
            return

        ui = self.collect_user_inputs()

        try:
            preprocessing.preprocess_all(self.piv, ui)
        except (ValueError, RuntimeError) as e:
            logger.error(f"Preprocessing failed: {e}")
            messagebox.showwarning(
                "Warning",
                "Preprocessing failed! Check log file for details, "
                "correct errors, and try again."
            )
            return

        messagebox.showinfo("Info", MSG_SUCCESS)
        return
