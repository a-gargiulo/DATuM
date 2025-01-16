"""Create the preprocessor application window."""

import sys
import tkinter as tk
from tkinter import messagebox
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ..core import preprocessing
from ..core.beverli import Beverli
from ..core.piv import Piv
from ..utility import apputils
from ..utility.configure import STYLES, system
from .pose_window import PoseWindow
from .widgets import Button, Checkbutton, Entry, FileLoader, Frame, Label, ScrollableCanvas, Section

# Constants
WINDOW_TITLE = "Preprocessor"
WINDOW_SIZE = (800, 600)
PAD_S = STYLES["pad"]["small"]


class PreprocessorWindow:
    """Generate the preprocessor GUI."""

    def __init__(self, master: tk.Tk):
        """Initialize the GUI and the resources."""
        self.geometry = Beverli(use_cad=True)
        self.piv = Piv()

        self.root = tk.Toplevel(master)
        self.configure_root()
        self.create_widgets()
        self.layout_widgets()
        self.plot_bump()
        self.scrollable_canvas.configure_frame()

    def configure_root(self):
        """Configure the main window."""
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}")
        self.root.resizable(False, False)
        self.root.configure(bg=STYLES["color"]["base"])
        self.root.option_add("*Font", (STYLES["font"], STYLES["font_size"]["regular"]))
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.vfcmd = self.root.register(self.validate_float)

    def create_widgets(self):
        """Create all widget entities."""
        self.scrollable_canvas = ScrollableCanvas(self.root, True, False)
        self.main_frame = self.scrollable_canvas.get_frame()
        self.geom_sect = Section(self.main_frame, "Geometry", 1)
        self.bump_plt_frame = Frame(self.geom_sect.content, 1, bd=2, relief="solid")
        self.general_sect = Section(self.geom_sect.content, "General", 2)
        self.bump_orientation_label = Label(self.general_sect.content, "Hill Orientation [deg]:", 2)
        self.bump_orientation_entry = Entry(self.general_sect.content, 2)
        self.bump_orientation_entry.config(validate="focusout", validatecommand=(self.vfcmd, "%P"))
        self.bump_orientation_entry.insert(0, "0")
        self.bump_orientation = 0.0
        self.bump_orientation_button = Button(self.general_sect.content, "Confirm", self.confirm_orientation)
        self.orientation_is_confirmed = False
        self.transform_sect = Section(self.geom_sect.content, "Pose & Transformation (Local PIV -> Global SWT)", 2)
        self.pose_button = Button(self.transform_sect.content, "Load/Calculate Tranformation Matrix", self.open_pose)
        self.pose_button.config(width=200 if system == "Darwin" else 20)
        self.pose_status_label = Label(self.transform_sect.content, "Nothing Loaded", 2)
        self.pose_status_label.config(fg="red")
        self.params_status_var = tk.BooleanVar(value=False)
        self.params_status_var.trace("w", lambda *args: self.toggle_params_status(*args))
        self.checkbox_interp = Checkbutton(
            self.transform_sect.content, 2, text="Interpolate data to regular grid", command=self.toggle_interp
        )
        self.checkbox_interp_var = self.checkbox_interp.get_var()
        self.interp_pts_label = Label(
            self.transform_sect.content, "Number of interp. grid points:", 2, state="disabled"
        )
        self.interp_pts_entry = Entry(self.transform_sect.content, 2, state="disabled")
        self.data_sect = Section(self.main_frame, "Raw (Matlab) Data", 2)
        mat_type = [("Matlab Files", "*.mat"), ("All Files", "*.*")]
        self.vel_loader = FileLoader(self.data_sect.content, "Mean Velocity", mat_type, 2)
        self.vel_loader.checkbox_var.set(1)
        self.vel_loader.checkbox.config(state="disabled")
        self.vel_loader.load_button.config(state="normal")
        self.vel_loader.listbox.config(state="normal")
        self.vel_loader.status_label.config(state="normal")
        self.checkbox_flip_u3 = Checkbutton(self.data_sect.content, 2, text="Flip U3 Velocity")
        self.checkbox_flip_u3_var = self.checkbox_flip_u3.get_var()
        self.stress_loader = FileLoader(self.data_sect.content, "Reynolds Stress", mat_type, 2)
        self.dissp_loader = FileLoader(self.data_sect.content, "Turbulence Dissipation", mat_type, 2)
        self.inst_vel_loader = FileLoader(self.data_sect.content, "Velocity Frame", mat_type, 2)
        self.cfd_sect = Section(self.main_frame, "Mean Velocity Gradient Tensor", 1)
        self.checkbox_gradient = Checkbutton(
            self.cfd_sect.content, 1, text="Enable combutation", command=self.toggle_gradient, state="disabled"
        )
        self.checkbox_gradient_var = self.checkbox_gradient.get_var()
        self.checkbox_gradient_opt = Checkbutton(
            self.cfd_sect.content, 1, text=r"dUdZ and dVdZ from CFD", state="disabled"
        )
        self.checkbox_gradient_opt_var = self.checkbox_gradient_opt.get_var()
        self.slice_loader = FileLoader(
            self.cfd_sect.content, "CFD Slice", [("Tecplot Slice", "*.dat"), ("All Files", "*.*")], 1, False
        )
        self.slice_loader.load_button.config(state="disabled")
        self.slice_loader.listbox.config(state="disabled")
        self.slice_loader.status_label.config(state="disabled")
        self.slice_zone_name_label = Label(self.cfd_sect.content, "Slice Zone Name:", 1, state="disabled")
        self.slice_zone_name = Entry(self.cfd_sect.content, 1, state="disabled")
        self.process_button = Button(self.main_frame, "Preprocess Data", self.preprocess_data)
        self.process_button.config(width=200 if system == "Darwin" else 20)

    def layout_widgets(self):
        """Layout all widgets on the window."""
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.geom_sect.grid(row=0, column=0, columnspan=2, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.geom_sect.content.grid(columnspan=2)
        self.geom_sect.content.grid_columnconfigure(0, weight=0)
        self.geom_sect.content.grid_columnconfigure(1, weight=1)
        self.bump_plt_frame.grid(row=0, column=0, rowspan=2, padx=(0, PAD_S), pady=PAD_S, sticky="nsew")
        self.general_sect.grid(row=0, column=1, padx=(PAD_S, 0), pady=PAD_S, sticky="nsew")
        self.general_sect.content.grid(columnspan=3)
        self.general_sect.content.grid_columnconfigure(0, weight=0)
        self.general_sect.content.grid_columnconfigure(1, weight=1)
        self.general_sect.content.grid_columnconfigure(2, weight=1)
        self.bump_orientation_label.grid(row=0, column=0, padx=PAD_S, pady=PAD_S, sticky="nsw")
        self.bump_orientation_entry.grid(row=0, column=1, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.bump_orientation_button.grid(row=0, column=2, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.transform_sect.grid(row=1, column=1, padx=(PAD_S, 0), pady=PAD_S, sticky="nsew")
        self.transform_sect.content.grid(columnspan=3)
        self.transform_sect.content.grid_columnconfigure(0, weight=1)
        self.transform_sect.content.grid_columnconfigure(1, weight=1)
        self.transform_sect.content.grid_columnconfigure(2, weight=1)
        self.pose_button.grid(row=0, column=0, columnspan=2, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.pose_status_label.grid(row=0, column=2, padx=PAD_S, sticky="nsew")
        self.checkbox_interp.grid(row=1, column=0, padx=PAD_S, sticky="nsew")
        self.interp_pts_label.grid(row=2, column=0, padx=PAD_S, pady=PAD_S, columnspan=2, sticky="nsw")
        self.interp_pts_entry.grid(row=2, column=2, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.data_sect.grid(row=2, column=0, columnspan=2, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.data_sect.content.grid_columnconfigure(0, weight=1)
        self.vel_loader.grid(row=0, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.checkbox_flip_u3.grid(row=0, column=1, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.stress_loader.grid(row=1, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.dissp_loader.grid(row=2, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.inst_vel_loader.grid(row=3, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.cfd_sect.grid(row=3, column=0, columnspan=2, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.cfd_sect.content.grid(columnspan=3)
        self.cfd_sect.content.grid_columnconfigure(0, weight=1)
        self.cfd_sect.content.grid_columnconfigure(1, weight=1)
        self.cfd_sect.content.grid_columnconfigure(2, weight=1)
        self.checkbox_gradient.grid(row=0, column=0, sticky="nsew")
        self.checkbox_gradient_opt.grid(row=0, column=1, sticky="nsew")
        self.slice_loader.grid(row=1, column=0, columnspan=3, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.slice_zone_name_label.grid(row=2, column=0, padx=PAD_S, pady=PAD_S, sticky="nsw")
        self.slice_zone_name.grid(row=2, column=1, padx=PAD_S, pady=PAD_S, sticky="nsw")
        self.process_button.grid(row=4, column=0, columnspan=2, pady=5, padx=10)

    def validate_float(self, input_value: str):
        """Check the bump orientation input."""
        if input_value == "":
            self.geometry.rotate(-self.bump_orientation)
            self.bump_orientation = 0.0
            self.geometry.orientation = self.bump_orientation
            self.orientation_is_confirmed = False
            self.plot_bump()
            return True
        try:
            float(input_value)
            self.geometry.rotate(float(input_value) - self.bump_orientation)
            self.bump_orientation = float(input_value)
            self.geometry.orientation = self.bump_orientation
            self.orientation_is_confirmed = False
            self.plot_bump()
            return True
        except ValueError:
            self.on_invalid_input()
            self.geometry.rotate(-self.bump_orientation)
            self.bump_orientation = 0.0
            self.geometry.orientation = self.bump_orientation
            self.orientation_is_confirmed = False
            return False

    def on_invalid_input(self):
        """Inform the user when the bump orientation input is wrong."""
        messagebox.showerror("Invalid Input", "Please enter a valid float.")

    def on_closing(self):
        """Free the resources after closing the window."""
        if hasattr(self, "bump_fig"):
            plt.close(self.bump_fig)

        if hasattr(self, "bump_canvas"):
            self.bump_canvas.get_tk_widget().grid_forget()
            self.bump_canvas.get_tk_widget().destroy()

        self.root.destroy()

    def toggle_interp(self):
        """Turn the data interpolation on/off."""
        is_interp_enabled = self.checkbox_interp_var.get()
        state = "normal" if is_interp_enabled else "disabled"

        self.interp_pts_entry.config(state=state)
        self.interp_pts_label.config(state=state)
        self.checkbox_gradient.config(state=state)

        if not is_interp_enabled:
            self.checkbox_gradient_var.set(0)
            self.checkbox_gradient_opt.config(state="disabled")
            self.checkbox_gradient_opt_var.set(0)
            self.slice_loader.load_button.config(state="disabled")
            self.slice_loader.listbox.config(state="disabled")
            self.slice_loader.status_label.config(state="disabled")
            self.slice_zone_name.config(state="disabled")

    def toggle_gradient(self):
        """Turn the gradient calculation on/off."""
        is_gradient_enabled = self.checkbox_gradient_var.get()
        state = "normal" if is_gradient_enabled else "disabled"
        if state == "disabled":
            self.checkbox_gradient_opt_var.set(0)

        self.checkbox_gradient_opt.config(state=state)
        self.slice_loader.load_button.config(state=state)
        self.slice_loader.listbox.config(state=state)
        self.slice_loader.status_label.config(state=state)
        self.slice_zone_name_label.config(state=state)
        self.slice_zone_name.config(state=state)

    def plot_bump(self):
        """Plot the bump contour."""
        if hasattr(self, "bump_fig") and hasattr(self, "bump_canvas"):
            self.bump_ax.clear()
        else:
            self.bump_fig = plt.figure(figsize=(2.5, 2.4))
            self.bump_ax = self.bump_fig.add_axes((0.28, 0.3, 0.75, 0.65))
            self.bump_canvas = FigureCanvasTkAgg(self.bump_fig, master=self.bump_plt_frame)
            self.bump_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        px, pz = self.geometry.calculate_perimeter()
        self.bump_ax.plot(px, pz, color="blue")
        self.bump_ax.set_xlabel(r"$x_1$ (m)", labelpad=10)
        self.bump_ax.set_ylabel(r"$x_3$ (m)", labelpad=10)
        self.bump_ax.set_xlim(-0.65, 0.65)
        self.bump_ax.set_ylim(0.65, -0.65)
        self.bump_ax.set_aspect("equal")
        self.bump_canvas.draw()

    def confirm_orientation(self):
        """Confirm the input bump orientation."""
        self.orientation_is_confirmed = True
        print(self.piv.pose.angle1)
        print(self.piv.pose.angle2)
        print(self.piv.pose.loc)
        print(self.piv.pose.glob)

    def toggle_params_status(self, *args):
        """Indicate the loading status of the pose parameters."""
        status = self.params_status_var.get()
        if status:
            self.pose_status_label.config(fg="green", text="Successfully Loaded")
        else:
            self.pose_status_label.config(fg="red", text="Nothing Loaded")

    def open_pose(self):
        """Open the pose app."""
        if self.orientation_is_confirmed:
            PoseWindow(self.root, self.piv, self.geometry, self.params_status_var)
        else:
            messagebox.showwarning("Warning", "You must first confirm the hill orientation.")

    def validate_inputs(self) -> bool:
        """Check the status of all required input fields."""
        if not self.params_status_var.get():
            messagebox.showwarning("Warning", "Not all data has been loaded! Please load all data.")
            return False

        loaders = [self.vel_loader, self.stress_loader, self.dissp_loader, self.inst_vel_loader, self.slice_loader]
        for loader in loaders:
            if loader.load_button.cget("state") == "normal":
                if loader.status_label_var.get() == "Nothing Loaded":
                    messagebox.showwarning("Warning", "Not all data has been loaded! Please load all data.")
                    return False

        return True

    def fetch_data_paths(self) -> Dict[str, str]:
        """Fetch all system paths for the PIV data."""
        data_paths = {}

        loaders = [
            (self.vel_loader, "mean_velocity"),
            (self.stress_loader, "reynolds_stress"),
            (self.dissp_loader, "turbulence_dissipation"),
            (self.inst_vel_loader, "instantaneous_velocity_frame"),
        ]

        for loader in loaders:
            if loader[0].load_button.cget("state") == "normal":
                data_paths[loader[1]] = loader[0].get_listbox_content()

        return data_paths

    def preprocess_data(self):
        """Preprocess the piv data."""
        if not self.validate_inputs():
            return
        data_paths = self.fetch_data_paths()

        state = {
            "interpolate_data": bool(self.checkbox_interp_var.get()),
            "num_interp_pts": int(self.interp_pts_entry.get()),
            "compute_gradients": bool(self.checkbox_gradient_var.get()),
            "slice_path": self.slice_loader.get_listbox_content()[0],
            "slice_name": self.slice_zone_name.get()
        }

        opts = {
            "flip_out_of_plane_component": bool(self.checkbox_flip_u3_var.get()),
            "use_dwdx_and_dwdy_from_cfd": bool(self.checkbox_gradient_opt_var.get()),
        }

        should_load = {
            "mean_velocity": bool(self.vel_loader.checkbox_var.get()),
            "reynolds_stress": bool(self.stress_loader.checkbox_var.get()),
            "turbulence_dissipation": bool(self.dissp_loader.checkbox_var.get()),
            "instantaneous_velocity_frame": bool(self.inst_vel_loader.checkbox_var.get()),
        }

        if preprocessing.preprocess_data(self.piv, state, opts, data_paths, should_load):
            messagebox.showinfo("Info", "PREPROCESSING SUCCESSFUL! Check output folder for preprocessed data.")
        else:
            sys.exit(-1)
