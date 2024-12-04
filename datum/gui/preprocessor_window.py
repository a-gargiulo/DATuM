"""Create the preprocessor application window."""
import sys
import tkinter as tk
from tkinter import messagebox

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ..core.beverli import Beverli
from ..core.piv import Piv
from ..core import preprocessing
from ..utility.configure import STYLES, system
from ..utility import apputils
from .pose_window import PoseWindow
from .widgets import Button, Checkbutton, Entry, FileLoader, Frame, Label, ScrollableCanvas, Section
from ..core.load import load_raw_data
from ..core import transform

# Constants
WINDOW_TITLE = "Preprocessor"
WINDOW_SIZE = (800, 600)


class PreprocessorWindow:
    """Generate the GUI for the preprocessor window and link it to the core functions."""

    def __init__(self, master: tk.Tk):
        """Initialize GUI and resources."""
        self.geometry = Beverli(use_cad=True)
        self.piv = Piv()

        self.root = tk.Toplevel(master)
        self._configure_root()
        self._create_widgets()
        self._layout_widgets()
        self.plot_bump()
        self.scrollable_canvas.configure_frame()

    def _configure_root(self):
        """Configure main window settings."""
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}")
        self.root.resizable(False, False)
        self.root.configure(bg=STYLES["color"]["base"])
        self.root.option_add("*Font", (STYLES["font"], STYLES["font_size"]["regular"]))
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.vfcmd = self.root.register(self._validate_float)

    def _create_widgets(self):
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
        self.bump_orientation_button = Button(self.general_sect.content, "Confirm", self._confirm_orientation)
        self.bump_orientation = 0.0
        self.orientation_is_confirmed = False
        self.transform_sect = Section(self.geom_sect.content, "Pose & Transformation (Local PIV -> Global SWT)", 2)
        self.pose_button = Button(self.transform_sect.content, "Load/Calculate Tranformation Matrix", self.open_pose)
        self.params_status_var = tk.BooleanVar(value=False)
        self.pose_button.config(width=200 if system == "Darwin" else 20)
        self.pose_status_label = Label(self.transform_sect.content, "Nothing Loaded", 2)
        self.pose_status_label.config(fg="red")
        self.params_status_var.trace("w", lambda *args: self._toggle_params_status(*args))
        self.checkbox_interp = Checkbutton(self.transform_sect.content, 2)
        self.checkbox_interp.config(text="Interpolate data to regular grid", command=self._toggle_interp)
        self.checkbox_interp_var = self.checkbox_interp.get_var()
        self.interp_pts_label = Label(self.transform_sect.content, "Number of interp. grid points:", 2)
        self.interp_pts_label.config(state="disabled")
        self.interp_pts_entry = Entry(self.transform_sect.content, 2, state="disabled")
        self.data_sect = Section(self.main_frame, "Raw (Matlab) Data", 2)
        self.checkbox_flip_u3 = Checkbutton(self.data_sect.content, 2, text="Flip U3 Velocity", state="disabled")
        self.checkbox_flip_u3_var = self.checkbox_flip_u3.get_var()
        mat_type = [("Matlab Files", "*.mat"), ("All Files", "*.*")]
        self.vel_loader = FileLoader(self.data_sect.content, "Mean Velocity", mat_type, 2)
        self.vel_loader.checkbox_var.trace("w", lambda *args: self._toggle_flip_opt(*args))
        self.stress_loader = FileLoader(self.data_sect.content, "Reynolds Stress", mat_type, 2)
        self.dissp_loader = FileLoader(self.data_sect.content, "Turbulence Dissipation", mat_type, 2)
        self.inst_vel_loader = FileLoader(self.data_sect.content, "Velocity Frame", mat_type, 2)
        self.cfd_sect = Section(self.main_frame, "Mean Velocity Gradient Tensor", 1)
        self.checkbox_gradient = Checkbutton(self.cfd_sect.content, 1)
        self.checkbox_gradient.config(text="Enable combutation", command=self._toggle_gradient, state="disabled")
        self.checkbox_gradient_var = self.checkbox_gradient.get_var()
        self.checkbox_gradient_opt = Checkbutton(self.cfd_sect.content, 1)
        self.checkbox_gradient_opt.config(text=r"dUdZ and dVdZ from CFD", state="disabled")
        self.checkbox_gradient_opt_var = self.checkbox_gradient_opt.get_var()
        self.slice_loader = FileLoader(self.cfd_sect.content, "CFD Slice", [("Tecplot Slice", "*.dat"), ("All Files", "*.*")], 1, False)
        self.slice_loader.load_button.config(state="disabled")
        self.slice_loader.listbox.config(state="disabled")
        self.slice_loader.status_label.config(state="disabled")
        self.slice_zone_name_label = Label(self.cfd_sect.content, "Slice Zone Name:", 1, state="disabled")
        self.slice_zone_name = Entry(self.cfd_sect.content, 1, state="disabled")
        self.process_button = Button(self.main_frame, "Preprocess Data", self.preprocess_data)
        self.process_button.config(width=200 if system == "Darwin" else 20)

    def _layout_widgets(self):
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.geom_sect.grid(
            row=0, column=0, columnspan=2, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.geom_sect.content.grid(columnspan=2)
        self.geom_sect.content.grid_columnconfigure(0, weight=0)
        self.geom_sect.content.grid_columnconfigure(1, weight=1)
        self.bump_plt_frame.grid(
            row=0,
            column=0,
            rowspan=2,
            padx=(0, STYLES["pad"]["small"]),
            pady=STYLES["pad"]["small"],
            sticky="nsew",
        )
        self.general_sect.grid(
            row=0, column=1, padx=(STYLES["pad"]["small"], 0), pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.general_sect.content.grid(columnspan=3)
        self.general_sect.content.grid_columnconfigure(0, weight=0)
        self.general_sect.content.grid_columnconfigure(1, weight=1)
        self.general_sect.content.grid_columnconfigure(2, weight=1)
        self.bump_orientation_label.grid(
            row=0, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsw"
        )
        self.bump_orientation_entry.grid(
            row=0, column=1, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.bump_orientation_button.grid(
            row=0, column=2, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )

        self.transform_sect.grid(
            row=1, column=1, padx=(STYLES["pad"]["small"], 0), pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.transform_sect.content.grid(columnspan=3)
        self.transform_sect.content.grid_columnconfigure(0, weight=1)
        self.transform_sect.content.grid_columnconfigure(1, weight=1)
        self.transform_sect.content.grid_columnconfigure(2, weight=1)
        self.pose_button.grid(
            row=0, column=0, columnspan=2, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.pose_status_label.grid(row=0, column=2, padx=STYLES["pad"]["small"], sticky="nsew")
        self.checkbox_interp.grid(row=1, column=0, padx=STYLES["pad"]["small"], sticky="nsew")
        self.interp_pts_label.grid(
            row=2, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], columnspan=2, sticky="nsw"
        )
        self.interp_pts_entry.grid(
            row=2, column=2, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.data_sect.grid(
            row=2, column=0, columnspan=2, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.data_sect.content.grid_columnconfigure(0, weight=1)
        self.vel_loader.grid(row=0, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew")
        self.checkbox_flip_u3.grid(
            row=0, column=1, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.stress_loader.grid(
            row=1, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.dissp_loader.grid(row=2, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew")
        self.inst_vel_loader.grid(
            row=3, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.cfd_sect.grid(
            row=3, column=0, columnspan=2, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.cfd_sect.content.grid(columnspan=3)
        self.cfd_sect.content.grid_columnconfigure(0, weight=1)
        self.cfd_sect.content.grid_columnconfigure(1, weight=1)
        self.cfd_sect.content.grid_columnconfigure(2, weight=1)
        self.checkbox_gradient.grid(row=0, column=0, sticky="nsew")
        self.checkbox_gradient_opt.grid(row=0, column=1, sticky="nsew")
        self.slice_loader.grid(row=1, column=0, columnspan=3, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew")
        self.slice_zone_name_label.grid(row=2, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsw")
        self.slice_zone_name.grid(row=2, column=1, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsw")
        self.process_button.grid(row=4, column=0, columnspan=2, pady=5, padx=10)

    def _validate_float(self, input_value):
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
            self._on_invalid_input()
            self.geometry.rotate(-self.bump_orientation)
            self.geometry.orientation = None
            self.orientation_is_confirmed = False
            return False

    def _on_invalid_input(self):
        messagebox.showerror("Invalid Input", "Please enter a valid float.")

    def _on_closing(self):
        if hasattr(self, "bump_fig"):
            plt.close(self.bump_fig)

        if hasattr(self, "bump_canvas"):
            self.bump_canvas.get_tk_widget().grid_forget()
            self.bump_canvas.get_tk_widget().destroy()

        self.root.destroy()

    def _toggle_interp(self):
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

    def _toggle_gradient(self):
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

    def _confirm_orientation(self):
        self.orientation_is_confirmed = True
        print(self.piv.pose.angle1)
        print(self.piv.pose.angle2)
        print(self.piv.pose.loc)
        print(self.piv.pose.glob)

    def _toggle_params_status(self, *args):
        status = self.params_status_var.get()
        if status:
            self.pose_status_label.config(fg="green", text="Successfully Loaded")
        else:
            self.pose_status_label.config(fg="red", text="Nothing Loaded")

    def _toggle_flip_opt(self, *args):
        if self.vel_loader.checkbox_var.get():
            self.checkbox_flip_u3.config(state="normal")
        else:
            self.checkbox_flip_u3.config(state="disabled")

    def open_pose(self):
        """Open the pose app."""
        if self.orientation_is_confirmed:
            PoseWindow(self.root, self.piv, self.geometry, self.params_status_var)
        else:
            messagebox.showwarning("Warning", "You must first confirm the hill orientation.")

    def preprocess_data(self):
        """Preprocess the piv data."""
        data_path = {}
        if self.vel_loader.get_listbox_content()[0] != "":
            data_path["mean_velocity"] = self.vel_loader.get_listbox_content()[0]
        if self.stress_loader.get_listbox_content()[0] != "":
            data_path["reynolds_stress"] = self.stress_loader.get_listbox_content()[0]
        if self.dissp_loader.get_listbox_content()[0] != "":
            data_path["turbulence_dissipation"] = self.dissp_loader.get_listbox_content()[0]
        if self.inst_vel_loader.get_listbox_content()[0] != "":
            data_path["instantaneous_velocity_frame"] = self.inst_vel_loader.get_listbox_content()[0]

        opts = {
            "flip_out_of_plane_component": bool(self.checkbox_flip_u3_var.get()),
            "use_dwdx_and_dwdy_from_cfd": bool(self.checkbox_gradient_opt_var.get())
        }
        should_load = {
            "mean_velocity": bool(self.vel_loader.checkbox_var.get()),
            "reynolds_stress": bool(self.stress_loader.checkbox_var.get()),
            "turbulence_dissipation": bool(self.dissp_loader.checkbox_var.get()),
            "instantaneous_velocity_frame": bool(self.inst_vel_loader.checkbox_var.get())
        }

        load_raw_data(self.piv, data_path, should_load, opts)
        if self.piv.data is None:
            sys.exit(-1)
        if bool(self.checkbox_interp_var.get()):
            transform.rotate_data(self.piv)
            transform.translate_data(self.piv)
            transform.scale_coordinates(self.piv, scale_factor=1e-3)
            if self.piv.pose.angle2 != 0.0:
                self.piv.data["coordinates"]["Z"] = self.piv.data["coordinates"]["X"]
        else:
            transform.rotate_data(self.piv)
            transform.interpolate_data(self.piv, int(self.interp_pts_entry.get()))
            transform.translate_data(self.piv)
            transform.scale_coordinates(self.piv, scale_factor=1e-3)

        if bool(self.checkbox_gradient_var.get()) and self.piv.pose.angle2 == 0.0:
            preprocessing.compute_velocity_gradient(self.piv, self.slice_loader.get_listbox_content()[0], self.slice_zone_name.get(), opts)

        apputils.write_pickle("./outputs/preprocessed.pkl", self.piv.data)
