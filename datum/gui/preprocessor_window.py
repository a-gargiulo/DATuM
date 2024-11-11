"""Preprocessor Application"""

import tkinter as tk
from tkinter import messagebox

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ..core.beverli import Beverli
from ..utility.configure import STYLES, system
from .pose_window import PoseWindow
from .widgets import (Button, Checkbutton, Entry, FileLoader, Frame, Label,
                      ScrollableCanvas, Section)

# Constants
WINDOW_TITLE = "Preprocessor"
WINDOW_SIZE = (800, 600)


class PreprocessorWindow:
    def __init__(self, master: tk.Tk):
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
        self.bump_orientation = 0
        self.transform_sect = Section(self.geom_sect.content, "Pose & Transformation (Local PIV -> Global SWT)", 2)
        self.pose_button = Button(self.transform_sect.content, "Load/Calculate Tranformation Matrix", self.open_pose)
        self.pose_button.config(width=200 if system == "Darwin" else 20)
        self.pose_status_label = Label(self.transform_sect.content, "Nothing Loaded", 2)
        self.pose_status_label.config(fg="red")
        self.checkbox_interp = Checkbutton(self.transform_sect.content, 2)
        self.checkbox_interp.config(text="Interpolate data to regular grid", command=self._toggle_interp)
        self.checkbox_interp_var = self.checkbox_interp.get_var()
        self.interp_pts_label = Label(self.transform_sect.content, "Number of interp. grid points:", 2)
        self.interp_pts_label.config(state="disabled")
        self.interp_pts_entry = Entry(self.transform_sect.content, 2, state="disabled")
        self.data_sect = Section(self.main_frame, "Raw (Matlab) Data", 2)
        mat_type = [("Matlab Files", "*.mat"), ("All Files", "*.*")]
        self.vel_loader = FileLoader(self.data_sect.content, "Mean Velocity", mat_type, 2)
        self.stress_loader = FileLoader(self.data_sect.content, "Reynolds Stress", mat_type, 2)
        self.dissp_loader = FileLoader(self.data_sect.content, "Turbulence Dissipation", mat_type, 2)
        self.inst_vel_loader = FileLoader(self.data_sect.content, "Velocity Frame", mat_type, 2)
        self.cfd_sect = Section(self.main_frame, "Mean Velocity Gradient Tensor", 1)
        self.checkbox_gradient = Checkbutton(self.cfd_sect.content, 1)
        self.checkbox_gradient.config(text="Enable combutation", command=self._toggle_gradient, state="disabled")
        self.checkbox_gradient_var = self.checkbox_gradient.get_var()
        self.checkbox_gradient_opt = Checkbutton(self.cfd_sect.content, 1)
        self.checkbox_gradient_opt.config(text=r"dUdZ and dVdZ from CFD", command=self._toggle_cfd, state="disabled")
        self.checkbox_gradient_opt_var = self.checkbox_gradient_opt.get_var()
        self.slice_button = Button(self.cfd_sect.content, "Load CFD Slice", self.load_cfd_slice, state="disabled")
        self.slice_button.config(width=200 if system == "Darwin" else 10)
        self.slice_listbox = tk.Listbox(self.cfd_sect.content, width=20, height=1, state="disabled")
        self.slice_status_label = Label(self.cfd_sect.content, "Nothing Loaded", 1, state="disabled")
        self.slice_status_label.config(fg="red")
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
            row=0, column=1, columnspan=2, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
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
        self.vel_loader.grid(
            row=0, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.stress_loader.grid(
            row=1, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.dissp_loader.grid(
            row=2, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.inst_vel_loader.grid(
            row=3, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        # self.data_sect.content.grid(columnspan=4)
        # self.data_sect.content.grid_columnconfigure(0, weight=1)
        # self.data_sect.content.grid_columnconfigure(1, weight=1)
        # self.data_sect.content.grid_columnconfigure(2, weight=1)
        # self.data_sect.content.grid_columnconfigure(3, weight=1)
        self.cfd_sect.grid(
            row=3, column=0, columnspan=2, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.cfd_sect.content.grid(columnspan=3)
        self.cfd_sect.content.grid_columnconfigure(0, weight=1)
        self.cfd_sect.content.grid_columnconfigure(1, weight=1)
        self.cfd_sect.content.grid_columnconfigure(2, weight=1)
        self.checkbox_gradient.grid(row=0, column=0, sticky="nsew")
        self.checkbox_gradient_opt.grid(row=0, column=1, sticky="nsew")
        self.slice_button.grid(row=1, column=0, columnspan=1, padx=5, pady=5, sticky="nsew")
        self.slice_listbox.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.slice_status_label.grid(row=1, column=2, padx=5, sticky="nsew")
        self.process_button.grid(row=4, column=0, columnspan=2, pady=5, padx=10)

    def _validate_float(self, input_value):
        if input_value == "":
            self.bump_orientation = 0
            self.plot_bump()
            return True
        try:
            float(input_value)
            self.bump_orientation = float(input_value)
            self.plot_bump()
            return True
        except ValueError:
            self._on_invalid_input()
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
            self.slice_button.config(state="disabled")
            self.slice_listbox.config(state="disabled")
            self.slice_status_label.config(state="disabled")

    def _toggle_gradient(self):
        is_gradient_enabled = self.checkbox_gradient_var.get()
        self.checkbox_gradient_opt.config(state="normal" if is_gradient_enabled else "disabled")

        if is_gradient_enabled and self.checkbox_gradient_opt_var.get():
            state = "normal"
        else:
            state = "disabled"
            self.checkbox_gradient_opt_var.set(0)

        self.slice_button.config(state=state)
        self.slice_listbox.config(state=state)

    def _toggle_cfd(self):
        is_cfd_enabled = self.checkbox_gradient_opt_var.get()
        state = "normal" if is_cfd_enabled else "disabled"
        self.slice_button.config(state=state)
        self.slice_listbox.config(state=state)
        self.slice_status_label.config(state=state)

    def plot_bump(self):
        if hasattr(self, "bump_fig") and hasattr(self, "bump_canvas"):
            self.bump_ax.clear()
        else:
            self.bump_fig = plt.figure(figsize=(2.5, 2.4))
            self.bump_ax = self.bump_fig.add_axes([0.28, 0.3, 0.75, 0.65])
            self.bump_canvas = FigureCanvasTkAgg(self.bump_fig, master=self.bump_plt_frame)
            self.bump_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        bev = Beverli(self.bump_orientation, "cad")
        px, pz = bev.compute_perimeter(self.bump_orientation)
        self.bump_ax.plot(px, pz, color="blue")
        self.bump_ax.set_xlabel(r"$x_1$ (m)", labelpad=10)
        self.bump_ax.set_ylabel(r"$x_3$ (m)", labelpad=10)
        self.bump_ax.set_xlim(-0.65, 0.65)
        self.bump_ax.set_ylim(0.65, -0.65)
        self.bump_ax.set_aspect("equal")
        self.bump_canvas.draw()

    def load_cfd_slice(self):
        pass

    def open_pose(self):
        PoseWindow(self.root)

    def preprocess_data(self):
        pass
