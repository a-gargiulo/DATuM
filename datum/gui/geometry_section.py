"""Define the geometry section of the preprocessor window."""

import tkinter as tk
from tkinter import messagebox
from typing import TYPE_CHECKING

from ..utility.configure import STYLES, system
from .pose_window import PoseWindow
from .widgets import Button, Checkbutton, Entry, Frame, Label, Section

PAD_S = STYLES["pad"]["small"]

if TYPE_CHECKING:
    from .preprocessor_window import PreprocessorWindow


class GeometrySection:
    """Geometry section of the preprocessor window."""

    def __init__(self, parent: tk.Frame, controller: "PreprocessorWindow"):
        """Construct the geometry section.

        :param parent: Parent frame handle.
        :param controller: Main control window.
        """
        self.controller = controller

        self.section = Section(parent, "Geometry", 1)
        self.content = self.section.content

        self.hill_plot = Frame(self.content, 1, bd=2, relief="solid")
        self.general_section = Section(self.content, "General", 2)
        self.general = self.general_section.content
        self.hill_orientation_lbl = Label(
            self.general, "Hill Orientation [deg]:", 2
        )
        self.hill_orientation_entry = Entry(self.general, 2)
        self.hill_orientation_entry.config(
            validate="focusout", validatecommand=(self.controller.vfcmd, "%P")
        )
        self.hill_orientation_entry.insert(0, "0")
        self.hill_orientation_button = Button(
            self.general,
            "Confirm",
            command=self.confirm_hill_orientation,
        )
        self.hill_orientation_entry.bind(
            "<Return>", self.on_hill_orientation_return
        )
        self.transformation_section = Section(
            self.content, "Pose & Transformation (Local PIV -> Global SWT)", 2
        )
        self.transformation = self.transformation_section.content
        self.pose_button = Button(
            self.transformation,
            "Load/Calculate Tranformation Matrix",
            command=self.open_pose_window,
        )
        self.pose_button.config(width=200 if system == "Darwin" else 20)
        self.pose_status_lbl = Label(self.transformation, "Nothing Loaded", 2)
        self.pose_status_lbl.config(fg="red")
        self.pose_status_var = tk.BooleanVar(value=False)
        self.pose_status_var.trace(
            "w", lambda *args: self.toggle_pose_status(*args)
        )
        self.checkbox_interpolation = Checkbutton(
            self.transformation,
            2,
            text="Interpolate data to regular grid",
            command=self.toggle_interpolation,
        )
        self.checkbox_interpolation_var = self.checkbox_interpolation.var
        self.interpolation_pts_lbl = Label(
            self.transformation,
            text="Number of interp. grid points:",
            category=2,
            state="disabled",
        )
        self.interpolation_pts_entry = Entry(
            self.transformation, 2, state="disabled"
        )

    def on_hill_orientation_return(self, event):
        """Binding function for hill orientation entry."""
        self.hill_orientation_entry.bind(
            "<Return>", self.hill_plot.focus_set()
        )
        self.confirm_hill_orientation()

    def layout(self):
        """Layout all widgets entities."""
        self.section.grid(
            row=0,
            column=0,
            columnspan=2,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.content.grid(columnspan=2)
        self.content.grid_columnconfigure(0, weight=0)
        self.content.grid_columnconfigure(1, weight=1)
        self.hill_plot.grid(
            row=0,
            column=0,
            rowspan=2,
            padx=(0, PAD_S),
            pady=PAD_S,
            sticky="nsew",
        )
        self.general_section.grid(
            row=0, column=1, padx=(PAD_S, 0), pady=PAD_S, sticky="nsew"
        )
        self.general.grid(columnspan=3)
        self.general.grid_columnconfigure(0, weight=0)
        self.general.grid_columnconfigure(1, weight=1)
        self.general.grid_columnconfigure(2, weight=1)
        self.hill_orientation_lbl.grid(
            row=0, column=0, padx=PAD_S, pady=PAD_S, sticky="nsw"
        )
        self.hill_orientation_entry.grid(
            row=0, column=1, padx=PAD_S, pady=PAD_S, sticky="nsew"
        )
        self.hill_orientation_button.grid(
            row=0, column=2, padx=PAD_S, pady=PAD_S, sticky="nsew"
        )
        self.transformation_section.grid(
            row=1, column=1, padx=(PAD_S, 0), pady=PAD_S, sticky="nsew"
        )
        self.transformation.grid(columnspan=3)
        self.transformation.grid_columnconfigure(0, weight=1)
        self.transformation.grid_columnconfigure(1, weight=1)
        self.transformation.grid_columnconfigure(2, weight=1)
        self.pose_button.grid(
            row=0,
            column=0,
            columnspan=2,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.pose_status_lbl.grid(row=0, column=2, padx=PAD_S, sticky="nsew")
        self.checkbox_interpolation.grid(
            row=1, column=0, padx=PAD_S, sticky="nsew"
        )
        self.interpolation_pts_lbl.grid(
            row=2, column=0, padx=PAD_S, pady=PAD_S, columnspan=2, sticky="nsw"
        )
        self.interpolation_pts_entry.grid(
            row=2, column=2, padx=PAD_S, pady=PAD_S, sticky="nsew"
        )

    def confirm_hill_orientation(self):
        """Confirm the hill orientation input by the user."""
        self.controller.hill_orientation_confirmed = True
        messagebox.showinfo(
            "SUCCESS!",
            "Hill orientation confirmed.",
            parent=self.controller.root
        )

    def toggle_pose_status(self, *args):
        """Indicate the completion status of the PIV plane pose calculation."""
        status = self.pose_status_var.get()
        if status:
            self.pose_status_lbl.config(fg="green", text="Successfully Loaded")
        else:
            self.pose_status_lbl.config(fg="red", text="Nothing Loaded")

    def toggle_interpolation(self):
        """Turn data interpolation on/off."""
        interpolation_enabled = self.checkbox_interpolation_var.get()
        state = "normal" if interpolation_enabled else "disabled"

        self.interpolation_pts_entry.config(state=state)
        self.interpolation_pts_lbl.config(state=state)
        self.controller.cfd_section.checkbox_gradient.config(state=state)

        if not interpolation_enabled:
            self.controller.cfd_section.checkbox_gradient_var.set(0)
            self.controller.cfd_section.checkbox_gradient_opt.config(
                state="disabled"
            )
            self.controller.cfd_section.checkbox_gradient_opt_var.set(0)
            self.controller.cfd_section.slice_loader.load_button.config(
                state="disabled"
            )
            self.controller.cfd_section.slice_loader.listbox.config(
                state="disabled"
            )
            self.controller.cfd_section.slice_loader.status_label.config(
                state="disabled"
            )
            self.controller.cfd_section.slice_zone_name.config(
                state="disabled"
            )

    def open_pose_window(self):
        """Open the pose calculator window."""
        if self.controller.hill_orientation_confirmed:
            PoseWindow(
                self.controller.root,
                self.controller.piv,
                self.controller.hill,
                self.pose_status_var,
            )
        else:
            messagebox.showwarning("Warning", "Confirm hill orientation.")
