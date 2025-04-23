"""Define the geometry section of the preprocessor window."""
import tkinter as tk
from tkinter import messagebox

from typing import List, TYPE_CHECKING, Union

from ..utility.configure import system
from .pose_window import PoseWindow
from .widgets import (
    Button,
    Checkbutton,
    Entry,
    FileLoader,
    Frame,
    Label,
    Section
)

if TYPE_CHECKING:
    from .preprocessor_window import PreprocessorWindow


Dependencies = List[Union[Checkbutton, tk.IntVar, FileLoader, Entry]]


class GeometrySection:
    """Geometry section of the preprocessor window."""

    def __init__(self, parent: tk.Frame, controller: PreprocessorWindow):
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
            self.general, "Confirm", command=self.confirm_hill_orientation
        )

        self.transformation_section = Section(
            self.content, "Pose & Transformation (Local PIV -> Global SWT)", 2
        )
        self.transformation = self.transformation_section.content
        self.pose_button = Button(
            self.transformation,
            "Load/Calculate Tranformation Matrix",
            command=self.open_pose_window
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
            command=self.toggle_interpolation(*dependencies),
        )
        self.checkbox_interpolation_var = self.checkbox_interpolation.var
        self.interpolation_pts_lbl = Label(
            self.transformation_section.content,
            text="Number of interp. grid points:",
            category=2,
            state="disabled",
        )
        self.interpolation_pts_entry = Entry(
            self.transformation, 2, state="disabled"
        )

    def confirm_hill_orientation(self):
        """Confirm the hill orientation input by the user."""
        self.controller.hill_orientation_confirmed = True

    def toggle_pose_status(self, *args):
        """Indicate the completion status of the PIV plane pose calculation."""
        status = self.pose_status_var.get()
        if status:
            self.pose_status_lbl.config(fg="green", text="Successfully Loaded")
        else:
            self.pose_status_lbl.config(fg="red", text="Nothing Loaded")

    def toggle_interpolation(self, *args):
        """Turn data interpolation on/off."""
        checkbox_gradient: Checkbutton = args[0]
        checkbox_gradient_var: tk.IntVar = args[1]
        checkbox_gradient_opt: Checkbutton = args[2]
        checkbox_gradient_opt_var: tk.IntVar = args[3]
        slice_loader: FileLoader = args[4]
        slice_zone_name: Entry = args[5]

        interpolation_enabled = self.checkbox_interpolation_var.get()
        state = "normal" if interpolation_enabled else "disabled"

        self.interpolation_pts_entry.config(state=state)
        self.interpolation_pts_lbl.config(state=state)
        checkbox_gradient.config(state=state)

        if not interpolation_enabled:
            checkbox_gradient_var.set(0)
            checkbox_gradient_opt.config(state="disabled")
            checkbox_gradient_opt_var.set(0)
            slice_loader.load_button.config(state="disabled")
            slice_loader.listbox.config(state="disabled")
            slice_loader.status_label.config(state="disabled")
            slice_zone_name.config(state="disabled")

    def open_pose_window(self):
        """Open the pose calculator window."""
        if self.hill_orientation_confirmed:
            PoseWindow(self.root, self.piv, self.hill, self.pose_status_var)
        else:
            messagebox.showwarning("Warning", "Confirm hill orientation.")
