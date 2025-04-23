"""Define the data section of the preporcessor window."""

import tkinter as tk
from typing import TYPE_CHECKING

from .widgets import Checkbutton, FileLoader, Section

if TYPE_CHECKING:
    from .preprocessor_window import PreprocessorWindow


class DataSection:
    """Data section of the preprocessor window."""

    def __init__(self, parent: tk.Frame, controller: PreprocessorWindow):
        """Construct the data section.

        :param parent: Parent frame handle.
        :param controller: Main control window.
        """
        self.controller = controller

        self.section = Section(parent, "Raw (Matlab) Data", 2)
        self.content = self.section.content

        mat_type = [("Matlab Files", "*.mat"), ("All Files", "*.*")]
        self.velocity_loader = FileLoader(
            self.content,
            "Mean Velocity",
            mat_type,
            2,
        )
        self.velocity_loader.checkbox_var.set(1)
        self.velocity_loader.checkbox.config(state="disabled")
        self.velocity_loader.load_button.config(state="normal")
        self.velocity_loader.listbox.config(state="normal")
        self.velocity_loader.status_label.config(state="normal")
        self.checkbox_flip_u3 = Checkbutton(
            self.content, 2, text="Flip U3 Velocity"
        )
        self.checkbox_flip_u3_var = self.checkbox_flip_u3.var
        self.stress_loader = FileLoader(
            self.content,
            "Reynolds Stress",
            mat_type,
            2,
        )
        self.dissipation_loader = FileLoader(
            self.content,
            "Turbulence Dissipation",
            mat_type,
            2,
        )
        self.inst_velocity_loader = FileLoader(
            self.content,
            "Velocity Frame",
            mat_type,
            2,
        )
