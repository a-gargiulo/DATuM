"""Define the CFD section of the preprocessor window."""

import tkinter as tk
from typing import TYPE_CHECKING

from ..utility.configure import STYLES
from .widgets import Checkbutton, Entry, FileLoader, Label, Section

if TYPE_CHECKING:
    from .preprocessor_window import PreprocessorWindow

PAD_S = STYLES["pad"]["small"]


class CfdSection:
    """CFD section of the preprocessor window."""

    def __init__(self, parent: tk.Frame, controller: "PreprocessorWindow"):
        """Construct the CFD section.

        :param parent: Parent frame handle
        :param controller: Main control window.
        """
        self.controller = controller

        self.section = Section(parent, "Mean Velocity Gradient Tensor", 1)
        self.content = self.section.content

        self.checkbox_gradient = Checkbutton(
            self.content,
            1,
            text="Enable computation",
            command=self.toggle_gradient,
            state="disabled",
        )
        self.checkbox_gradient_var = self.checkbox_gradient.var
        self.checkbox_gradient_opt = Checkbutton(
            self.content,
            1,
            text=r"dUdZ and dVdZ from CFD",
            state="disabled",
        )
        self.checkbox_gradient_opt_var = self.checkbox_gradient_opt.var
        self.slice_loader = FileLoader(
            self.content,
            "CFD Slice",
            [("Tecplot Slice", "*.dat"), ("All Files", "*.*")],
            1,
            False,
        )
        self.slice_loader.load_button.config(state="disabled")
        self.slice_loader.listbox.config(state="disabled")
        self.slice_loader.status_label.config(state="disabled")
        self.slice_zone_name_label = Label(
            self.content,
            "Slice Zone Name:",
            1,
            state="disabled",
        )
        self.slice_zone_name = Entry(self.content, 1, state="disabled")

    def layout(self):
        """Layout all widget entities."""
        self.section.grid(
            row=3,
            column=0,
            columnspan=2,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.content.grid(columnspan=3)
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_columnconfigure(1, weight=1)
        self.content.grid_columnconfigure(2, weight=1)
        self.checkbox_gradient.grid(row=0, column=0, sticky="nsew")
        self.checkbox_gradient_opt.grid(row=0, column=1, sticky="nsew")
        self.slice_loader.grid(
            row=1,
            column=0,
            columnspan=3,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.slice_zone_name_label.grid(
            row=2, column=0, padx=PAD_S, pady=PAD_S, sticky="nsw"
        )
        self.slice_zone_name.grid(
            row=2, column=1, padx=PAD_S, pady=PAD_S, sticky="nsw"
        )

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
