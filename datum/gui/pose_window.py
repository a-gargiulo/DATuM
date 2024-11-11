"""Pose Calculator"""

import tkinter as tk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ..utility.configure import STYLES, system
from .widgets import Button, FileLoader, Frame, Label, ScrollableCanvas, Section

# Constants
WINDOW_TITLE = "Pose Calculator"
WINDOW_SIZE = (600, 600)
CASES = [
    "Load transformation file.",
    "Calculate local and load global pose.",
    "Calculate local and global pose."
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

        self.selection_sect = Section(self.main_frame, "Select Calculation Mode", 1)
        self.option_selector_var = tk.StringVar()
        self.option_selector_var.set(CASES[0])
        self.option_selector_var.trace("w", self._on_selection)
        self.option_selector = tk.OptionMenu(self.selection_sect.content, self.option_selector_var, *CASES)

        self.load_transform_button = Button(
            self.main_frame, "Load Transformation File", command=self._load_transform_file
        )

        self.calplate_loader = FileLoader(self.selection_sect.content, "Calibration Image:", [("Calibration Image", "*.dat"), ("All Files", "*.*")], 1, False)
        self.local_sect = Section(self.main_frame, "Local Pose", 2)
        self.calplate_plt = Frame(self.local_sect.content, 2, bd=2, relief="solid")

    def _layout_widgets_default(self):
        self.local_sect.grid_forget()

        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=0)

        self.selection_sect.grid(
            row=0, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.selection_sect.content.grid_columnconfigure(0, weight=1)
        self.selection_sect.content.grid_rowconfigure(0, weight=1)

        self.option_selector.grid(
            row=0, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.load_transform_button.grid(
            row=1, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"]
        )

    def _layout_widgets_local_only(self):
        self._layout_widgets_default()
        self.load_transform_button.grid_forget()
        self.main_frame.grid_rowconfigure(1, weight=1)

        self.local_sect.grid(
            row=1, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )

    def _layout_widgets_local_global(self):
        self._layout_widgets_default()
        self.load_transform_button.grid_forget()
        self.main_frame.grid_rowconfigure(1, weight=1)

        self.local_sect.grid(
            row=1, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )

    def _on_selection(self, *args):
        selected_option = self.option_selector_var.get()
        if selected_option == CASES[0]:
            self._layout_widgets_default()
            self.scrollable_canvas.configure_frame()
        elif selected_option == CASES[1]:
            self._layout_widgets_local_only()
            self.scrollable_canvas.configure_frame()
        else:
            self._layout_widgets_local_global()
            self.scrollable_canvas.configure_frame()

    def _load_transform_file(self):
        pass


    # def plot_bump(self):
    #     if hasattr(self, "bump_fig") and hasattr(self, "bump_canvas"):
    #         self.bump_ax.clear()
    #     else:
    #         self.bump_fig = plt.figure(figsize=(2.5, 2.4))
    #         self.bump_ax = self.bump_fig.add_axes([0.28, 0.3, 0.75, 0.65])
    #         self.bump_canvas = FigureCanvasTkAgg(self.bump_fig, master=self.bump_plt_frame)
    #         self.bump_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    #     bev = Beverli(self.bump_orientation, "cad")
    #     px, pz = bev.compute_perimeter(self.bump_orientation)
    #     self.bump_ax.plot(px, pz, color="blue")
    #     self.bump_ax.set_xlabel(r"$x_1$ (m)", labelpad=10)
    #     self.bump_ax.set_ylabel(r"$x_3$ (m)", labelpad=10)
    #     self.bump_ax.set_xlim(-0.65, 0.65)
    #     self.bump_ax.set_ylim(0.65, -0.65)
    #     self.bump_ax.set_aspect("equal")
    #     self.bump_canvas.draw()

