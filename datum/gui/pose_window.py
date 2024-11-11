"""Pose Calculator"""

import tkinter as tk

from ..utility.configure import STYLES, system
from .widgets import Button, ScrollableCanvas, Section

# Constants
WINDOW_TITLE = "Pose Calculator"
WINDOW_SIZE = (600, 600)


class PoseWindow:
    def __init__(self, master: tk.Tk):
        self.root = tk.Toplevel(master)
        self._configure_root()
        # self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self._create_widgets()
        self._layout_widgets()
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
        self.local_pose_sect = Section(self.main_frame, "Local Pose", 1)

    def _layout_widgets(self):
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.local_pose_sect.grid(
            row=0, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
