"""Main DATuM window."""

import tkinter as tk
from typing import Tuple

from PIL import Image, ImageTk

from datum.gui.preprocessor_window import PreprocessorWindow
from datum.gui.profiler_window import ProfilerWindow
from datum.gui.widgets import Button
from datum.utility import logging
from datum.utility.configure import STYLES

# Constants
WINDOW_TITLE = "DATuM"
WINDOW_SIZE = (600, 600)
PAD_M = STYLES["pad"]["medium"]
BANNER_IMG_PATH = "./datum/resources/images/banner.png"


class DatumWindow:
    """DATuM Window."""

    def __init__(self, root: tk.Tk) -> None:
        """Construct the window.

        :param root: Main application handle.
        """
        self.root = root
        self.configure_root()
        self.create_widgets()
        self.layout_widgets()
        logging.logger.info("DATuM started up successfully.")


    def configure_root(self) -> None:
        """Configure the window."""
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}")
        self.root.resizable(False, False)
        self.root.configure(bg=STYLES["color"]["base"])
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self) -> None:
        """Create all widget entities."""
        self.banner_frame = tk.Frame(self.root, bg=STYLES["color"]["base"])
        self.banner_label, self.banner_image = self.create_banner(
            self.banner_frame, BANNER_IMG_PATH
        )
        self.preprocessor_button = Button(
            self.root, "Preprocessor", self.open_preprocessor
        )
        self.profiler_button = Button(
            self.root, "Profiler", self.open_profiler
        )

    def layout_widgets(self) -> None:
        """Layout widgets on the window."""
        self.root.grid_columnconfigure(0, weight=1)
        self.banner_frame.grid(
            row=0, column=0, padx=PAD_M, pady=PAD_M, sticky="ew"
        )
        self.banner_frame.grid_columnconfigure(0, weight=1)
        self.banner_label.grid(row=0, column=0, padx=0, pady=0)
        self.preprocessor_button.grid(row=1, column=0, padx=0, pady=PAD_M)
        self.profiler_button.grid(row=2, column=0, padx=0, pady=PAD_M)

    def on_closing(self) -> None:
        """Free resources after closing the window."""
        if hasattr(self, 'pp_window'):
            if self.pp_window.root.winfo_exists():
                self.pp_window.on_closing()
        if hasattr(self, 'pr_window'):
            if self.pr_window.root.winfo_exists():
                self.pr_window.on_closing()
        self.root.destroy()
        logging.logger.info("DATuM closed successfully.")

    def create_banner(
        self, parent: tk.Frame, img_path: str
    ) -> Tuple[tk.Label, ImageTk.PhotoImage]:
        """Generate title banner.

        :param parent: Handle to the host window.
        :param img_path: Path to the banner image.

        :return: Banner label and banner image.
        :rtype: Tuple[tkinter.Label, ImageTk.PhotoImage]
        """
        banner_image = self._load_resized_image(
            img_path, int(WINDOW_SIZE[0] / 1.3)
        )
        banner_label = tk.Label(
            parent, image=banner_image, bg=STYLES["color"]["base"]
        )
        return banner_label, banner_image

    def _load_resized_image(
        self, img_path: str, target_width: int
    ) -> ImageTk.PhotoImage:
        image = Image.open(img_path)
        width, height = image.size
        aspect_ratio = width / height
        resized_image = image.resize(
            (target_width, int(target_width / aspect_ratio)),
            Image.Resampling.LANCZOS,
        )
        return ImageTk.PhotoImage(resized_image)

    def open_preprocessor(self) -> None:
        """Open the preprocessor window."""
        self.pp_window = PreprocessorWindow(self.root)

    def open_profiler(self) -> None:
        """Open the profiler window."""
        self.pr_window = ProfilerWindow(self.root)
