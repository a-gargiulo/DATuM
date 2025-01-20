"""Main application window."""
import tkinter as tk
from typing import Tuple

from PIL import Image, ImageTk

from ..utility.configure import STYLES
from .preprocessor_window import PreprocessorWindow
from .profiler_window import ProfilerWindow
from .widgets import Button

# Constants
WINDOW_TITLE = "DaTUM"
WINDOW_SIZE = (600, 600)
PAD_M = STYLES["pad"]["medium"]
BANNER_IMG_PATH = "./datum/resources/images/banner.png"


class DatumWindow:
    """Class for the main application window."""

    def __init__(self, root: tk.Tk):
        """
        Class constructor.

        :param root: Main application handle.
        """
        self.root = root
        self.configure_root()
        self.create_widgets()
        self.layout_widgets()

    def configure_root(self):
        """Configure the main window."""
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}")
        self.root.resizable(False, False)
        self.root.configure(bg=STYLES["color"]["base"])

    def create_widgets(self):
        """Generate widgets for the main window."""
        self.banner_frame = tk.Frame(self.root, bg=STYLES["color"]["base"])
        self.banner_label, self.banner_image = self.create_banner(self.banner_frame, BANNER_IMG_PATH)
        self.preprocessor_button = Button(self.root, text="Preprocessor", command=self.open_preprocessor)
        self.profiler_button = Button(self.root, text="Profiler", command=self.open_profiler)

    def layout_widgets(self):
        """Layout widgets on the main window."""
        self.root.grid_columnconfigure(0, weight=1)
        self.banner_frame.grid(row=0, column=0, padx=PAD_M, pady=PAD_M, sticky="ew")
        self.banner_frame.grid_columnconfigure(0, weight=1)
        self.banner_label.grid(row=0, column=0, padx=0, pady=0)
        self.preprocessor_button.grid(row=1, column=0, padx=0, pady=PAD_M)
        self.profiler_button.grid(row=2, column=0, padx=0, pady=PAD_M)

    def create_banner(self, parent: tk.Frame, img_path: str) -> Tuple[tk.Label, ImageTk.PhotoImage]:
        """
        Generate programm banner.

        :param parent: Handle to the host window.
        :param img_path: Path to the banner image.

        :return: A tuple containing the banner label and banner image.
        :rtype: Tuple[tk.Label, ImageTk.PhotoImage]
        """
        banner_image = self._load_resized_image(img_path, int(WINDOW_SIZE[0] / 1.3))
        banner_label = tk.Label(parent, image=banner_image, bg=STYLES["color"]["base"])  # type: ignore
        return banner_label, banner_image

    def _load_resized_image(self, img_path: str, target_width: int) -> ImageTk.PhotoImage:
        image = Image.open(img_path)
        width, height = image.size
        aspect_ratio = width / height
        resized_image = image.resize((target_width, int(target_width / aspect_ratio)), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(resized_image)

    def open_preprocessor(self):
        """Open the preprocessor window."""
        PreprocessorWindow(self.root)

    def open_profiler(self):
        """Open the profiler window."""
        ProfilerWindow(self.root)
