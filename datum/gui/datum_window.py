"""The main application window."""
import tkinter as tk
from typing import Tuple

from PIL import Image, ImageTk

from ..utility.configure import STYLES
from .preprocessor_window import PreprocessorWindow
from .widgets import Button
from ..core.piv import Piv
from ..core.beverli import Beverli


# Constants
WINDOW_TITLE = "DaTUM"
WINDOW_SIZE = (600, 600)
BANNER_IMG_PATH = "./datum/resources/images/banner.png"


class DatumWindow:
    """Class for the main application window."""

    def __init__(self, root: tk.Tk):
        """
        Class constructor.

        :param root: The main application handle.
        """
        self.geometry = Beverli(use_cad=True)
        self.piv = Piv()

        self.root = root
        self._configure_root()
        self._create_widgets()
        self._layout_widgets()

    def _configure_root(self):
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}")
        self.root.resizable(False, False)
        self.root.configure(bg=STYLES["color"]["base"])

    def _create_widgets(self):
        self.banner_frame = tk.Frame(self.root, bg=STYLES["color"]["base"])
        self.banner_label, self.banner_image = self._create_banner(self.banner_frame, BANNER_IMG_PATH)
        self.preprocessor_button = Button(self.root, text="Preprocessor", command=lambda: self.open_preprocessor(self.geometry, self.piv))
        self.profiler_button = Button(self.root, text="Profiler", command=lambda: self.open_profiler(self.geometry, self.piv))

    def _layout_widgets(self):
        self.root.grid_columnconfigure(0, weight=1)
        self.banner_frame.grid(row=0, column=0, padx=STYLES["pad"]["medium"], pady=STYLES["pad"]["medium"], sticky="ew")
        self.banner_frame.grid_columnconfigure(0, weight=1)
        self.banner_label.grid(row=0, column=0, padx=0, pady=0)
        self.preprocessor_button.grid(row=1, column=0, padx=0, pady=STYLES["pad"]["medium"])
        self.profiler_button.grid(row=2, column=0, padx=0, pady=STYLES["pad"]["medium"])

    def _create_banner(self, parent: tk.Frame, img_path: str) -> Tuple[tk.Label, ImageTk.PhotoImage]:
        banner_image = self._load_resized_image(img_path, int(WINDOW_SIZE[0] / 1.3))
        banner_label = tk.Label(parent, image=banner_image, bg=STYLES["color"]["base"])
        return banner_label, banner_image

    def _load_resized_image(self, img_path: str, target_width: int) -> ImageTk.PhotoImage:
        image = Image.open(img_path)
        width, height = image.size
        aspect_ratio = width / height
        resized_image = image.resize((target_width, int(target_width / aspect_ratio)), Image.LANCZOS)
        return ImageTk.PhotoImage(resized_image)

    def open_preprocessor(self, geometry: Beverli, piv: Piv):
        """Open the preprocessor window."""
        PreprocessorWindow(self.root, geometry, piv)

    def open_profiler(self, geometry: Beverli, piv: Piv):
        pass
