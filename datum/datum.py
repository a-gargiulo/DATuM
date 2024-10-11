import tkinter as tk

from PIL import Image, ImageTk

from .utility.configure import system, colors, default_font
from .preprocessor import Preprocessor

if system == "Darwin":
    from tkmacosx import Button
elif system == "Windows":
    from tkinter import Button

class Datum:
    def __init__(self, root: tk.Tk):
        w_width = 600
        w_height = 600

        self.root = root
        self.root.title("DaTUM")
        self.root.geometry(f"{w_width}x{w_height}")
        self.root.resizable(False, False)
        self.root.configure(bg=colors["base"])

        # Banner
        self.banner_frame = tk.Frame(self.root, bg=colors["base"])
        self.banner_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.original_banner = Image.open("./resources/images/banner.png")
        self.original_banner_width, self.original_banner_height = self.original_banner.size

        aspect_ratio = self.original_banner_width / self.original_banner_height
        resized_banner_width = int(w_width / 1.3)
        resized_banner_height = int(resized_banner_width / aspect_ratio)
        resized_banner_image = self.original_banner.resize(
            (resized_banner_width, resized_banner_height), Image.LANCZOS
        )
        self.banner_photo = ImageTk.PhotoImage(resized_banner_image)

        self.banner = tk.Label(self.banner_frame, image=self.banner_photo, bg=colors["base"])
        self.banner.grid(row=0, column=0, padx=0, pady=0)

        if system == "Darwin":
            additional_button_params = {"borderless": 1}
        if system == "Windows":
            additional_button_params = {}

        self.preprocessor_button = Button(
            self.root,
            text="Preprocessor",
            command=self.open_preprocessor,
            bg=colors["accent"],
            fg=colors["s1_content"],
            font=(default_font[0], default_font[1], "bold"),
            **additional_button_params,
        )
        self.preprocessor_button.grid(row=1, column=0, padx=0, pady=10)

        self.root.grid_columnconfigure(0, weight=1)
        self.banner_frame.grid_columnconfigure(0, weight=1)

    def open_preprocessor(self):
        Preprocessor(self.root)
