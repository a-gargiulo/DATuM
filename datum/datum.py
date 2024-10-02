import tkinter as tk
from tkinter import font

from .config import colors, default_font
from .preprocessor import Preprocessor


class Datum:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DaTUM")
        self.root.geometry("600x600")
        self.root.resizable(False, False)
        self.root.configure(bg=colors["base"])

        # Banner
        self.banner_str = self.load_banner("./assets/banner.txt")
        self.banner_font = font.Font(family="Courier", size=12)
        self.banner = tk.Label(
            self.root,
            text=self.banner_str,
            justify="left",
            bg=colors["base"],
            fg="white",
        )
        self.banner.config(font=self.banner_font)
        self.banner.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.preprocessor_button = tk.Button(
            self.root,
            text="Preprocessor",
            command=self.open_preprocessor,
            bg=colors["accent"],
            fg=colors["f1_content"],
            font=(default_font[0], default_font[1], "bold"),
        )
        self.preprocessor_button.grid(row=1, column=0, padx=0, pady=10)

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self.root.bind("<Configure>", self.adjust_font_size)

    @staticmethod
    def load_banner(f_path):
        banner = ""
        with open(f_path, "r", encoding="utf-8") as f:
            for line in f:
                banner += line

        return banner

    def open_preprocessor(self):
        Preprocessor(self.root)

    def adjust_font_size(self, event):
        window_width = event.width
        font_size = window_width // 58
        self.banner_font.configure(size=font_size)
