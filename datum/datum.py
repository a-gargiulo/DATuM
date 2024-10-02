import tkinter as tk

from .preprocessor import Preprocessor


class Datum:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DaTUM")
        self.root.geometry("600x600")
        self.root.resizable(False, False)

        self.banner_str = self.load_banner("./assets/banner.txt")
        self.banner = tk.Label(
            root, text=self.banner_str, font=("Courier", 14), justify="left"
        )
        self.banner.grid(row=0, column=0, padx=10, pady=10)

        self.preprocessor_button = tk.Button(
            self.root, text="Preprocessor", command=self.open_preprocessor
        )
        self.preprocessor_button.grid(row=1, column=0, padx=0, pady=10)

    @staticmethod
    def load_banner(f_path):
        banner = ""
        with open(f_path, "r", encoding="utf-8") as f:
            for line in f:
                banner += line

        return banner

    def open_preprocessor(self):
        Preprocessor(self.root)
