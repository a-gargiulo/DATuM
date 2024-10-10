import tkinter as tk

from .config import colors


class Pose:
    def __init__(self, master: tk.Tk):
        self.root = tk.Toplevel(master)
        self.root.title("Pose")
        self.root.geometry("600x600")
        self.root.resizable(False, False)
        self.root.configure(bg=colors["base"])
        self.root.grid_columnconfigure(0, weight=1)
