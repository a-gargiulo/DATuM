import tkinter as tk
from tkinter import font, PhotoImage
from PIL import Image, ImageTk

from .config import Button, colors, default_font, system
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
        self.original_image = Image.open("./assets/banner.png")
        self.original_width, self.original_height = self.original_image.size


        self.photo = ImageTk.PhotoImage(self.original_image)




        self.banner_font = font.Font(family="Courier", size=12)
        self.banner = tk.Label(self.root, image=self.photo)
        # self.banner = tk.Label(
        #     self.root,
        #     text=self.banner_str,
        #     justify="left",
        #     bg=colors["base"],
        #     fg="white",
        # )
        # self.banner.config(font=self.banner_font)
        self.banner.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        if system == "Darwin":
            additional_button_params = {"borderless": 1}
        if system == "Windows":
            additional_button_params = {}

        self.preprocessor_button = Button(
            self.root,
            text="Preprocessor",
            command=self.open_preprocessor,
            bg=colors["accent"],
            fg=colors["f1_content"],
            font=(default_font[0], default_font[1], "bold"),
            **additional_button_params,
        )
        self.preprocessor_button.grid(row=1, column=0, padx=0, pady=10)

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # self.root.bind("<Configure>", self.adjust_font_size)
        self.root.bind("<Configure>", self.resize_image)

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



    def resize_image(self, event):
        # Get the new width and height of the window
        new_width = event.width
        new_height = event.height

        # Calculate the new dimensions to maintain aspect ratio
        aspect_ratio = self.original_width / self.original_height

        # Calculate new size based on the limiting dimension (width or height)
        if new_width / aspect_ratio <= new_height:
            # Width is the limiting factor
            resized_width = new_width
            resized_height = int(new_width / aspect_ratio)
        else:
            # Height is the limiting factor
            resized_height = new_height
            resized_width = int(new_height * aspect_ratio)

        # Resize the image
        resized_image = self.original_image.resize((resized_width, resized_height), Image.LANCZOS)

        # Update the PhotoImage object with the resized image
        new_image = ImageTk.PhotoImage(resized_image)

        # Update the label with the new image
        self.banner.config(image=new_image)
        self.banner.image = new_image
