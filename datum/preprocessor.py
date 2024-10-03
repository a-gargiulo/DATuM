import tkinter as tk
from tkinter import filedialog
from .config import system, default_font, Button, colors


class Preprocessor:
    def __init__(self, master: tk.Tk):
        self.root = tk.Toplevel(master)
        self.root.title("Preprocessor")
        self.root.geometry("800x600")
        self.root.resizable(False, False)
        self.root.configure(bg=colors["base"])
        self.root.option_add("*Font", default_font)
        self.root.grid_columnconfigure(0, weight=1)

        self.loader_frame = tk.Frame(self.root, bg=colors["base"])
        self.loader_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        self.loader_frame.grid_columnconfigure(0, weight=1)

        self.loader_option_frame = tk.Frame(self.loader_frame, borderwidth=1, relief="solid", bg="#413d46")
        self.loader_option_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.loader_option_frame.grid_columnconfigure(0, weight=1)

        self.loader_data_frame = tk.Frame(self.loader_frame, borderwidth=1, relief="solid", bg="#373737")
        self.loader_data_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.loader_data_frame.grid_columnconfigure(0, weight=1)

        loader_option_label_fmt = (default_font[0], 12, "bold")
        self.loader_option_label = tk.Label(self.loader_option_frame, text="Options", font=loader_option_label_fmt, borderwidth=1, relief="solid", bg="#2a262f", fg="white")
        self.loader_option_label.grid(row=0, column=0, padx=5, pady=5, ipady=5, sticky="ew")

        self.checkbox_var = tk.IntVar()
        self.checkbox = tk.Checkbutton(self.loader_option_frame, text="Turbulence Dissipation ENABLE", variable=self.checkbox_var, command=self.toggle_state, bg="#413d46", fg="white")
        self.checkbox.grid(row=1, column=0, sticky="w", padx=5)

        loader_data_label_fmt = (default_font[0], 12, "bold")
        self.loader_data_label = tk.Label(self.loader_data_frame, text="Raw (Matlab) Data", borderwidth=1, relief="solid", bg="#1e1e1e", fg="white", font=loader_data_label_fmt)
        self.loader_data_label.grid(row=0, column=0, columnspan=3, padx=5, pady=5, ipady=5, sticky="ew")

        self.create_file_loader(self.loader_data_frame, "Mean Velocity", 1, "normal")
        self.create_file_loader(self.loader_data_frame, "Reynolds Stress", 2, "normal")
        self.create_file_loader(self.loader_data_frame, "Turbulence Dissipation", 3, "disabled")
        self.create_file_loader(self.loader_data_frame, "Inst. Velocity Frame", 4, "disabled")

    def toggle_state(self):
        if self.checkbox_var.get():
            self.load_button_turbulence_dissipation.config(state="normal")
            self.status_label_turbulence_dissipation.config(state="normal")
            self.listbox_turbulence_dissipation.config(state="normal")
        else:
            self.load_button_turbulence_dissipation.config(state="disabled")
            self.status_label_turbulence_dissipation.config(state="disabled")
            self.listbox_turbulence_dissipation.config(state="disabled")




    def create_file_loader(self, frame, data_type, row, state):
        dt = data_type.lower().replace(' ', '_')

        button_font = (default_font[0], default_font[1], "bold")
        if system == "Darwin":
            additional_button_params = {"width": 200, "borderless": 1}
        elif system == "Windows":
            additional_button_params = {"width": 20}

        setattr(self, f"load_button_{dt}", Button(frame, text=f"{data_type}", command=lambda: self.load_files(data_type, frame), state=state, font=button_font, bg=colors["accent"], fg=colors["f1_content"], **additional_button_params))

        load_button = getattr(self, f"load_button_{dt}")
        load_button.grid(row=row, column=0, padx=5, pady=5, sticky="nsew")
        # load_button.grid_columnconfigure(0, weight=1)

        setattr(self, f"listbox_{dt}", tk.Listbox(frame, state=state, width=20, height=1))
        listbox = getattr(self, f"listbox_{dt}")
        listbox.grid(row=row, column=1, padx=5, pady=5, sticky="nsew")
        # listbox.grid_columnconfigure(1, weight=1)

        setattr(self, f"status_label_{dt}", tk.Label(frame, state=state, text="Nothing Loaded", width=13, bg="#373737", fg="white"))
        status_label = getattr(self, f"status_label_{dt}")
        status_label.grid(row=row, column=2, padx=5, pady=5, sticky="nsew")
        # status_label.grid_columnconfigure(2, weight=1)

    def load_files(self, data_type, section_frame):
        # Open file dialog and get the selected file paths
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Data Files", "*.mat"), ("All Files", "*.*")]
        )

        if file_paths:
            # Get the corresponding listbox and status label for this data type
            listbox = getattr(self, f"listbox_{data_type.lower().replace(' ', '_')}")
            status_label = getattr(self, f"status_label_{data_type.lower().replace(' ', '_')}")

            # Clear previous items in the listbox
            listbox.delete(0, tk.END)

            # Add new file paths to the listbox
            for file_path in file_paths:
                listbox.insert(tk.END, file_path)

            status_label.config(text="File Loaded")
        else:
            status_label.config(text="Nothing Loaded")
