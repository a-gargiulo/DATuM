import tkinter as tk
from tkinter import messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


from .config import system, default_font, Button, colors
from .beverli import Beverli

class Preprocessor:
    def __init__(self, master: tk.Tk):
        self.root = tk.Toplevel(master)
        self.root.title("Preprocessor")
        self.root.geometry("800x600")
        self.root.resizable(False, False)
        self.root.configure(bg=colors["base"])
        self.root.option_add("*Font", default_font)
        self.root.grid_columnconfigure(0, weight=1)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        vcmd = self.root.register(self.validate_float)

        # Geometry Frame
        # --------------
        geom_sec_pos = {"row": 0, "column": 0, "columnspan": 2, "padx": 15, "pady": 5, "sticky": "nsew"}
        geom_sec_title_pos = {"row": 0, "column": 0, "columnspan": 2, "padx": 5, "pady": 5, "ipady": 5, "sticky": "ew"}
        self.create_section("Geometry", 1, self.root, geom_sec_pos, geom_sec_title_pos)
        self.geometry_frame.grid_columnconfigure(0, weight=0)
        self.geometry_frame.grid_columnconfigure(1, weight=1)

        self.bump_plot_frame = tk.Frame(self.geometry_frame, bg=colors["f1_content"])
        self.bump_plot_frame.grid(row=1, column=0, columnspan=1, padx=10, pady=5, sticky="nsew")
        self.bump_plot_frame.grid_columnconfigure(0, weight=1)

        self.orientation_entry = tk.Entry(self.geometry_frame, validate="key", validatecommand=(vcmd, '%P'))
        self.orientation_entry.grid(row=1, column=1, columnspan=1)

        self.orientation = 0
        self.plot_graph()

        # Loader Frame
        # ------------
        self.loader_frame = tk.Frame(self.root, bg=colors["base"])
        self.loader_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        self.loader_frame.grid_columnconfigure(0, weight=1)

        opt_sec_pos = {"row": 0, "column": 0, "padx": 5, "pady": 5, "sticky": "nsew"}
        opt_title_pos = {"row": 0, "column": 0, "padx": 5, "pady": 5, "ipady": 5, "sticky": "ew"}
        self.create_section("Options", 2, self.loader_frame, opt_sec_pos, opt_title_pos)
        self.options_frame.grid_columnconfigure(0, weight=1)

        data_sec_pos = {"row": 0, "column": 1, "padx": 5, "pady": 5, "sticky": "nsew"}
        data_title_pos = {"row": 0, "column": 0, "columnspan": 3, "padx": 5, "pady": 5, "ipady": 5, "sticky": "ew"}
        self.create_section("Raw Matlab Data", 1, self.loader_frame, data_sec_pos, data_title_pos)
        self.raw_matlab_data_frame.grid_columnconfigure(0, weight=1)

        self.create_file_loader(self.raw_matlab_data_frame, "Mean Velocity", 1, "normal")
        self.create_file_loader(self.raw_matlab_data_frame, "Reynolds Stress", 2, "normal")
        self.create_file_loader(self.raw_matlab_data_frame, "Turbulence Dissipation", 3, "disabled")
        self.create_file_loader(self.raw_matlab_data_frame, "Inst. Velocity Frame", 4, "disabled")

        self.create_loader_checkbox(self.options_frame, "Turbulence Dissipation", 1)
        self.create_loader_checkbox(self.options_frame, "Inst. Velocity Frame", 2)

    def on_invalid_input(self):
        messagebox.showerror("Invalid Input", "Please enter a valid float.")

    def validate_float(self, input_value):
        if input_value == "":  # Allow empty input
            self.orientation = 0
            self.plot_graph()
            return True
        try:
            # Try to convert the input to a float
            float(input_value)
            self.plot_graph()
            return True
        except ValueError:
            self.on_invalid_input()
            return False

    def on_closing(self):
        self.canvas.get_tk_widget().grid_forget()  # Remove the canvas from the GUI
        plt.close(self.fig)  # Close the figure to free up resources

        self.root.destroy()

    def plot_graph(self):
        bev = Beverli(self.orientation, "cad")
        px, pz = bev.compute_perimeter(45)
        self.fig, ax = plt.subplots(figsize=(2, 2))
        # Create a matplotlib figure
        # fig = plt.figure(figsize=(2,2))
        # ax = fig.add_axes([0.15, 0.15, 0.85, 0.85])

        # Plot the data
        ax.plot(px, pz, label='y = x^2')
        ax.set_title("Sample Plot")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.legend()

        # Embed the plot in the Tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.bump_plot_frame)
        self.canvas.draw()

        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def create_section(self, section_name, section_type, master_frame, section_pos, title_pos):
        name = section_name.lower().replace(' ', '_')
        setattr(
            self,
            f"{name}_frame",
            tk.Frame(
                master_frame,
                bg=colors[f"f{section_type}_content"],
                borderwidth=1,
                relief="solid",
            )
        )
        new_frame = getattr(self, f"{name}_frame")
        new_frame.grid(**section_pos)
        new_frame.grid_columnconfigure(0, weight=1)

        section_title_fmt = (default_font[0], default_font[1] + 2, "bold")
        setattr(
            self,
            f"{name}_title",
            tk.Label(
                new_frame,
                text=section_name,
                font=section_title_fmt,
                borderwidth=1,
                relief="solid",
                bg=colors[f"f{section_type}_header"],
                fg="white",
            )
        )
        section_title = getattr(self, f"{name}_title")
        section_title.grid(**title_pos)


    def create_loader_checkbox(self, frame, quantity, row):
        name = quantity.lower().replace(' ', '_')
        setattr(self, f"checkbox_{name}_var", tk.IntVar())
        checkvar = getattr(self, f"checkbox_{name}_var")
        setattr(self, f"checkbox_{name}", tk.Checkbutton(frame, text=f"{quantity} ENABLE", variable=checkvar, command=lambda: self.toggle_state(quantity), bg=colors["f2_content"], fg="white"))
        checkbox = getattr(self, f"checkbox_{name}")
        checkbox.grid(row=row, column=0, sticky="w", padx=5)

    def toggle_state(self, quantity):
        name = quantity.lower().replace(' ', '_')
        button = getattr(self, f"load_button_{name}")
        status_label = getattr(self, f"status_label_{name}")
        listbox = getattr(self, f"listbox_{name}")
        checkbox_var = getattr(self, f"checkbox_{name}_var")

        if checkbox_var.get():
            button.config(state="normal")
            status_label.config(state="normal")
            listbox.config(state="normal")
        else:
            button.config(state="disabled")
            status_label.config(state="disabled")
            listbox.config(state="disabled")




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

        setattr(self, f"status_label_{dt}", tk.Label(frame, state=state, text="Nothing Loaded", width=13, bg=colors["f1_content"], fg="white"))
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
