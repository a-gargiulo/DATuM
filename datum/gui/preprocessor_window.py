"""Module defining the preprocessor window."""

import tkinter as tk

from tkinter import messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from .widgets import Button, Checkbutton, Entry, Label, Frame, Section, ScrollableCanvas
from ...utility.configure import system, STYLES

from ..core.beverli import Beverli
# from .pose import Pose


# Constants
WINDOW_TITLE = "Preprocessor"
WINDOW_SIZE = (800, 600)


class PreprocessorWindow:
    def __init__(self, master: tk.Tk) -> None:
        self.root = tk.Toplevel(master)
        self._configure_root()

        # Widgets
        # -------
        scrollable_canvas = ScrollableCanvas(self.root, True, False)
        self.main_frame = scrollable_canvas.get_frame()

        self.geom_sect = Section(self.main_frame, "Geometry", 1)
        self.bump_plt_frame = Frame(self.geom_sect.content, 1, bd=2, relief="solid")
        self.general_sect = Section(self.geom_sect.content, "General", 2)
        self.orientation_label = Label(self.general_sect.content, "Hill Orientation [deg]:", 2)
        self.orientation_entry = Entry(self.general_sect.content, 2)
        self.orientation_entry.config(validate="focusout", validatecommand=(self.vfcmd, "%P"))
        self.transform_sect = Section(self.geom_sect.content, "Pose & Transformation (Local PIV -> Global SWT)")
        self.pose_button = Button(self.transform_sect.content, "Load/Calculate Tranformation Matrix", self.open_pose)
        self.pose_button.config(width=200 if system == "Darwin" else 20)
        self.pose_status_label = Label(self.transform_sect.content, "Nothing Loaded", 2, fg="red")
        self.checkbox_interp = Checkbutton(self.transform_sect.content, 2)
        self.checkbox_interp.config(text="Interpolate data to regular grid", command=self.toggle_interp, anchor="w")
        self.checkbox_interp_var = self.checkbox_interp.get_var()
        self.interp_pts_label = Label(self.transform_sect.content, "Number of interp. grid points:", 2)
        self.interp_pts_label.config(state="disabled")
        self.interp_points_entry = Entry(self.transformation_content, 2, state="disabled")

        self.data_sect = Section(self.main_frame, "Raw (Matlab) Data", 2)
        self.data_section, self.data_content = gui.create_section(
            frame=self.main_frame,
            title="Raw (Matlab) Data",
            position={"row": 2, "column": 0, "columnspan": 2, "padx": 5, "pady": 5, "sticky": "nsew"},
            content_columnspan=4,
            section_kwargs={"bg": colors["s2_content"]},
            section_title_kwargs={"bg": colors["s2_header"], "fg": "white"},
            section_content_kwargs={"bg": colors["s2_content"]},
        )

        self.create_file_loader(self.data_content, "Mean Velocity", 0, "disabled", self.load_files)
        self.create_file_loader(self.data_content, "Reynolds Stress", 1, "disabled", self.load_files)
        self.create_file_loader(self.data_content, "Turbulence Dissipation", 2, "disabled", self.load_files)
        self.create_file_loader(self.data_content, "Velocity Frame", 3, "disabled", self.load_files)

        self.cfd_section, self.cfd_content = gui.create_section(
            frame=self.main_frame,
            title="Mean Velocity Gradient Tensor",
            position={"row": 3, "column": 0, "columnspan": 2, "padx": 5, "pady": 5, "sticky": "nsew"},
            content_columnspan=3,
            section_kwargs={"bg": colors["s1_content"]},
            section_title_kwargs={"bg": colors["s1_header"], "fg": "white"},
            section_content_kwargs={"bg": colors["s1_content"]},
        )

        self.gradient_checkbox_var = tk.IntVar()
        self.gradient_checkbox = tk.Checkbutton(
            self.cfd_content,
            text="Enable computation",
            variable=self.gradient_checkbox_var,
            command=self.toggle_gradient,
            bg=colors["s1_content"],
            fg="white",
            anchor="w",
            state="disabled",
        )

        self.gradient_opt_checkbox_var = tk.IntVar()
        self.gradient_opt_checkbox = tk.Checkbutton(
            self.cfd_content,
            text=r"dUdZ and dVdZ from CFD",
            variable=self.gradient_opt_checkbox_var,
            command=self.toggle_cfd,
            bg=colors["s1_content"],
            fg="white",
            anchor="w",
            state="disabled",
        )

        slice_button_font = (default_font[0], default_font[1], "bold")
        if system == "Darwin":
            additional_slice_button_params = {"width": 200, "borderless": 1}
        elif system == "Windows":
            additional_slice_button_params = {"width": 10}
        self.slice_button = Button(
            self.cfd_content,
            text="Load CFD Slice",
            command=self.load_cfd_slice,
            font=slice_button_font,
            bg=colors["accent"],
            fg=colors["s1_content"],
            **additional_slice_button_params,
            state="disabled",
        )

        self.slice_listbox = tk.Listbox(self.cfd_content, width=20, height=1, state="disabled")

        self.slice_status_label = tk.Label(
            self.cfd_content,
            text="Nothing Loaded",
            bg=colors["s1_content"],
            fg="red",
            state="disabled",
        )

        process_button_font = (default_font[0], default_font[1], "bold")
        if system == "Darwin":
            additional_process_button_params = {"width": 200, "borderless": 1}
        elif system == "Windows":
            additional_process_button_params = {"width": 20}

        self.process_button = Button(
            self.main_frame,
            text="Preprocess Data",
            command=self.preprocess_data,
            font=process_button_font,
            bg=colors["accent"],
            fg=colors["s1_content"],
            **additional_process_button_params,
        )

        # ----------------------------------------
        # UPDATES
        # ----------------------------------------
        self.orientation = 0
        self.orientation_entry.insert(self.orientation, "0")
        self.plot_graph()
        self.adjust_layout()
        scrollable_canvas.configure_frame()

    def _configure_root(self):
        """Configure main window settings."""
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}")
        self.root.resizable(False, False)
        self.root.configure(bg=STYLES["color"]["base"])
        self.root.option_add("*Font", (STYLES["font"], STYLES["font_size"]))
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.vfcmd = self.root.register(self._validate_float)

    def on_invalid_input(self):
        messagebox.showerror("Invalid Input", "Please enter a valid float.")

    def _validate_float(self, input_value):
        if input_value == "":  # Allow empty input
            self.orientation = 0
            self.plot_graph()
            return True
        try:
            # Try to convert the input to a float
            float(input_value)
            self.orientation = float(input_value)
            self.plot_graph()
            return True
        except ValueError:
            self.on_invalid_input()
            return False

    def _on_closing(self):
        if hasattr(self, "fig"):
            plt.close(self.fig)
            del self.fig

        if hasattr(self, "bump_canvas"):
            self.bump_canvas.get_tk_widget().grid_forget()  # Remove the canvas from the GUI
            self.bump_canvas.get_tk_widget().destroy()
            del self.bump_canvas

        self.root.destroy()

    def plot_graph(self):
        if hasattr(self, "fig") and hasattr(self, "bump_canvas"):
            self.ax.clear()
        else:
            self.fig = plt.figure(figsize=(2.5, 2.3))
            self.ax = self.fig.add_axes([0.3, 0.3, 0.75, 0.65])
            self.bump_canvas = FigureCanvasTkAgg(self.fig, master=self.bump_plt_frame)
            self.bump_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        bev = Beverli(self.orientation, "cad")
        px, pz = bev.compute_perimeter(self.orientation)

        # Plot the data
        self.ax.plot(px, pz, color="blue")
        self.ax.set_xlabel(r"$x_1$ (m)", labelpad=10)
        self.ax.set_ylabel(r"$x_3$ (m)", labelpad=10)
        self.ax.set_xlim(-0.65, 0.65)
        self.ax.set_ylim(0.65, -0.65)
        self.ax.set_aspect("equal")

        # Embed the plot in the Tkinter window
        self.bump_canvas.draw()

    def open_pose(self):
        Pose(self.root)

    def create_file_loader(self, frame, data_type, row, state, function):
        dt = data_type.lower().replace(" ", "_")

        button_font = (default_font[0], default_font[1], "bold")
        if system == "Darwin":
            additional_button_params = {"width": 200, "borderless": 1}
        elif system == "Windows":
            additional_button_params = {"width": 20}

        setattr(self, f"checkbox_{dt}_var", tk.IntVar())
        checkvar = getattr(self, f"checkbox_{dt}_var")
        setattr(
            self,
            f"checkbox_{dt}",
            tk.Checkbutton(
                frame,
                variable=checkvar,
                command=lambda: self.toggle_state(data_type),
                bg=colors["s2_content"],
                fg="black",
                anchor="w",
            ),
        )
        checkbox = getattr(self, f"checkbox_{dt}")
        checkbox.grid(row=row, column=0, sticky="w", padx=5)

        setattr(
            self,
            f"load_button_{dt}",
            Button(
                frame,
                text=f"{data_type}",
                command=lambda: function(data_type, frame),
                state=state,
                font=button_font,
                bg=colors["accent"],
                fg=colors["s2_content"],
                **additional_button_params,
            ),
        )

        load_button = getattr(self, f"load_button_{dt}")
        load_button.grid(row=row, column=1, padx=5, pady=5, sticky="nsew")
        # load_button.grid_columnconfigure(0, weight=1)

        setattr(self, f"listbox_{dt}", tk.Listbox(frame, state=state, width=20, height=1))
        listbox = getattr(self, f"listbox_{dt}")
        listbox.grid(row=row, column=2, padx=5, pady=5, sticky="nsew")
        # listbox.grid_columnconfigure(1, weight=1)

        setattr(
            self,
            f"status_label_{dt}",
            tk.Label(
                frame,
                state=state,
                text="Nothing Loaded",
                width=13,
                bg=colors["s2_content"],
                fg="red",
            ),
        )
        status_label = getattr(self, f"status_label_{dt}")
        status_label.grid(row=row, column=3, padx=5, pady=5, sticky="nsew")
        # status_label.grid_columnconfigure(2, weight=1)

    def load_files(self, data_type, section_frame):
        # Open file dialog and get the selected file paths
        file_path = filedialog.askopenfilename(filetypes=[("Data Files", "*.mat"), ("All Files", "*.*")])

        if file_path:
            # Get the corresponding listbox and status label for this data type
            listbox = getattr(self, f"listbox_{data_type.lower().replace(' ', '_')}")
            status_label = getattr(self, f"status_label_{data_type.lower().replace(' ', '_')}")

            # Clear previous items in the listbox
            listbox.delete(0, tk.END)

            # Add new file paths to the listbox
            listbox.insert(tk.END, file_path)

            status_label.config(text="File Loaded", fg="green")
        else:
            listbox = getattr(self, f"listbox_{data_type.lower().replace(' ', '_')}")
            status_label = getattr(self, f"status_label_{data_type.lower().replace(' ', '_')}")
            listbox.delete(0, tk.END)
            listbox.insert(tk.END, file_path)
            status_label.config(text="Nothing Loaded", fg="red")

    def toggle_gradient(self):
        if self.gradient_checkbox_var.get():
            self.gradient_opt_checkbox.config(state="normal")
            if self.gradient_opt_checkbox_var.get():
                self.slice_button.config(state="normal")
                self.slice_listbox.config(state="normal")
                self.slice_status_label.config(state="normal")
        else:
            self.gradient_opt_checkbox.config(state="disabled")
            self.gradient_opt_checkbox_var.set(0)
            self.slice_button.config(state="disabled")
            self.slice_listbox.config(state="disabled")
            self.slice_status_label.config(state="disabled")

    def toggle_state(self, data_type):
        dt = data_type.lower().replace(" ", "_")
        checkvar = getattr(self, f"checkbox_{dt}_var")
        load_button = getattr(self, f"load_button_{dt}")
        listbox = getattr(self, f"listbox_{dt}")
        status_label = getattr(self, f"status_label_{dt}")
        if checkvar.get():
            load_button.config(state="normal")
            listbox.config(state="normal")
            status_label.config(state="normal")
        else:
            load_button.config(state="disabled")
            listbox.config(state="disabled")
            status_label.config(state="disabled")

    def toggle_cfd(self):
        if self.gradient_opt_checkbox_var.get():
            self.slice_button.config(state="normal")
            self.slice_listbox.config(state="normal")
            self.slice_status_label.config(state="normal")
        else:
            self.slice_button.config(state="disabled")
            self.slice_listbox.config(state="disabled")
            self.slice_status_label.config(state="disabled")

    def load_cfd_slice(self):
        pass

    def preprocess_data(self):
        pass

    def toggle_interp(self):
        if self.checkbox_interpolation_var.get():
            self.interp_points_entry.config(state="normal")
            self.interp_pts_label.config(state="normal")
            self.gradient_checkbox.config(state="normal")
        else:
            self.interp_points_entry.config(state="disabled")
            self.interp_pts_label.config(state="disabled")
            self.gradient_checkbox.config(state="disabled")
            self.gradient_checkbox_var.set(0)
            self.gradient_opt_checkbox.config(state="disabled")
            self.gradient_opt_checkbox_var.set(0)
            self.slice_button.config(state="disabled")
            self.slice_listbox.config(state="disabled")
            self.slice_status_label.config(state="disabled")

    def adjust_layout(self):
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)

        self.geom_sect.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        self.geom_sect.content.grid(columnspan=2)
        self.geom_sect.content.grid_columnconfigure(0, weight=0)
        self.geom_sect.content.grid_columnconfigure(1, weight=1)

        self.general_sect.grid(row=0, column=1, columnspan=1, padx=(5, 0), pady=5, sticky="nsew")
        self.general_sect.content.grid(columnspan=2)
        self.general_sect.content.grid_columnconfigure(0, weight=0)
        self.general_sect.content.grid_columnconfigure(1, weight=1)

        self.bump_plt_frame.grid(row=0, column=0, columnspan=1, padx=(0, 5), pady=5, rowspan=2, sticky="nsew")
        self.orientation_label.grid(row=0, column=0, padx=5, pady=5, sticky="nsw")
        self.orientation_entry.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky="nsew")

        self.transform_sect.grid(row=1, column=1, columnspan=1, padx=(5, 0), pady= 5, sticky="nsew")
        self.transform_sect.content.gride(columnspan=3)
        self.transform_sect.content.grid_columnconfigure(0, weight=1)
        self.transform_sect.content.grid_columnconfigure(1, weight=1)
        self.transform_sect.content.grid_columnconfigure(2, weight=1)

        self.pose_button.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        self.pose_status_label.grid(row=0, column=2, padx=5, sticky="nsew")

        self.checkbox_interpolation.grid(row=1, column=0, padx=5, sticky="nsew")
        self.data_content.grid_columnconfigure(0, weight=1)
        self.data_content.grid_columnconfigure(1, weight=1)
        self.data_content.grid_columnconfigure(2, weight=1)
        self.data_content.grid_columnconfigure(3, weight=1)

        # self.interpolation_entry.grid(row=2, column=1, padx=5, sticky="nsew")
        self.interp_pts_label.grid(row=2, column=0, padx=5, pady=5, columnspan=2, sticky="nsw")
        self.interp_points_entry.grid(row=2, column=2, padx=5, pady=5, sticky="nsew")

        self.cfd_content.grid_columnconfigure(0, weight=1)
        self.cfd_content.grid_columnconfigure(1, weight=1)
        self.cfd_content.grid_columnconfigure(2, weight=1)

        self.gradient_checkbox.grid(row=0, column=0, sticky="nsew")
        self.gradient_opt_checkbox.grid(row=0, column=1, sticky="nsew")
        self.slice_button.grid(row=1, column=0, columnspan=1, padx=5, pady=5, sticky="nsew")
        self.slice_listbox.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.slice_status_label.grid(row=1, column=2, padx=5, sticky="nsew")

        self.process_button.grid(row=4, column=0, columnspan=2, pady=5, padx=10)
