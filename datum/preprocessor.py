import tkinter as tk
from tkinter import messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


from .config import system, default_font, Button, colors
from .beverli import Beverli
from .pose import Pose


class Preprocessor:
    def __init__(self, master: tk.Tk):
        self.root = tk.Toplevel(master)
        self.root.title("Preprocessor")
        self.root.geometry("800x625")
        self.root.resizable(False, False)
        self.root.configure(bg=colors["base"])
        self.root.option_add("*Font", default_font)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        vcmd = self.root.register(self.validate_float)


        # Geometry Frame
        # --------------
        geom_sec_pos = {"row": 0, "column": 0, "columnspan": 2, "padx": 15, "pady": 5, "sticky": "nsew"}
        geom_sec_title_pos = {"row": 0, "column": 0, "columnspan": 2, "padx": 5, "pady": 5, "ipady": 5, "sticky": "ew"}
        self.create_section("geometry", "Geometry", 1, self.root, geom_sec_pos, geom_sec_title_pos)
        self.geometry_frame.grid_columnconfigure(0, weight=0)
        self.geometry_frame.grid_columnconfigure(1, weight=1)
        # self.geometry_frame.grid_rowconfigure(1, weight=1)

        self.bump_plot_frame = tk.Frame(self.geometry_frame, borderwidth=2, relief="solid", bg=colors["f1_content"])
        self.bump_plot_frame.grid(row=1, column=0, columnspan=1, rowspan=2, padx=5, pady=5, sticky="nsew")
        self.bump_plot_frame.grid_columnconfigure(0, weight=1)
        self.bump_plot_frame.grid_rowconfigure(0, weight=1)
        self.bump_plot_frame.grid_rowconfigure(1, weight=1)

        general_sec_pos = {"row": 1, "column": 1, "columnspan": 1, "padx": 5, "pady": 5, "sticky": "nsew"}
        general_sec_title_pos = {"row": 0, "column": 0, "columnspan": 3, "padx": 5, "pady": 5, "ipady": 5, "sticky": "ew"}
        self.create_section("general", "General", 2, self.geometry_frame, general_sec_pos, general_sec_title_pos)
        # self.general_frame = tk.Frame(self.geometry_frame, bg=colors["f1_content"])
        # self.general_frame.grid(row=1, column=1, columnspan=2, padx=10, pady=5, sticky="nsew")
        self.general_frame.grid_columnconfigure(0, weight=1)
        self.general_frame.grid_columnconfigure(1, weight=1)
        self.general_frame.grid_columnconfigure(2, weight=1)
        # self.general_frame.grid_rowconfigure(0, weight=1)
        # self.general_frame.grid_rowconfigure(1, weight=1)
        # self.general_frame.grid_rowconfigure(2, weight=1)


        self.orientation_label = tk.Label(self.general_frame, text="Hill Orientation [deg]:", bg=colors["f2_content"], fg="white")
        self.orientation_label.grid(row=1, column=0, padx=5,  pady=5, sticky="nsew")
        self.orientation_entry = tk.Entry(self.general_frame, validate="focusout", validatecommand=(vcmd, '%P'), fg=colors["f1_content"], bd=1, relief="solid", highlightthickness=0, highlightbackground=colors["f1_content"])
        self.orientation_entry.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky="nsew")



        piv_sec_pos = {"row": 2, "column": 1, "columnspan": 1, "padx": 5, "pady": 5, "sticky": "nsew"}
        piv_sec_title_pos = {"row": 0, "column": 0, "columnspan": 3, "padx": 5, "pady": 5, "ipady": 5, "sticky": "ew"}
        self.create_section("piv", "Pose & Transformation (Local PIV -> Global SWT)", 2, self.geometry_frame, piv_sec_pos, piv_sec_title_pos)
        self.piv_frame.grid_columnconfigure(0, weight=1)
        self.piv_frame.grid_columnconfigure(1, weight=1)
        self.piv_frame.grid_columnconfigure(2, weight=1)


        pose_button_font = (default_font[0], default_font[1], "bold")
        if system == "Darwin":
            additional_pose_button_params = {"width": 200, "borderless": 1}
        elif system == "Windows":
            additional_pose_button_params = {"width": 20}

        self.pose_button = Button(self.piv_frame, text="Load/Calculate Tranformation Matrix", command=self.open_pose, font=pose_button_font, bg=colors["accent"], fg=colors["f1_content"], **additional_pose_button_params)
        self.pose_button.grid(row=1, column=0, columnspan=2,padx=5, pady=5, sticky="nsew")

        self.pose_status_label = tk.Label(self.piv_frame, text="Nothing Loaded", bg=colors["f2_content"], fg="red")
        self.pose_status_label.grid(row=1, column=2, padx=5, sticky="nsew")


        self.checkbox_interp_var = tk.IntVar()
        self.checkbox_interp = tk.Checkbutton(self.piv_frame, text="Interpolate to regular grid", variable=self.checkbox_interp_var, command=self.toggle_interp, bg=colors["f2_content"], fg="white", anchor="w")
        self.checkbox_interp.grid(row=2, column=0, padx=5, sticky="nsew")


        # self.interp_entry = tk.Entry(self.piv_frame, bd=1, relief="solid", highlightthickness=0, highlightbackground=colors["f2_content"])
        # self.interp_entry.grid(row=2, column=1, padx=5, sticky="nsew")
        # self.pose_listbox = tk.Listbox(self.piv_frame, width=20, height=1)
        # self.pose_listbox.grid(row=2, column=1, padx=5, pady=5, sticky="nsew")



        self.piv_plane_label = tk.Label(self.piv_frame, text="Number of interpolation grid points (square grid):", bg=colors["f2_content"], fg="white", state="disabled")
        self.piv_plane_label.grid(row=3, column=0, padx=5, pady=5, columnspan=2, sticky="nsw")
        self.piv_plane_entry = tk.Entry(self.piv_frame, bd=1, relief="solid", highlightthickness=0, highlightbackground=colors["f2_content"], state="disabled")
        self.piv_plane_entry.grid(row=3, column=2, padx=5, pady=5, sticky="nsew")



        self.orientation = 0
        self.plot_graph()

        # Loader Frame
        # ------------
        # self.loader_frame = tk.Frame(self.root, bg=colors["base"])
        # self.loader_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        # self.loader_frame.grid_columnconfigure(0, weight=1)


        data_sec_pos = {"row": 1, "column": 0,  "columnspan": 2, "padx": 5, "pady": 5, "sticky": "nsew"}
        data_title_pos = {"row": 0, "column": 0, "columnspan": 4, "padx": 5, "pady": 5, "ipady": 5, "sticky": "ew"}
        self.create_section("raw_matlab_data", "Raw (Matlab) Data", 1, self.root, data_sec_pos, data_title_pos)
        self.raw_matlab_data_frame.grid_columnconfigure(0, weight=1)
        self.raw_matlab_data_frame.grid_columnconfigure(1, weight=1)
        self.raw_matlab_data_frame.grid_columnconfigure(2, weight=1)
        self.raw_matlab_data_frame.grid_columnconfigure(3, weight=1)

        cfd_sec_pos = {"row": 2, "column": 0, "columnspan": 2, "padx": 5, "pady": 5, "sticky": "nsew"}
        cfd_title_pos = {"row": 0, "column": 0, "columnspan": 3, "padx": 5, "pady": 5, "ipady": 5, "ipadx": 10, "sticky": "ew"}
        self.create_section("cfd", "Mean Velocity Gradient Tensor", 2, self.root, cfd_sec_pos, cfd_title_pos)
        self.cfd_frame.grid_columnconfigure(0, weight=1)
        self.cfd_frame.grid_columnconfigure(1, weight=1)
        self.cfd_frame.grid_columnconfigure(2, weight=1)

        self.create_file_loader(self.raw_matlab_data_frame, "Mean Velocity", 1, "disabled", self.load_files)
        self.create_file_loader(self.raw_matlab_data_frame, "Reynolds Stress", 2, "disabled", self.load_files)
        self.create_file_loader(self.raw_matlab_data_frame, "Turbulence Dissipation", 3, "disabled", self.load_files)
        self.create_file_loader(self.raw_matlab_data_frame, "Velocity Frame", 4, "disabled", self.load_files)




        self.gradient_checkbox_var = tk.IntVar()
        self.gradient_checkbox = tk.Checkbutton(self.cfd_frame, text="Enable computation", variable=self.gradient_checkbox_var, command=self.toggle_gradient, bg=colors["f2_content"], fg="white", anchor="w", state="disabled")
        self.gradient_checkbox.grid(row=1, column=0, sticky="nsew")

        slice_button_font = (default_font[0], default_font[1], "bold")
        if system == "Darwin":
            additional_slice_button_params = {"width": 200, "borderless": 1}
        elif system == "Windows":
            additional_slice_button_params = {"width": 10}

        self.slice_button = Button(self.cfd_frame, text="Load CFD Slice", command=self.load_cfd_slice, font=slice_button_font, bg=colors["accent"], fg=colors["f1_content"], **additional_slice_button_params, state="disabled")
        self.slice_button.grid(row=2, column=0, columnspan=1,padx=5, pady=5, sticky="nsew")

        self.slice_listbox = tk.Listbox(self.cfd_frame, width=20, height=1, state="disabled")
        self.slice_listbox.grid(row=2, column=1, padx=5, pady=5, sticky="nsew")

        self.slice_status_label = tk.Label(self.cfd_frame, text="Nothing Loaded", bg=colors["f2_content"], fg="red", state="disabled")
        self.slice_status_label.grid(row=2, column=2, padx=5, sticky="nsew")



        self.gradient_opt_checkbox_var = tk.IntVar()
        self.gradient_opt_checkbox = tk.Checkbutton(self.cfd_frame, text=r"dUdZ and dVdZ from CFD", variable=self.gradient_opt_checkbox_var, command=self.toggle_cfd, bg=colors["f2_content"], fg="white", anchor="w", state="disabled")
        self.gradient_opt_checkbox.grid(row=1, column=1, sticky="nsew")



        process_button_font = (default_font[0], default_font[1], "bold")
        if system == "Darwin":
            additional_process_button_params = {"width": 200, "borderless": 1}
        elif system == "Windows":
            additional_process_button_params = {"width": 20}



        self.process_button = Button(self.root, text="Preprocess Data", command=self.preprocess_data, font=process_button_font, bg=colors["accent"], fg=colors["f1_content"], **additional_process_button_params)
        self.process_button.grid(row=3, column=0, columnspan=2, pady=5, padx=10)



#         self.create_loader_checkbox(self.options_frame, "Mean Velocity", 1)
#         self.create_loader_checkbox(self.options_frame, "Reynolds Stress", 2)
#         self.create_loader_checkbox(self.options_frame, "Turbulence Dissipation", 3)
#         self.create_loader_checkbox(self.options_frame, "Velocity Frame", 4)

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
            self.orientation = float(input_value)
            self.plot_graph()
            return True
        except ValueError:
            self.on_invalid_input()
            return False

    def on_closing(self):
        if hasattr(self, 'fig'):
            plt.close(self.fig)
            del self.fig

        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().grid_forget()  # Remove the canvas from the GUI
            self.canvas.get_tk_widget().destroy()
            del self.canvas

        self.root.destroy()

    def plot_graph(self):
        if hasattr(self, 'fig') and hasattr(self, 'canvas'):
            self.ax.clear()
        else:
            self.fig = plt.figure(figsize=(2.4, 2.2))
            self.ax = self.fig.add_axes([0.3, 0.3, 0.75, 0.65])
            # self.fig, self.ax = plt.subplots(figsize=(2,2))
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.bump_plot_frame)
            self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        bev = Beverli(self.orientation, "cad")
        px, pz = bev.compute_perimeter(self.orientation)

        # self.fig, ax = plt.subplots(figsize=(2, 2))
        # Create a matplotlib figure
        # fig = plt.figure(figsize=(2,2))
        # ax = fig.add_axes([0.15, 0.15, 0.85, 0.85])

        # Plot the data
        self.ax.plot(px, pz, color="blue")
        self.ax.set_xlabel(r"$x_1$ (m)", labelpad=10)
        self.ax.set_ylabel(r"$x_3$ (m)", labelpad=10)
        self.ax.set_xlim(-0.65, 0.65)
        self.ax.set_ylim(0.65, -0.65)
        self.ax.set_aspect('equal')


        # Embed the plot in the Tkinter window
        # self.canvas = FigureCanvasTkAgg(self.fig, master=self.bump_plot_frame)
        self.canvas.draw()

        # self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
    def open_pose(self):
        Pose(self.root)


    def create_section(self, varname, section_title, section_type, master_frame, section_pos, title_pos):
        # name = section_name.lower().replace(' ', '_')
        setattr(
            self,
            f"{varname}_frame",
            tk.Frame(
                master_frame,
                bg=colors[f"f{section_type}_content"],
                borderwidth=1,
                relief="solid",
            )
        )
        new_frame = getattr(self, f"{varname}_frame")
        new_frame.grid(**section_pos)
        new_frame.grid_columnconfigure(0, weight=1)

        section_title_fmt = (default_font[0], default_font[1] + 2, "bold")
        setattr(
            self,
            f"{varname}_title",
            tk.Label(
                new_frame,
                text=section_title,
                font=section_title_fmt,
                borderwidth=1,
                relief="solid",
                bg=colors[f"f{section_type}_header"],
                fg="white",
            )
        )
        section_title = getattr(self, f"{varname}_title")
        section_title.grid(**title_pos)


    # def create_loader_checkbox(self, frame, quantity, row):
    #     name = quantity.lower().replace(' ', '_')
    #     setattr(self, f"checkbox_{name}_var", tk.IntVar())
    #     checkvar = getattr(self, f"checkbox_{name}_var")
    #     setattr(self, f"checkbox_{name}", tk.Checkbutton(frame, text=f"{quantity}", variable=checkvar, command=lambda: self.toggle_state(quantity), bg=colors["f2_content"], fg=colors["accent"], anchor="w"))
    #     checkbox = getattr(self, f"checkbox_{name}")
    #     checkbox.grid(row=row, column=0, sticky="w", padx=5)

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




    def create_file_loader(self, frame, data_type, row, state, function):
        dt = data_type.lower().replace(' ', '_')

        button_font = (default_font[0], default_font[1], "bold")
        if system == "Darwin":
            additional_button_params = {"width": 200, "borderless": 1}
        elif system == "Windows":
            additional_button_params = {"width": 20}

        setattr(self, f"checkbox_{dt}_var", tk.IntVar())
        checkvar = getattr(self, f"checkbox_{dt}_var")
        setattr(self, f"checkbox_{dt}", tk.Checkbutton(frame, variable=checkvar, command=lambda: self.toggle_state(data_type), bg=colors["f1_content"], fg="black", anchor="w"))
        checkbox = getattr(self, f"checkbox_{dt}")
        checkbox.grid(row=row, column=0, sticky="w", padx=5)

        setattr(self, f"load_button_{dt}", Button(frame, text=f"{data_type}", command=lambda: function(data_type, frame), state=state, font=button_font, bg=colors["accent"], fg=colors["f1_content"], **additional_button_params))

        load_button = getattr(self, f"load_button_{dt}")
        load_button.grid(row=row, column=1, padx=5, pady=5, sticky="nsew")
        # load_button.grid_columnconfigure(0, weight=1)

        setattr(self, f"listbox_{dt}", tk.Listbox(frame, state=state, width=20, height=1))
        listbox = getattr(self, f"listbox_{dt}")
        listbox.grid(row=row, column=2, padx=5, pady=5, sticky="nsew")
        # listbox.grid_columnconfigure(1, weight=1)

        setattr(self, f"status_label_{dt}", tk.Label(frame, state=state, text="Nothing Loaded", width=13, bg=colors["f1_content"], fg="red"))
        status_label = getattr(self, f"status_label_{dt}")
        status_label.grid(row=row, column=3, padx=5, pady=5, sticky="nsew")
        # status_label.grid_columnconfigure(2, weight=1)

    def load_files(self, data_type, section_frame):
        # Open file dialog and get the selected file paths
        file_path = filedialog.askopenfilename(
            filetypes=[("Data Files", "*.mat"), ("All Files", "*.*")]
        )

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
        if self.checkbox_interp_var.get():
            self.piv_plane_entry.config(state="normal")
            self.piv_plane_label.config(state="normal")
            self.gradient_checkbox.config(state="normal")
        else:
            self.piv_plane_entry.config(state="disabled")
            self.piv_plane_label.config(state="disabled")
            self.gradient_checkbox.config(state="disabled")
            self.gradient_checkbox_var.set(0)
            self.gradient_opt_checkbox.config(state="disabled")
            self.gradient_opt_checkbox_var.set(0)
            self.slice_button.config(state="disabled")
            self.slice_listbox.config(state="disabled")
            self.slice_status_label.config(state="disabled")

