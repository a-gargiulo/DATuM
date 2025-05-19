"""Profiler application."""

import tkinter as tk
from tkinter import messagebox
from typing import cast, Optional

from datum.core import profiles
from datum.core.beverli import Beverli
from datum.core.my_types import PivData, PRInputs
from datum.core.piv import Piv
from datum.gui.widgets import (
    Button,
    Checkbutton,
    Field,
    FileLoader,
    Label,
    ScrollableCanvas,
    Section,
)
from datum.utility import apputils
from datum.utility.configure import STYLES
from datum.utility.logging import logger


# ---------- Constants ----------
WINDOW_TITLE = "Profiler"
WINDOW_SIZE = (600, 600)
PAD_S = STYLES["pad"]["small"]
SYM_ORIENTATIONS = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0]


class ProfilerWindow:
    """Profiler application GUI."""

    def __init__(self, master: tk.Tk) -> None:
        """Initialize GUI.

        :param master: Parent window handle.
        """
        self.root = tk.Toplevel(master)
        self.configure_root()
        self.create_widgets()
        self.layout_widgets()
        self.scrollable_canvas.configure_frame()
        logger.info("Profiler window opened successfully.")

    def configure_root(self) -> None:
        """Configure window."""
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}")
        self.root.resizable(False, False)
        self.root.configure(bg=STYLES["color"]["base"])
        self.root.option_add(
            "*Font", (STYLES["font"], STYLES["font_size"]["regular"])
        )
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self) -> None:
        """Create all widget entities."""
        # ---------- Main ----------
        self.scrollable_canvas = ScrollableCanvas(self.root, True, False)
        self.main_frame = self.scrollable_canvas.frame

        # ********** General **********
        self.general_section = Section(self.main_frame, "General", 1)

        # ********** Hill Orientation **********
        self.hill_orientation = Field(
            self.general_section.content, 1, "Hill orientation [deg]:"
        )

        # ********** Reynolds Number **********
        self.reynolds_number = Field(
            self.general_section.content, 1, "Reynolds number:"
        )

        # ********** Tunnel Entry **********
        self.tunnel_entry = Field(
            self.general_section.content, 1, "Wind tunnel entry:"
        )

        # ********** Air Properties **********
        self.bypass_properties = Checkbutton(
            self.general_section.content,
            category=1,
            text=(
                "Bypass automatic calculation of "
                "selected fluid and flow properties"
            ),
            command=self.toggle_bypass,
        )

        self.bypass_section = Section(
            self.general_section.content, "Enter a number to enable bypass. To disable, enter 'NaN'.", 2)

        self.gas_constant = Field(
            self.bypass_section.content, 2, "Gas constant [J/kg/k]:"
        )
        self.gas_constant.entry.insert(0, "287.0")

        self.gamma = Field(
            self.bypass_section.content, 2, "Heat capacity ratio:"
        )
        self.gamma.entry.insert(0, "1.4")

        self.density = Field(
            self.bypass_section.content, 2, "Density [kg/m3]:"
        )
        self.density.entry.insert(0, "1.103")

        self.mu = Field(
            self.bypass_section.content, 2, "Dynamic viscosity [kg/m/s]:"
        )
        self.mu.entry.insert(0, "1.8559405e-5")

        self.uinf = Field(self.bypass_section.content, 2, "U_inf [m/s]:")
        self.uinf.entry.insert(0, "NaN")

        # ********** Add Gradients Checkbox **********
        self.add_gradients = Checkbutton(
            self.general_section.content,
            category=1,
            text="Add gradients",
            command=self.toggle_gradients,
        )

        # ********** PIV Data **********
        self.data_loader = FileLoader(
            self.general_section.content,
            title="Piv data (no interp.)",
            filetypes=[("Pickle File", "*.pkl"), ("All Files", "*.*")],
            category=1,
            isCheckable=False,
        )
        self.data_loader_interp = FileLoader(
            self.general_section.content,
            title="Piv data (interp.)",
            filetypes=[("Pickle File", "*.pkl"), ("All Files", "*.*")],
            category=1,
            isCheckable=False,
        )
        self.data_loader_interp.load_button.config(state="disabled")
        self.data_loader_interp.listbox.config(state="disabled")
        self.data_loader_interp.status_label.config(state="disabled")

        # ********** Transformation Parameters **********
        self.pose_loader = FileLoader(
            self.general_section.content,
            title="Transformation parameters",
            filetypes=[("Pose File", "*.json"), ("All Files", "*.*")],
            category=1,
            isCheckable=False,
        )

        # ********** Reference Conditions **********
        self.ref_conditions_loader = FileLoader(
            self.general_section.content,
            title="Reference conditions (.stat)",
            filetypes=[
                ("Reference Conditions File", "*.stat"),
                ("All Files", "*.*"),
            ],
            category=1,
            isCheckable=False,
        )

        # ---------- Profiler ----------
        self.profiler_section = Section(self.main_frame, "Profiles", 2)

        # ********** Coordinate System **********
        self.coordinates_selector_label = Label(
            self.profiler_section.content, "Coordinate system:", 2
        )
        self.coordinates_selector_var = tk.StringVar()
        self.coordinates_selector_var.set("Tunnel")
        self.coordinates_selector = tk.OptionMenu(
            self.profiler_section.content,
            self.coordinates_selector_var,
            "Tunnel",
            "Shear",
        )
        self.coordinates_selector_var.trace("w", self.on_coordinates_selection)

        # ********** Number of Profiles **********
        self.num_profile = Field(
            self.profiler_section.content, 2, "Number of profiles:"
        )

        # ********** Number of Profile Points **********
        self.num_profile_pts = Field(
            self.profiler_section.content, 2, "Number of profile points:"
        )

        # ********** Profile Height **********
        self.profile_height = Field(
            self.profiler_section.content, 2, "Profile height [m]:"
        )

        # ********** Reconstruct Profile **********
        self.reconstruction_checkbox = Checkbutton(
            self.profiler_section.content,
            category=2,
            text="Add reconstruction points",
            command=self.toggle_reconstruction,
        )
        self.reconstruction_checkbox.config(state="disabled")

        # ********** Number of Reconstruction Points **********
        self.num_reconstruction_pts = Field(
            self.profiler_section.content,
            2,
            "Number of reconstruction points:",
        )
        self.num_reconstruction_pts.label.config(state="disabled")
        self.num_reconstruction_pts.entry.config(state="disabled")

        # ********** Add CFD **********
        self.checkbox_cfd = Checkbutton(
            self.profiler_section.content,
            category=2,
            text="Extract CFD profiles (expensive)",
            command=self.toggle_cfd,
        )

        # ********** Fluent Data **********
        self.fluent_case_loader = FileLoader(
            self.profiler_section.content,
            title="Fluent case",
            filetypes=[("Fluent Case", "*.cas"), ("All Files", ".*.")],
            category=2,
            isCheckable=False,
        )
        self.fluent_data_loader = FileLoader(
            self.profiler_section.content,
            title="Fluent data",
            filetypes=[("Fluent Data", "*.dat"), ("All Files", ".*.")],
            category=2,
            isCheckable=False,
        )
        self.fluent_case_loader.load_button.config(state="disabled")
        self.fluent_case_loader.listbox.config(state="disabled")
        self.fluent_case_loader.status_label.config(state="disabled")
        self.fluent_data_loader.load_button.config(state="disabled")
        self.fluent_data_loader.listbox.config(state="disabled")
        self.fluent_data_loader.status_label.config(state="disabled")

        # ********** Pressure Data (for Boundary Layer Calculation) **********
        self.pressure_section = Section(
            self.profiler_section.content,
            "Boundary layer parameters caluclation",
            1,
        )
        self.port_loader = FileLoader(
            self.pressure_section.content,
            title="Port Wall Pressure",
            filetypes=[
                ("Port Wall Pressure File", "*.stat"),
                ("All Files", "*.*"),
            ],
            category=1,
            isCheckable=False,
        )
        self.hill_loader = FileLoader(
            self.pressure_section.content,
            title="Hill Surface Pressure",
            filetypes=[
                ("Hill Surface Pressure File", "*.stat"),
                ("All Files", "*.*"),
            ],
            category=1,
            isCheckable=False,
        )
        self.info_loader = FileLoader(
            self.pressure_section.content,
            title="Pressure Data Info File",
            filetypes=[
                ("Pressure Data Info File", "*.stat"),
                ("All Files", "*.*"),
            ],
            category=1,
            isCheckable=False,
        )

        # ********** Submit Button **********
        self.calculate_button = Button(
            self.main_frame, text="Submit", command=self.extract_profiles
        )

    def layout_widgets(self) -> None:
        """Layout widgets on the window."""
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.general_section.grid(
            row=0, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew"
        )
        self.general_section.content.grid(columnspan=3)
        self.general_section.content.grid_columnconfigure(0, weight=1)
        self.general_section.content.grid_columnconfigure(1, weight=1)
        self.general_section.content.grid_columnconfigure(2, weight=1)

        self.hill_orientation.grid(row=0, column=0)
        self.hill_orientation.label.grid(padx=(PAD_S, PAD_S))
        self.hill_orientation.entry.grid(padx=0)

        self.reynolds_number.grid(row=1, column=0)
        self.reynolds_number.label.grid(padx=(PAD_S, 3.5 * PAD_S))
        self.reynolds_number.entry.grid(padx=0)

        self.tunnel_entry.grid(row=2, column=0)
        self.tunnel_entry.label.grid(padx=(PAD_S, 3.5 * PAD_S))
        self.tunnel_entry.entry.grid(padx=0)

        self.bypass_properties.grid(
            row=3,
            column=0,
            columnspan=3,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )

        self.add_gradients.grid(
            row=4,
            column=0,
            columnspan=3,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.data_loader.grid(
            row=5,
            column=0,
            columnspan=3,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.data_loader_interp.grid(
            row=6,
            column=0,
            columnspan=3,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.pose_loader.grid(
            row=7,
            column=0,
            columnspan=3,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.ref_conditions_loader.grid(
            row=8,
            column=0,
            columnspan=3,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )

        self.profiler_section.grid(
            row=2, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew"
        )
        self.profiler_section.content.grid(columnspan=3)
        self.profiler_section.content.grid_columnconfigure(0, weight=1)
        self.profiler_section.content.grid_columnconfigure(1, weight=1)
        self.profiler_section.content.grid_columnconfigure(2, weight=1)
        self.coordinates_selector_label.grid(
            row=0, column=0, padx=PAD_S, pady=PAD_S, sticky="nsw"
        )
        self.coordinates_selector.grid(
            row=0,
            column=1,
            columnspan=2,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )

        self.num_profile.grid(row=1, column=0, columnspan=3)
        self.num_profile.label.grid(padx=(0, 15 * PAD_S))
        self.num_profile.entry.grid(padx=0)

        self.num_profile_pts.grid(row=2, column=0, columnspan=3)
        self.num_profile_pts.label.grid(padx=(0, 9 * PAD_S))
        self.num_profile_pts.entry.grid(padx=0)

        self.profile_height.grid(row=3, column=0, columnspan=3)
        self.profile_height.label.grid(padx=(0, 17 * PAD_S))
        self.profile_height.entry.grid(padx=0)

        self.reconstruction_checkbox.grid(
            row=4, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew"
        )

        self.num_reconstruction_pts.grid(row=5, column=0, columnspan=3)
        self.num_reconstruction_pts.label.grid(padx=(0, PAD_S))
        self.num_reconstruction_pts.entry.grid(padx=0)

        self.checkbox_cfd.grid(
            row=6, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew"
        )
        self.fluent_case_loader.grid(
            row=7,
            column=0,
            columnspan=3,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.fluent_data_loader.grid(
            row=8,
            column=0,
            columnspan=3,
            padx=PAD_S,
            pady=PAD_S,
            sticky="nsew",
        )
        self.calculate_button.grid(
            row=3, column=0, padx=PAD_S, pady=PAD_S, sticky="ns"
        )

    def on_closing(self) -> None:
        """Free resources when closing the window."""
        self.root.destroy()
        logger.info("Profiler window closed successfully.")

    def toggle_gradients(self) -> None:
        """Activate/deactivate the addition of gradient data."""
        if bool(self.add_gradients.var.get()):
            ss = "normal"
        else:
            ss = "disabled"
        self.data_loader_interp.load_button.config(state=ss)
        self.data_loader_interp.listbox.config(state=ss)
        self.data_loader_interp.status_label.config(state=ss)

    def toggle_reconstruction(self) -> None:
        """Activate/deactivate the profile reconstruction option."""
        if bool(self.reconstruction_checkbox.var.get()):
            ss = "normal"
        else:
            ss = "disabled"
        self.num_reconstruction_pts.label.config(state=ss)
        self.num_reconstruction_pts.entry.config(state=ss)

    def toggle_bypass(self) -> None:
        """Activate/deactivate properties bypassing."""
        if bool(self.bypass_properties.var.get()):
            self.bypass_section.grid(
                row=4,
                column=0,
                columnspan=3,
                padx=PAD_S,
                pady=PAD_S,
                sticky="nsew",
            )
            self.bypass_section.content.grid_columnconfigure(0, weight=1)
            self.bypass_section.content.grid_columnconfigure(1, weight=1)
            self.bypass_section.content.grid_columnconfigure(2, weight=1)

            self.gas_constant.label.grid(padx=(0, 5.5 * PAD_S))
            self.gamma.label.grid(padx=(0, 7.5 * PAD_S))
            self.uinf.label.grid(padx=(0, 16 * PAD_S))
            self.density.label.grid(padx=(0, 10.5 * PAD_S))

            self.add_gradients.grid(row=5)
            self.data_loader.grid(row=6)
            self.data_loader_interp.grid(row=7)
            self.pose_loader.grid(row=8)
            self.ref_conditions_loader.grid(row=9)
            self.scrollable_canvas.configure_frame()
        else:
            self.bypass_section.grid_forget()
            self.add_gradients.grid(row=4)
            self.data_loader.grid(row=5)
            self.data_loader_interp.grid(row=6)
            self.pose_loader.grid(row=7)
            self.ref_conditions_loader.grid(row=8)
            self.scrollable_canvas.configure_frame()

    def on_coordinates_selection(self, *args) -> None:
        """Handle inputs based on the choice of coordiante system."""
        if self.coordinates_selector_var.get() == "Shear":
            self.reconstruction_checkbox.config(state="normal")
            self.num_reconstruction_pts.label.config(state="disabled")
            self.num_reconstruction_pts.entry.config(state="disabled")

            self.pressure_section.grid(
                row=9,
                column=0,
                padx=2 * PAD_S,
                pady=PAD_S,
                sticky="nsew",
                columnspan=3,
            )
            self.pressure_section.content.grid(columnspan=3)
            self.pressure_section.content.grid_columnconfigure(0, weight=1)
            self.pressure_section.content.grid_columnconfigure(1, weight=1)
            self.pressure_section.content.grid_columnconfigure(2, weight=1)
            self.port_loader.grid(
                row=0,
                column=0,
                columnspan=3,
                padx=PAD_S,
                pady=PAD_S,
                sticky="nsew",
            )
            self.hill_loader.grid(
                row=1,
                column=0,
                columnspan=3,
                padx=PAD_S,
                pady=PAD_S,
                sticky="nsew",
            )
            self.info_loader.grid(
                row=2,
                column=0,
                columnspan=3,
                padx=PAD_S,
                pady=PAD_S,
                sticky="nsew",
            )
            self.scrollable_canvas.configure_frame()
        else:
            self.reconstruction_checkbox.var.set(0)
            self.reconstruction_checkbox.config(state="disabled")
            self.num_reconstruction_pts.label.config(state="disabled")
            self.num_reconstruction_pts.entry.config(state="disabled")
            self.port_loader.reset()
            self.hill_loader.reset()
            self.info_loader.reset()
            self.pressure_section.grid_forget()

    def toggle_cfd(self) -> None:
        """Activate/deactivate an equivalent profile extraction from CFD."""
        if bool(self.checkbox_cfd.var.get()):
            ss = "normal"
        else:
            ss = "disabled"
        self.fluent_case_loader.load_button.config(state=ss)
        self.fluent_case_loader.listbox.config(state=ss)
        self.fluent_case_loader.status_label.config(state=ss)
        self.fluent_data_loader.load_button.config(state=ss)
        self.fluent_data_loader.listbox.config(state=ss)
        self.fluent_data_loader.status_label.config(state=ss)

    def collect_ui(self) -> PRInputs:
        """Collect user inputs.

        :raises RuntimeError: If the selected hill orientation is not allowed.
        :raises ValueError: If any of the inputs is provided in the wrong
            format.
        :return: User inputs.
        :rtype: PRInputs
        """
        ui: PRInputs = {
            "hill_orientation": float(self.hill_orientation.entry.get()),
            "reference_stat_file": (
                self.ref_conditions_loader.get_listbox_content()
            ),
            "reynolds_number": float(self.reynolds_number.entry.get()),
            "tunnel_entry": int(self.tunnel_entry.entry.get()),
            "bypass_properties": bool(self.bypass_properties.var.get()),
            "gas_constant": None,
            "gamma": None,
            "density": None,
            "mu": None,
            "uinf": None,
            "add_gradients": bool(self.add_gradients.var.get()),
            "add_cfd": bool(self.checkbox_cfd.var.get()),
            "fluent_case": None,
            "fluent_data": None,
            "number_of_profiles": int(self.num_profile.entry.get()),
            "number_of_profile_pts": int(self.num_profile_pts.entry.get()),
            "coordinate_system": str(self.coordinates_selector_var),
            "profile_height": float(self.profile_height.entry.get()),
            "port_wall_pressure": None,
            "hill_pressure": None,
            "pressure_readme": None,
            "add_reconstruction_points": None,
            "number_of_reconstruction_points": None,
        }

        def collect_bypass(quantity: str) -> Optional[float]:
            if quantity == "NaN":
                return None
            else:
                return float(quantity)

        if ui["bypass_properties"]:
            ui["gas_constant"] = collect_bypass(self.gas_constant.entry.get())
            ui["gamma"] = collect_bypass(self.gamma.entry.get())
            ui["density"] = collect_bypass(self.density.entry.get())
            ui["mu"] = collect_bypass(self.density.entry.get())
            ui["uinf"] = collect_bypass(self.uinf.entry.get())

        if ui["coordinate_system"] == "Shear":
            if ui["hill_orientation"] not in SYM_ORIENTATIONS:
                raise RuntimeError(
                    "Only symmetric hill orientations allowed in 'Shear' "
                    "mode."
                )

            ui["port_wall_pressure"] = self.port_loader.get_listbox_content()
            ui["hill_pressure"] = self.hill_loader.get_listbox_content()
            ui["pressure_readme"] = self.info_loader.get_listbox_content()

            ui["add_reconstruction_points"] = bool(
                self.reconstruction_checkbox.var.get()
            )
            if ui["add_reconstruction_points"]:
                ui["number_of_reconstruction_points"] = int(
                    self.num_reconstruction_pts.entry.get()
                )

        return ui

    def ready_to_extract(self) -> bool:
        """
        Check the status of all required input fields.

        :return: Boolean indicating whether all inputs are valid or not.
        """
        loaders = [
            self.data_loader,
            self.ref_conditions_loader,
            self.pose_loader,
        ]

        entries = [
            self.hill_orientation.entry,
            self.reynolds_number.entry,
            self.tunnel_entry.entry,
            self.num_profile.entry,
            self.num_profile_pts,
            self.profile_height
        ]

        if bool(self.bypass_properties.var.get()):
            entries.append(self.gas_constant.entry)
            entries.append(self.gamma.entry)
            entries.append(self.density.entry)
            entries.append(self.mu.entry)
            entries.append(self.uinf.entry)

        if bool(self.add_gradients.var.get()):
            loaders.append(self.data_loader_interp)

        if bool(self.checkbox_cfd.var.get()):
            loaders.append(self.fluent_case_loader)
            loaders.append(self.fluent_data_loader)

        if self.coordinates_selector_var.get() == "Shear":
            if bool(self.reconstruction_checkbox.var.get()):
                entries.append(self.num_reconstruction_pts.entry)
            loaders.append(self.hill_loader)
            loaders.append(self.port_loader)
            loaders.append(self.info_loader)

        empty_fields = [entry for entry in entries if not entry.get().strip()]
        if empty_fields:
            messagebox.showwarning("Warning", "Please fill in all fields.")
            return False

        for loader in loaders:
            if loader.load_button.cget("state") == "normal":
                if loader.status_label_var.get() == "Nothing Loaded":
                    messagebox.showwarning(
                        "Warning",
                        (
                            r"Not all requied data has been loaded! "
                            r"Please load all data."
                        ),
                    )
                    return False

        return True

    def extract_profiles(self) -> None:
        """Run the profiler to extract 1D profiles from 2D PIV data."""
        try:
            if not self.ready_to_extract():
                return

            ui = self.collect_ui()

            self.geometry = Beverli(ui["hill_orientation"], use_cad=True)

            tp_path = self.pose_loader.get_listbox_content()
            trans_params = apputils.load_transformation_parameters(tp_path)
            pose = apputils.make_pose_from_trans_params(trans_params)

            pth_no_interp = self.data_loader.get_listbox_content()
            data_no_interp = cast(PivData, apputils.load_pickle(pth_no_interp))

            self.piv_intrp = None
            if ui["add_gradients"]:
                pth_interp = self.data_loader_interp.get_listbox_content()
                data_interp = cast(PivData, apputils.load_pickle(pth_interp))
                self.piv_intrp = Piv(data_interp, pose)

            self.piv_no_intrp = Piv(data_no_interp, pose)

            # ==================================================
            profiles.extract_profiles(
                self.piv_no_intrp, self.piv_intrp, self.geometry, ui
            )
            # ==================================================

            logger.info("Profile extraction completed successfully.")
            messagebox.showinfo(
                "INFO", "Profile extraction completed successfully."
            )
        except Exception as e:
            logger.error(
                f"An error occured during the profile extraction: {e}"
            )
            messagebox.showerror(
                "ERROR!",
                "An error occured during the profile extraction. "
                "Read the log and try again.",
            )
            return
