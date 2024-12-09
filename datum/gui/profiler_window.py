"""Create the profiler application window."""
import sys
import tkinter as tk
from ..utility.configure import STYLES
from .widgets import Button, Checkbutton, Entry, FileLoader, Label, ScrollableCanvas, Section
from ..core.beverli import Beverli
from ..core.piv import Piv
from ..core.pose import Pose
from ..utility import apputils
from ..core.my_types import PivData
from ..core import profiles
from typing import cast
# Constants
WINDOW_TITLE = "Profiler"
WINDOW_SIZE = (600, 600)


class ProfilerWindow:
    """Generate the GUI for the profiler window and link it to the core functions."""

    def __init__(self, master: tk.Tk):
        """Initialize GUI."""
        self.root = tk.Toplevel(master)
        self._configure_root()
        self._create_widgets()
        self._layout_widgets()
        self.scrollable_canvas.configure_frame()

    def _configure_root(self):
        """Configure main window settings."""
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}")
        self.root.resizable(False, False)
        self.root.configure(bg=STYLES["color"]["base"])
        self.root.option_add("*Font", (STYLES["font"], STYLES["font_size"]["regular"]))
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _create_widgets(self):
        """Create all widget entities."""
        self.scrollable_canvas = ScrollableCanvas(self.root, True, False)
        self.main_frame = self.scrollable_canvas.get_frame()
        self.general_sect = Section(self.main_frame, "General", 1)
        self.hill_orientation_label = Label(self.general_sect.content, "Hill  Orientation:", 1)
        self.hill_orientation = Entry(self.general_sect.content, 1)
        self.data_loader = FileLoader(
            self.general_sect.content,
            "Piv Data (no interp)",
            [("Pickle File", "*.pkl"), ("All Files", "*.*")],
            1,
            False,
        )
        self.data_loader_interp = FileLoader(
            self.general_sect.content,
            "Piv Data (interp)",
            [("Pickle File", "*.pkl"), ("All Files", "*.*")],
            1,
            False,
        )
        self.pose_loader = FileLoader(
            self.general_sect.content,
            "Pose File",
            [("Pose File", "*.json"), ("All Files", "*.*")],
            1,
            False,
        )
        self.diag_checkbox = Checkbutton(self.general_sect.content, 1, text="Plane is Diagonal")
        self.properties_loader = FileLoader(
            self.general_sect.content,
            "Fluid and Flow Properties",
            [("Properties File", "*.json"), ("All Files", "*.*")],
            1,
            False,
        )
        self.ref_conditions_loader = FileLoader(
            self.general_sect.content,
            "Reference Conditions",
            [("Reference Conditions File", "*.stat"), ("All Files", "*.*")],
            1,
            False,
        )
        self.pressure_sect = Section(self.main_frame, "Pressure Data", 2)
        self.port_loader = FileLoader(
            self.pressure_sect.content,
            "Port Wall Pressure",
            [("Port Wall Pressure File", "*.stat"), ("All Files", "*.*")],
            2,
            False,
        )
        self.hill_loader = FileLoader(
            self.pressure_sect.content,
            "Hill Surface Pressure",
            [("Hill Surface Pressure File", "*.stat"), ("All Files", "*.*")],
            2,
            False,
        )
        self.info_loader = FileLoader(
            self.pressure_sect.content,
            "Pressure Data Info File",
            [("Pressure Data Info File", "*.stat"), ("All Files", "*.*")],
            2,
            False,
        )
        self.profiler_sect = Section(self.main_frame, "Profiles", 1)
        self.num_prof_label = Label(self.profiler_sect.content, "Number of Profiles:", 1)
        self.num_prof = Entry(self.profiler_sect.content, 1)
        self.num_prof_pts_label = Label(self.profiler_sect.content, "Number of Profile Points:", 1)
        self.num_prof_pts = Entry(self.profiler_sect.content, 1)
        self.prof_height_label = Label(self.profiler_sect.content, "Profile Height:", 1)
        self.prof_height = Entry(self.profiler_sect.content, 1)
        self.coord_selector_label = Label(self.profiler_sect.content, "Coordinate System:", 1)
        self.coord_selector_var = tk.StringVar()
        self.coord_selector_var.set("Tunnel")
        self.coord_selector = tk.OptionMenu(self.profiler_sect.content, self.coord_selector_var, "Tunnel", "Shear")
        self.coord_selector_var.trace("w", self._on_coord_selection)
        self.reconstr_checkbox = Checkbutton(
            self.profiler_sect.content, 1, text="Add Reconstruction Points", command=self._toggle_reconstruction
        )
        self.num_reconstr_pts_label = Label(self.profiler_sect.content, "Number of Reconstruction Points:", 1)
        self.num_reconstr_pts = Entry(self.profiler_sect.content, 1)
        self.cfd_checkbox = Checkbutton(
            self.profiler_sect.content, 1, text="Extract CFD Profiles (expensive)", command=self._toggle_cfd
        )
        self.fluent_case_loader = FileLoader(
            self.profiler_sect.content, "Fluent Case", [("Fluent Case", "*.cas"), ("All Files", ".*.")], 1, False
        )
        self.fluent_data_loader = FileLoader(
            self.profiler_sect.content, "Fluent Data", [("Fluent Data", "*.dat"), ("All Files", ".*.")], 1, False
        )
        self.calculate_button = Button(self.main_frame, "Submit", self.calculate)

    def _layout_widgets(self):
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.general_sect.grid(row=0, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew")
        self.general_sect.content.grid(columnspan=3)
        self.general_sect.content.grid_columnconfigure(0, weight=1)
        self.general_sect.content.grid_columnconfigure(1, weight=1)
        self.general_sect.content.grid_columnconfigure(2, weight=1)
        self.hill_orientation_label.grid(
            row=0, column=0, columnspan=3, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.hill_orientation.grid(
            row=0, column=1, columnspan=3, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.data_loader.grid(
            row=1, column=0, columnspan=3, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.data_loader_interp.grid(
            row=2, column=0, columnspan=3, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.pose_loader.grid(
            row=3, column=0, columnspan=3, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.diag_checkbox.grid(
            row=4, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.properties_loader.grid(
            row=5, column=0, columnspan=3, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.ref_conditions_loader.grid(
            row=6, column=0, columnspan=3, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.pressure_sect.grid(
            row=1, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.pressure_sect.content.grid(columnspan=3)
        self.pressure_sect.content.grid_columnconfigure(0, weight=1)
        self.pressure_sect.content.grid_columnconfigure(1, weight=1)
        self.pressure_sect.content.grid_columnconfigure(2, weight=1)
        self.port_loader.grid(
            row=0, column=0, columnspan=3, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.hill_loader.grid(
            row=1, column=0, columnspan=3, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.info_loader.grid(
            row=2, column=0, columnspan=3, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.profiler_sect.grid(
            row=2, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.profiler_sect.content.grid(columnspan=3)
        self.profiler_sect.content.grid_columnconfigure(0, weight=1)
        self.profiler_sect.content.grid_columnconfigure(1, weight=1)
        self.profiler_sect.content.grid_columnconfigure(2, weight=1)
        self.num_prof_label.grid(
            row=0, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsw"
        )
        self.num_prof.grid(
            row=0, column=1, columnspan=2, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.num_prof_pts_label.grid(
            row=1, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsw"
        )
        self.num_prof_pts.grid(
            row=1, column=1, columnspan=2, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.prof_height_label.grid(
            row=2, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsw"
        )
        self.prof_height.grid(
            row=2, column=1, columnspan=2, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.coord_selector_label.grid(
            row=3, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsw"
        )
        self.coord_selector.grid(
            row=3, column=1, columnspan=2, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.reconstr_checkbox.grid(
            row=4, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.num_reconstr_pts_label.grid(
            row=5, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsw"
        )
        self.num_reconstr_pts.grid(
            row=5, column=1, columnspan=2, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.cfd_checkbox.grid(row=6, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew")
        self.fluent_case_loader.grid(
            row=7, column=0, columnspan=3, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.fluent_data_loader.grid(
            row=8, column=0, columnspan=3, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew"
        )
        self.calculate_button.grid(
            row=3, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="ns"
        )

    def _on_closing(self):
        self.root.destroy()

    def _toggle_reconstruction(self):
        pass

    def _on_coord_selection(self, *args):
        pass

    def _toggle_cfd(self):
        pass

    def calculate(self):
        """Extract profiles."""
        self.geometry = Beverli(orientation=float(self.hill_orientation.get()), use_cad=True)
        self.geometry.rotate(self.geometry.orientation)

        #TODO: add a check for the orientation and the shear mode

        pose_data = apputils.read_json(self.pose_loader.get_listbox_content()[0])
        if pose_data is None:
            sys.exit(-1)
        pose = Pose(
            angle1=(
                cast(float, pose_data["rotation"]["angle_1_deg"])
                if bool(self.diag_checkbox.get_var())
                else cast(float, pose_data["rotation"]["angle_deg"])
            ),
            angle2=(
                cast(float, pose_data["rotation"]["angle_2_deg"])
                if bool(self.diag_checkbox.get_var())
                else 0.0
            ),
            loc=[
                cast(float, pose_data["translation"]["x_1_loc_ref_mm"]),
                cast(float, pose_data["translation"]["x_2_loc_ref_mm"]),
            ],
            glob=[
                cast(float, pose_data["translation"]["x_1_glob_ref_m"]),
                cast(float, pose_data["translation"]["x_2_glob_ref_m"]),
                cast(float, pose_data["translation"]["x_3_glob_ref_m"]),
            ]
        )

        data_intrp = apputils.load_pickle(self.data_loader_interp.get_listbox_content()[0])
        data_no_intrp = apputils.load_pickle(self.data_loader.get_listbox_content()[0])

        self.piv_intrp = Piv(data=data_intrp, pose=pose)
        self.piv_no_intrp = Piv(data=data_no_intrp, pose=pose)
        try:
            profiles.extract_data(self.piv_no_intrp, self.piv_intrp, self.geometry, opts)
        except 


