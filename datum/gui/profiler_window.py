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
from ..core import profiles  # Deactivate to run without errors
from typing import cast, Optional


# Constants
WINDOW_TITLE = "Profiler"
WINDOW_SIZE = (600, 600)
PAD_S = STYLES["pad"]["small"]


class ProfilerWindow:
    """Generate the GUI for the profiler window and link it to the core functions."""

    def __init__(self, master: tk.Tk):
        """Initialize GUI."""
        self.root = tk.Toplevel(master)
        self.configure_root()
        self.create_widgets()
        self.layout_widgets()
        self.scrollable_canvas.configure_frame()

    def configure_root(self):
        """Configure main window settings."""
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}")
        self.root.resizable(False, False)
        self.root.configure(bg=STYLES["color"]["base"])
        self.root.option_add("*Font", (STYLES["font"], STYLES["font_size"]["regular"]))
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        """Create all widget entities."""
        self.scrollable_canvas = ScrollableCanvas(self.root, True, False)
        self.main_frame = self.scrollable_canvas.get_frame()
        self.general_section = Section(self.main_frame, title="General", category=1)
        self.hill_orientation_label = Label(
            self.general_section.get_content_frame(), text="Hill Orientation:", category=1
        )
        self.hill_orientation = Entry(self.general_section.get_content_frame(), category=1)
        self.data_loader = FileLoader(
            self.general_section.get_content_frame(),
            title="Piv Data (no interp)",
            filetypes=[("Pickle File", "*.pkl"), ("All Files", "*.*")],
            category=1,
            isCheckable=False,
        )
        self.data_loader_interpolation = FileLoader(
            self.general_section.get_content_frame(),
            title="Piv Data (interp)",
            filetypes=[("Pickle File", "*.pkl"), ("All Files", "*.*")],
            category=1,
            isCheckable=False,
        )
        self.pose_loader = FileLoader(
            self.general_section.get_content_frame(),
            title="Pose File",
            filetypes=[("Pose File", "*.json"), ("All Files", "*.*")],
            category=1,
            isCheckable=False,
        )
        self.checkbox_diagonal = Checkbutton(
            self.general_section.get_content_frame(), category=1, text="Plane is Diagonal"
        )
        self.properties_loader = FileLoader(
            self.general_section.get_content_frame(),
            title="Fluid and Flow Properties",
            filetypes=[("Properties File", "*.json"), ("All Files", "*.*")],
            category=1,
            isCheckable=False,
        )
        self.ref_conditions_loader = FileLoader(
            self.general_section.get_content_frame(),
            title="Reference Conditions",
            filetypes=[("Reference Conditions File", "*.stat"), ("All Files", "*.*")],
            category=1,
            isCheckable=False,
        )
        self.pressure_section = Section(self.main_frame, title="Pressure Data", category=2)
        self.port_loader = FileLoader(
            self.pressure_section.get_content_frame(),
            title="Port Wall Pressure",
            filetypes=[("Port Wall Pressure File", "*.stat"), ("All Files", "*.*")],
            category=2,
            isCheckable=False,
        )
        self.hill_loader = FileLoader(
            self.pressure_section.get_content_frame(),
            title="Hill Surface Pressure",
            filetypes=[("Hill Surface Pressure File", "*.stat"), ("All Files", "*.*")],
            category=2,
            isCheckable=False,
        )
        self.info_loader = FileLoader(
            self.pressure_section.get_content_frame(),
            title="Pressure Data Info File",
            filetypes=[("Pressure Data Info File", "*.stat"), ("All Files", "*.*")],
            category=2,
            isCheckable=False,
        )
        self.profiler_section = Section(self.main_frame, title="Profiles", category=1)
        self.num_profile_label = Label(
            self.profiler_section.get_content_frame(), text="Number of Profiles:", category=1
        )
        self.num_profile = Entry(self.profiler_section.get_content_frame(), category=1)
        self.num_profile_pts_label = Label(
            self.profiler_section.get_content_frame(), text="Number of Profile Points:", category=1
        )
        self.num_profile_pts = Entry(self.profiler_section.get_content_frame(), category=1)
        self.profile_height_label = Label(self.profiler_section.get_content_frame(), text="Profile Height:", category=1)
        self.profile_height = Entry(self.profiler_section.get_content_frame(), category=1)
        self.coordinates_selector_label = Label(
            self.profiler_section.get_content_frame(), text="Coordinate System:", category=1
        )
        self.coordinates_selector_var = tk.StringVar()
        self.coordinates_selector_var.set("Tunnel")
        self.coordinates_selector = tk.OptionMenu(
            self.profiler_section.get_content_frame(), self.coordinates_selector_var, "Tunnel", "Shear"
        )
        self.coordinates_selector_var.trace("w", self.on_coordinates_selection)
        self.reconstruction_checkbox = Checkbutton(
            self.profiler_section.get_content_frame(),
            category=1,
            text="Add Reconstruction Points",
            command=self.toggle_reconstruction
        )
        self.num_reconstruction_pts_label = Label(
            self.profiler_section.get_content_frame(), text="Number of Reconstruction Points:", category=1
        )
        self.num_reconstruction_pts = Entry(self.profiler_section.get_content_frame(), category=1)
        self.checkbox_cfd = Checkbutton(
            self.profiler_section.get_content_frame(),
            category=1,
            text="Extract CFD Profiles (expensive)",
            command=self.toggle_cfd
        )
        self.fluent_case_loader = FileLoader(
            self.profiler_section.get_content_frame(),
            title="Fluent Case",
            filetypes=[("Fluent Case", "*.cas"), ("All Files", ".*.")],
            category=1,
            isCheckable=False
        )
        self.fluent_data_loader = FileLoader(
            self.profiler_section.get_content_frame(),
            title="Fluent Data",
            filetypes=[("Fluent Data", "*.dat"), ("All Files", ".*.")],
            category=1,
            isCheckable=False
        )
        self.calculate_button = Button(self.main_frame, text="Submit", command=self.calculate)

    def layout_widgets(self):
        """Layout all widgets on the window."""
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.general_section.grid(row=0, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.general_section.get_content_frame().grid(columnspan=3)
        self.general_section.get_content_frame().grid_columnconfigure(0, weight=1)
        self.general_section.get_content_frame().grid_columnconfigure(1, weight=1)
        self.general_section.get_content_frame().grid_columnconfigure(2, weight=1)
        self.hill_orientation_label.grid(row=0, column=0, padx=(2 * PAD_S, 0), pady=PAD_S, sticky="nsw")
        self.hill_orientation.grid(row=0, column=1, columnspan=2, padx=0, pady=PAD_S, sticky="nsew")
        self.data_loader.grid(row=1, column=0, columnspan=3, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.data_loader_interpolation.grid(row=2, column=0, columnspan=3, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.pose_loader.grid(row=3, column=0, columnspan=3, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.checkbox_diagonal.grid(row=4, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.properties_loader.grid(row=5, column=0, columnspan=3, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.ref_conditions_loader.grid(row=6, column=0, columnspan=3, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.pressure_section.grid(row=1, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.pressure_section.get_content_frame().grid(columnspan=3)
        self.pressure_section.get_content_frame().grid_columnconfigure(0, weight=1)
        self.pressure_section.get_content_frame().grid_columnconfigure(1, weight=1)
        self.pressure_section.get_content_frame().grid_columnconfigure(2, weight=1)
        self.port_loader.grid(row=0, column=0, columnspan=3, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.hill_loader.grid(row=1, column=0, columnspan=3, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.info_loader.grid(row=2, column=0, columnspan=3, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.profiler_section.grid(row=2, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.profiler_section.get_content_frame().grid(columnspan=3)
        self.profiler_section.get_content_frame().grid_columnconfigure(0, weight=1)
        self.profiler_section.get_content_frame().grid_columnconfigure(1, weight=1)
        self.profiler_section.get_content_frame().grid_columnconfigure(2, weight=1)
        self.num_profile_label.grid(row=0, column=0, padx=PAD_S, pady=PAD_S, sticky="nsw")
        self.num_profile.grid(row=0, column=1, columnspan=2, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.num_profile_pts_label.grid(row=1, column=0, padx=PAD_S, pady=PAD_S, sticky="nsw")
        self.num_profile_pts.grid(row=1, column=1, columnspan=2, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.profile_height_label.grid(row=2, column=0, padx=PAD_S, pady=PAD_S, sticky="nsw")
        self.profile_height.grid(row=2, column=1, columnspan=2, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.coordinates_selector_label.grid(row=3, column=0, padx=PAD_S, pady=PAD_S, sticky="nsw")
        self.coordinates_selector.grid(row=3, column=1, columnspan=2, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.reconstruction_checkbox.grid(row=4, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.num_reconstruction_pts_label.grid(row=5, column=0, padx=PAD_S, pady=PAD_S, sticky="nsw")
        self.num_reconstruction_pts.grid(row=5, column=1, columnspan=2, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.checkbox_cfd.grid(row=6, column=0, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.fluent_case_loader.grid(row=7, column=0, columnspan=3, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.fluent_data_loader.grid(row=8, column=0, columnspan=3, padx=PAD_S, pady=PAD_S, sticky="nsew")
        self.calculate_button.grid(row=3, column=0, padx=PAD_S, pady=PAD_S, sticky="ns")

    def on_closing(self):
        """Free resources and clean up when closing the window."""
        self.root.destroy()

    def toggle_reconstruction(self):
        """Activate/deactivate the profile reconstruction option."""
        pass

    def on_coordinates_selection(self, *args):
        """Perform an action when the coordinate system is selected."""
        pass

    def toggle_cfd(self):
        """Activate/deactivate the option to extract equivalent profiles from CFD data."""
        pass

    def load_pose(self, pose_path: str, is_diagonal: bool) -> Optional[Pose]:
        """
        Load the pose of the PIV plane from the specified file.

        :param pose_path: Filepath.
        :param is_diagonal: Boolean indicating whether the PIV plane is diagonal.

        :return: A pose object.
        :rtype: Pose
        """
        pose_data = apputils.read_json(pose_path)
        if pose_data is None:
            return None
        pose = Pose(
            angle1=(
                float(cast(dict, pose_data["rotation"])["angle_1_deg"])
                if is_diagonal
                else float(cast(dict, pose_data["rotation"])["angle_deg"])
            ),
            angle2=(
                float(cast(dict, pose_data["rotation"])["angle_2_deg"])
                if is_diagonal
                else 0.0
            ),
            loc=[
                float(cast(dict, pose_data["translation"])["x_1_loc_ref_mm"]),
                float(cast(dict, pose_data["translation"])["x_2_loc_ref_mm"]),
            ],
            glob=[
                float(cast(dict, pose_data["translation"])["x_1_glob_ref_m"]),
                float(cast(dict, pose_data["translation"])["x_2_glob_ref_m"]),
                float(cast(dict, pose_data["translation"])["x_3_glob_ref_m"]),
            ]
        )
        return pose


    def calculate(self):
        """Extract 1D profile data from 2D PIV planes."""
        self.geometry = Beverli(orientation=float(self.hill_orientation.get()), use_cad=True)

        # TODO: add a check for the orientation and the shear mode

        pose = self.load_pose(self.pose_loader.get_listbox_content(), bool(self.checkbox_diagonal.get_var()))
        if pose is None:
            sys.exit(-1)

        data_intrp = apputils.load_pickle(self.data_loader_interpolation.get_listbox_content())
        data_no_intrp = apputils.load_pickle(self.data_loader.get_listbox_content())
        # TODO: check if the data was properly loaded.
        self.piv_intrp = Piv(data=data_intrp, pose=pose)
        self.piv_no_intrp = Piv(data=data_no_intrp, pose=pose)
        pass
        # profiles.extract_data(self.piv_no_intrp, self.piv_intrp, self.geometry, opts)
