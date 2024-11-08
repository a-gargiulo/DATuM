import tkinter as tk


from .utility.configure import system, colors, default_font
from .utility import gui

if system == "Darwin":
    from tkmacosx import Button
elif system == "Windows":
    from tkinter import Button

W_WIDTH = 600
W_HEIGHT = 600


class Pose:
    def __init__(self, master: tk.Tk):
        self.root = tk.Toplevel(master)
        self.root.title("Pose")
        self.root.geometry("600x600")
        self.root.resizable(False, False)
        self.root.configure(bg=colors["base"])
        self.root.option_add("*Font", default_font)
        # self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.root_canvas, self.scrollbar, _ = gui.create_scrollable_canvas(
            self.root, True, False, None, {"bg": colors["base"]}, None
        )
        self.main_frame = tk.Frame(self.root_canvas, bg=colors["base"])
        main_frame_window_kwargs = {"window": self.main_frame, "anchor": "nw"}
        self.main_frame_window = self.root_canvas.create_window((0, 0), **main_frame_window_kwargs)

        self.root_canvas.bind_all("<MouseWheel>", self.on_vertical)

        self.local_pose_section, self.local_pose_content = gui.create_section(
            frame=self.main_frame,
            title="Local Pose",
            position={"row": 0, "column": 0, "columnspan": 1, "padx": 5, "pady": 5, "sticky": "nsew"},
            content_columnspan=1,
            section_kwargs={"bg": colors["s1_content"]},
            section_title_kwargs={"bg": colors["s1_header"], "fg": "white"},
            section_content_kwargs={"bg": colors["s1_content"]},
        )


        self.adjust_layout()

        self.main_frame.update_idletasks()
        self.root_canvas.itemconfig(
            self.main_frame_window,
            width=W_WIDTH - self.scrollbar.winfo_width(),
            height=self.main_frame.winfo_height(),
        )
        self.root_canvas.config(scrollregion=self.root_canvas.bbox("all"))


    def adjust_layout(self):
        self.main_frame.grid_columnconfigure(0, weight=1)


    def on_vertical(self, event):
        self.root_canvas.yview_scroll(-1 * event.delta, "units")

        # self.create_file_loader(self.piv_frame, "Calibration Image", 2, "normal", self.load_calibration)
        # self.status_label_calibration_image.config(bg=colors["f2_content"])


    # def load_calibration(self, data_type, section_frame):
    #     # Open file dialog and get the selected file paths
    #     file_path = filedialog.askopenfilename(
    #         filetypes=[("Data Files", "*.dat"), ("All Files", "*.*")]
    #     )

    #     if file_path:
    #         # Get the corresponding listbox and status label for this data type
    #         listbox = getattr(self, f"listbox_{data_type.lower().replace(' ', '_')}")
    #         status_label = getattr(self, f"status_label_{data_type.lower().replace(' ', '_')}")

    #         # Clear previous items in the listbox
    #         listbox.delete(0, tk.END)

    #         # Add new file paths to the listbox
    #         listbox.insert(tk.END, file_path)

    #         status_label.config(text="File Loaded", fg="green")
    #     else:
    #         listbox = getattr(self, f"listbox_{data_type.lower().replace(' ', '_')}")
    #         status_label = getattr(self, f"status_label_{data_type.lower().replace(' ', '_')}")
    #         listbox.delete(0, tk.END)
    #         listbox.insert(tk.END, file_path)
    #         status_label.config(text="Nothing Loaded", fg="red")
