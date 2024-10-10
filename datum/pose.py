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
