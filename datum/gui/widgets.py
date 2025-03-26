"""Create user-defined widgets."""

import platform
import tkinter as tk
from tkinter import filedialog
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..utility.configure import STYLES

is_mac = platform.system() == "Darwin"

BaseButton = tk.Button
if is_mac:
    try:
        from tkmacosx import Button as MacButton

        BaseButton = MacButton
    except ImportError:
        print("Warning: tkmacosx not installed. Using tk.Button instead.")


# Custom types
Parent = Union[tk.Tk, tk.Toplevel, tk.Widget]
Command = Callable[[], None]
LayoutFuncSC = Optional[
    Callable[
        [
            Parent,
            tk.Canvas,
            Optional[tk.Scrollbar],
            Optional[tk.Scrollbar],
        ],
        None,
    ]
]


class Button(BaseButton):
    """Define a custom button."""

    def __init__(self, parent: Parent, text: str, command: Command, **kwargs):
        """Construct a user-defined button.

        :param parent: Parent window or widget.
        :param text: Button label.
        :param command: Callback function to be executed when the button is pressed.
        """
        super().__init__(
            parent,
            text=text,
            command=command,
            bg=STYLES["color"]["accent"],
            fg=STYLES["color"]["s1_content"],
            font=(STYLES["font"], STYLES["font_size"]["regular"], "bold"),
            **kwargs,
        )
        if is_mac:
            self.config(borderless=1)


class Checkbutton(tk.Checkbutton):
    """Define a custom checkbutton."""

    def __init__(self, parent: Parent, category: int, **kwargs):
        """Construct a user-defined checkbutton.

        :param parent: Parent window or widget.
        :param category: Style category to be applied to the checkbutton.
        """
        self.var = tk.IntVar()
        super().__init__(
            parent,
            variable=self.var,
            bg=STYLES["color"][f"s{category}_content"],
            fg="white",
            selectcolor="gray",
            anchor="w",
            **kwargs,
        )


class Entry(tk.Entry):
    """Define a custom entry."""

    def __init__(self, parent: Parent, category: int, **kwargs):
        """Construct a user-defined entry field.

        :param parent: Parent window or widget.
        :param category: Style category to be applied to the entry field.
        """
        super().__init__(
            parent,
            bg="white",
            bd=1,
            relief="solid",
            highlightthickness=0,
            highlightbackground=STYLES["color"][f"s{category}_content"],
            **kwargs,
        )


class FileLoader(tk.Frame):
    """Custom file loading context."""

    def __init__(
        self,
        parent: Parent,
        title: str,
        filetypes: List[Tuple[str, str]],
        category: int,
        isCheckable: bool = True,
        **kwargs,
    ):
        """Construct a user-defined file loading context.

        :param parent: Parent window or frame.
        :param title: Title label.
        :param filetypes: List of allowed file types.
        :param category: Style category to be applied to the file loader.
        :param isCheckable: Whether the file loader should be linked to a checkbox.
            Defaults to True.
        """
        self._parent = parent
        self._title = title
        self._filetypes = filetypes
        self._category = category
        self._isCheckable = isCheckable

        super().__init__(parent, bg=STYLES["color"][f"s{category}_content"], **kwargs)

        self._create_widgets()
        self._layout_widgets()

    def _create_widgets(self):
        self.checkbox = Checkbutton(self, 2)
        self.checkbox_var = self.checkbox.var
        self.checkbox.config(command=self._toggle_state)
        self.load_button = Button(
            self,
            self._title,
            self._load_files,
            state="disabled" if self._isCheckable else "normal",
        )
        self.load_button.config(width=200 if is_mac else 20)
        self.listbox = tk.Listbox(
            self,
            state="disabled" if self._isCheckable else "normal",
            width=20,
            height=1,
        )
        self.status_label_var = tk.StringVar()
        self.status_label_var.set("Nothing Loaded")
        self.status_label = Label(
            self,
            "",
            self._category,
            state="disabled" if self._isCheckable else "normal",
            textvariable=self.status_label_var,
        )
        self.status_label.config(width=13, fg="red")

    def _layout_widgets(self):
        idx = 1
        if self._isCheckable:
            idx = 0
            self.checkbox.grid(
                row=0,
                column=0,
                sticky="w",
                padx=STYLES["pad"]["small"],
            )

        self.load_button.grid(
            row=0,
            column=1 - idx,
            padx=STYLES["pad"]["small"],
            pady=STYLES["pad"]["small"],
            sticky="nsew",
        )
        self.listbox.grid(
            row=0,
            column=2 - idx,
            padx=STYLES["pad"]["small"],
            pady=STYLES["pad"]["small"],
            sticky="nsew",
        )
        self.status_label.grid(
            row=0,
            column=3 - idx,
            padx=STYLES["pad"]["small"],
            pady=STYLES["pad"]["small"],
            sticky="nsew",
        )
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        if self._isCheckable:
            self.grid_columnconfigure(3, weight=1)

    def _load_files(self):
        file_path = filedialog.askopenfilename(filetypes=self._filetypes)

        self.listbox.delete(0, tk.END)

        if file_path:
            self.listbox.insert(tk.END, file_path)
            self.status_label_var.set("File Loaded")
            self.status_label.config(fg="green")
        else:
            self.status_label_var.set("Nothing Loaded")
            self.status_label.config(fg="red")

    def _toggle_state(self):
        state = "normal" if self.checkbox_var.get() else "disabled"
        self.load_button.config(state=state)
        self.listbox.config(state=state)
        self.status_label.config(state=state)

    def get_listbox_content(self) -> str:
        """Get the current listbox content.

        :return: Current listbox content.
        :rtype: str
        """
        return self.listbox.get(0, tk.END)[0]

    def reset(self):
        """Reset the file loader."""
        self.listbox.delete(0, tk.END)
        self.status_label_var.set("Nothing Loaded")
        self.status_label.config(fg="red")


class Frame(tk.Frame):
    """Define a custom frame."""

    def __init__(self, parent: Parent, category: int, **kwargs):
        """Construct a user-defined frame.

        :param parent: Parent window or widget.
        :param category: Style category to be applied to the frame.
        """
        super().__init__(parent, bg=STYLES["color"][f"s{category}_content"], **kwargs)


class Label(tk.Label):
    """Define a custom label."""

    def __init__(self, parent: Parent, text: str, category: int, **kwargs):
        """Construct a user-defined label.

        :param parent: Parent window or widget.
        :param text: Label text.
        :param category: Style category to be applied to the label.
        """
        super().__init__(
            parent,
            text=text,
            bg=STYLES["color"][f"s{category}_content"],
            fg="white",
            **kwargs,
        )


class ScrollableCanvas:
    """Define a scrollable frame."""

    def __init__(
        self,
        parent: Parent,
        vertical: bool = True,
        horizontal: bool = True,
        layout: LayoutFuncSC = None,
        canvas_kwargs: Optional[Dict[str, Any]] = None,
        scrollbar_kwargs: Optional[Dict[str, Any]] = None,
        frame_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Construct a scrollable canvas with optional scrollbars.

        :param frame: Parent frame to which the canvas and scrollbars are added.
        :param vertical: Whether to add a vertical scrollbar. Defaults to True.
        :param horizontal: Whether to add a horizontal scrollbar. Defaults to True.
        :param layout: A custom layout function to position the canvas and scrollbars.
        :param canvas_kwargs: Optional canvas keyword arguments.
        :param scrollbar_kwargs: Optional scrollbar keyword arguments.
        :param frame_kwargs: Optional frame keyword arguments.
        """
        canvas_kwargs = canvas_kwargs or {}
        scrollbar_kwargs = scrollbar_kwargs or {}
        frame_kwargs = frame_kwargs or {}

        self._parent = parent

        self.canvas = tk.Canvas(
            self._parent,
            bg=STYLES["color"]["base"],
            bd=0,
            highlightthickness=0,
            **canvas_kwargs,
        )

        self.v_scrollbar = None
        self.h_scrollbar = None
        if vertical:
            self.v_scrollbar = tk.Scrollbar(
                self._parent,
                orient=tk.VERTICAL,
                command=self.canvas.yview,
                **scrollbar_kwargs,
            )
            self.canvas.configure(yscrollcommand=self.v_scrollbar.set)
        if horizontal:
            self.h_scrollbar = tk.Scrollbar(
                self._parent,
                orient=tk.HORIZONTAL,
                command=self.canvas.xview,
                **scrollbar_kwargs,
            )
            self.canvas.configure(xscrollcommand=self.h_scrollbar.set)

        if layout:
            layout(self._parent, self.canvas, self.v_scrollbar, self.h_scrollbar)
        else:
            self._default_layout()

        self.frame = tk.Frame(self.canvas, bg=STYLES["color"]["base"], **frame_kwargs)
        self._frame_window = self.canvas.create_window(
            (0, 0), window=self.frame, anchor="nw"
        )
        self.frame.bind("<Configure>", self._update_scroll_region)

        if vertical:
            self._parent.bind("<MouseWheel>", self._on_vertical_scroll)
        if horizontal:
            self._parent.bind("<Shift-MouseWheel>", self._on_horizontal_scroll)

    def _default_layout(self):
        self._parent.grid_columnconfigure(0, weight=1)
        self._parent.grid_columnconfigure(1, weight=0)
        self._parent.grid_rowconfigure(0, weight=1)
        self._parent.grid_rowconfigure(1, weight=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        if self.v_scrollbar:
            self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        if self.h_scrollbar:
            self.h_scrollbar.grid(row=1, column=0, sticky="ew")

    def _update_scroll_region(self, event=None):
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def _on_vertical_scroll(self, event):
        self.canvas.yview_scroll(-1 * event.delta, "units")

    def _on_horizontal_scroll(self, event):
        self.canvas.xview_scroll(-1 * event.delta, "units")

    def configure_frame(self):
        """Update the frame's dimensions based on the canvas and scrollbar sizes."""
        self.frame.update_idletasks()

        width = self.frame.winfo_reqwidth()
        height = self.frame.winfo_reqheight()

        if self.v_scrollbar and not self.h_scrollbar:
            width = self._parent.winfo_width() - self.v_scrollbar.winfo_width()
            height = max(height, self._parent.winfo_height())
        elif not self.v_scrollbar and self.h_scrollbar:
            width = max(width, self._parent.winfo_width())
            height = self._parent.winfo_height() - self.h_scrollbar.winfo_height()
        elif self.v_scrollbar and self.h_scrollbar:
            width = max(
                width, self._parent.winfo_width() - self.v_scrollbar.winfo_height()
            )
            height = max(
                height, self._parent.winfo_height() - self.h_scrollbar.winfo_height()
            )
        else:
            width = self._parent.winfo_width()
            height = self._parent.winfo_height()

        self.canvas.itemconfig(
            self._frame_window,
            width=width,
            height=height,
        )


class Section(tk.Frame):
    """Custom formatted section."""

    def __init__(self, parent: Parent, title: str, category: int, **kwargs):
        """Construct a nicely formatted user-defined section.

        :param parent: Parent window or frame.
        :param title: Title of the section.
        :param category: Style category to be applied to the section.
        """
        super().__init__(
            parent,
            bg=STYLES["color"][f"s{category}_content"],
            bd=1,
            relief="solid",
            **kwargs,
        )

        self.label = tk.Label(
            self,
            text=title,
            bg=STYLES["color"][f"s{category}_header"],
            fg="white",
            bd=1,
            relief="solid",
        )
        self.content = tk.Frame(self, bd=0, bg=STYLES["color"][f"s{category}_content"])

        self.label.grid(
            row=0,
            column=0,
            padx=STYLES["pad"]["small"],
            pady=STYLES["pad"]["small"],
            ipady=STYLES["pad"]["small"],
            sticky="ew",
        )
        self.content.grid(
            row=1,
            column=0,
            padx=STYLES["pad"]["small"],
            pady=STYLES["pad"]["small"],
            sticky="nsew",
        )
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
