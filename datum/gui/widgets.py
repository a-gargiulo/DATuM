"""Define app-specific widgets."""

import platform
import tkinter as tk
from tkinter import filedialog
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..utility.configure import STYLES

is_mac = platform.system() == "Darwin"
if is_mac:
    from tkmacosx import Button as BaseButton
else:
    BaseButton = tk.Button


# Custom types
LayoutFuncSC = Optional[
    Callable[
        [
            Union[tk.Tk, tk.Toplevel, tk.Frame],
            tk.Canvas,
            Optional[tk.Scrollbar],
            Optional[tk.Scrollbar],
        ],
        None,
    ]
]


class Button(BaseButton):
    """Define a custom `Button`."""

    def __init__(
        self,
        parent: Union[tk.Tk, tk.Toplevel, tk.Widget],
        text: str,
        command: Callable[[], None],
        **kwargs,
    ):
        """
        Class constructor.

        :param parent: The parent window or widget.
        :param text: The button label.
        :param command: The function/command to perform when the button is pressed.
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
    """Define a custom `Checkbutton`."""

    def __init__(self, parent: Union[tk.Tk, tk.Toplevel, tk.Widget], category: int, **kwargs):
        """
        Class constructor.

        :param parent: The parent window or widget.
        :param category: The style category applied to the `Checkbutton`.
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

    def get_var(self) -> tk.IntVar:
        """Get the checkbox variable."""
        return self.var


class Frame(tk.Frame):
    """Define a custom `Frame`."""

    def __init__(self, parent: Union[tk.Tk, tk.Toplevel, tk.Widget], category: int, **kwargs):
        """
        Class constructor.

        :param parent: The parent window or widget.
        :param category: The style category applied to the `Frame`.
        """
        super().__init__(parent, bg=STYLES["color"][f"s{category}_content"], **kwargs)


class Label(tk.Label):
    """Define a custom `Label`."""

    def __init__(
        self,
        parent: Union[tk.Tk, tk.Toplevel, tk.Widget],
        text: str,
        category: int,
        **kwargs,
    ):
        """
        Class constructor.

        :param parent: The parent window or widget.
        :param text: The label.
        :param category: The style category applied to the `Label`.
        """
        super().__init__(
            parent,
            text=text,
            bg=STYLES["color"][f"s{category}_content"],
            fg="white",
            **kwargs,
        )


class Entry(tk.Entry):
    """Define a custom `Entry`."""

    def __init__(self, parent: Union[tk.Tk, tk.Toplevel, tk.Widget], category: int, **kwargs):
        """
        Class constructor.

        :param parent: The parent window or widget.
        :param category: The style category applied to the `Entry`.
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


class Section(tk.Frame):
    """App-specific widget that creates a nicely formatted section."""

    def __init__(
        self,
        parent: Union[tk.Tk, tk.Toplevel, tk.Frame],
        title: str,
        category: int,
        **kwargs: Any,
    ):
        """
        Class constructor.

        :param parent: The parent window or frame.
        :param title: The title of the section.
        :param category: The style category applied to the `Section`.
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

    def get_content_frame(self) -> tk.Frame:
        """Get content frame."""
        return self.content


class ScrollableCanvas:
    """Obtain a scrollable frame."""

    def __init__(
        self,
        parent: Union[tk.Tk, tk.Toplevel, tk.Frame],
        vertical: bool = True,
        horizontal: bool = True,
        layout: LayoutFuncSC = None,
        canvas_kwargs: Optional[Dict[str, Any]] = None,
        scrollbar_kwargs: Optional[Dict[str, Any]] = None,
        frame_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a scrollable canvas with optional horizontal and vertical scrollbars.

        :param frame: The parent frame where the canvas and scrollbars are added.
        :param vertical: Whether to add a vertical scrollbar. Defaults to True.
        :param horizontal: Whether to add a horizontal scrollbar. Defaults to True.
        :param layout: A custom layout function to position the canvas and scrollbars.
        """
        canvas_kwargs = canvas_kwargs or {}
        scrollbar_kwargs = scrollbar_kwargs or {}
        frame_kwargs = frame_kwargs or {}

        self.parent = parent
        self.canvas = tk.Canvas(
            self.parent,
            bg=STYLES["color"]["base"],
            bd=0,
            highlightthickness=0,
            **canvas_kwargs,
        )

        self.v_scrollbar = None
        self.h_scrollbar = None

        if vertical:
            self.v_scrollbar = tk.Scrollbar(
                self.parent,
                orient=tk.VERTICAL,
                command=self.canvas.yview,
                **scrollbar_kwargs,
            )
            self.canvas.configure(yscrollcommand=self.v_scrollbar.set)

        if horizontal:
            self.h_scrollbar = tk.Scrollbar(
                self.parent,
                orient=tk.HORIZONTAL,
                command=self.canvas.xview,
                **scrollbar_kwargs,
            )
            self.canvas.configure(xscrollcommand=self.h_scrollbar.set)

        if layout:
            layout(self.parent, self.canvas, self.v_scrollbar, self.h_scrollbar)
        else:
            self._default_layout()

        self.frame = tk.Frame(self.canvas, bg=STYLES["color"]["base"], **frame_kwargs)
        self.frame_window = self.canvas.create_window((0, 0), window=self.frame, anchor="nw")
        self.frame.bind("<Configure>", self._update_scroll_region)

        if vertical:
            self.parent.bind("<MouseWheel>", self._on_vertical_scroll)
        if horizontal:
            self.parent.bind("<Shift-MouseWheel>", self._on_horizontal_scroll)

    def _default_layout(self):
        self.parent.grid_columnconfigure(0, weight=1)
        self.parent.grid_columnconfigure(1, weight=0)
        self.parent.grid_rowconfigure(0, weight=1)
        self.parent.grid_rowconfigure(1, weight=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        # self.canvas.grid_columnconfigure(0, weight=1)
        # self.canvas.grid_rowconfigure(0, weight=1)
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
        """Update frame dimensions based on canvas and scrollbar sizes."""
        self.frame.update_idletasks()

        width = self.frame.winfo_reqwidth()
        height = self.frame.winfo_reqheight()

        if self.v_scrollbar and not self.h_scrollbar:
            width = self.parent.winfo_width() - self.v_scrollbar.winfo_width()
            height = max(height, self.parent.winfo_height())
        elif not self.v_scrollbar and self.h_scrollbar:
            width = max(width, self.parent.winfo_width())
            height = self.parent.winfo_height() - self.h_scrollbar.winfo_height()
        elif self.v_scrollbar and self.h_scrollbar:
            width = max(width, self.parent.winfo_width() - self.v_scrollbar.winfo_height())
            height = max(height, self.parent.winfo_height() - self.h_scrollbar.winfo_height())
        else:
            width = self.parent.winfo_width()
            height = self.parent.winfo_height()

        self.canvas.itemconfig(
            self.frame_window,
            width=width,
            height=height,
        )

    def get_frame(self) -> tk.Frame:
        """
        Return the scrollable main frame.

        :return: The scrollable main frame.
        """
        return self.frame


class FileLoader(tk.Frame):
    """A widget providing a file loading context."""

    def __init__(
        self,
        parent: Union[tk.Tk, tk.Toplevel, tk.Frame],
        title: str,
        filetypes: List[Tuple[str, str]],
        category: int,
        isCheckable: bool = True,
        **kwargs: Any,
    ):
        """
        Class constructor.

        :param parent: The parent window or frame.
        :param title: The label.
        :param filetypes: The allowed file types.
        :param category: The style category applied to the `Section`.
        :param isCheckable: Boolean indicating whether the file loader should be linked to a checkbox.
        """
        self.parent = parent
        self.title = title
        self.filetypes = filetypes
        self.category = category
        self.isCheckable = isCheckable

        super().__init__(parent, bg=STYLES["color"][f"s{category}_content"], **kwargs)

        self._create_widgets()
        self._layout_widgets()

    def _create_widgets(self):
        self.checkbox = Checkbutton(self, 2)
        self.checkbox_var = self.checkbox.get_var()
        self.checkbox.config(command=self._toggle_state)
        self.load_button = Button(
            self,
            self.title,
            self._load_files,
            state="disabled" if self.isCheckable else "normal",
        )
        self.load_button.config(width=200 if is_mac else 20)
        self.listbox = tk.Listbox(
            self,
            state="disabled" if self.isCheckable else "normal",
            width=20,
            height=1,
        )
        self.status_label_var = tk.StringVar()
        self.status_label_var.set("Nothing Loaded")
        self.status_label = Label(
            self,
            "",
            self.category,
            state="disabled" if self.isCheckable else "normal",
            textvariable=self.status_label_var,
        )
        self.status_label.config(width=13, fg="red")

    def _layout_widgets(self):
        idx = 1
        if self.isCheckable:
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
        if self.isCheckable:
            self.grid_columnconfigure(3, weight=1)

    def _load_files(self):
        file_path = filedialog.askopenfilename(filetypes=self.filetypes)

        self.listbox.delete(0, tk.END)

        if file_path:
            self.listbox.insert(tk.END, file_path)
            self.status_label_var.set("File Loaded")
            # self.status_label.config(text="File Loaded", fg="green")
            self.status_label.config(fg="green")
        else:
            # self.status_label.config(text="Nothing Loaded", fg="red")
            self.status_label_var.set("Nothing Loaded")
            self.status_label.config(fg="red")

    def _toggle_state(self):
        state = "normal" if self.checkbox_var.get() else "disabled"
        self.load_button.config(state=state)
        self.listbox.config(state=state)
        self.status_label.config(state=state)

    def get_listbox(self) -> tk.Listbox:
        """
        Get the path listbox handle.

        :return: The listbox handle.
        :rtype: tk.Listbox
        """
        return self.listbox

    def get_listbox_content(self) -> str:
        """
        Get current listbox content.

        :return: Current listbox content.
        :rtype: str
        """
        return self.listbox.get(0, tk.END)[0]

    def get_checkbox(self) -> Checkbutton:
        """
        Get the checkbox handle.

        :return: Checkbox handle.
        :rtype: Checkbutton
        """
        return self.checkbox

    def get_checkbox_var(self) -> tk.IntVar:
        """
        Get the checkbox variable.

        :return: Checkbox variable.
        :rtype: tk.IntVar
        """
        return self.checkbox_var

    def get_load_button(self) -> Button:
        """
        Get the button handle.

        :return: Load button handle.
        :rtype: Button
        """
        return self.load_button

    def get_status_label(self) -> Label:
        """
        Get the status label handle.

        :return: Status label handle.
        :rtype: Label
        """
        return self.status_label

    def get_status_label_var(self) -> tk.StringVar:
        """
        Get the status label variable.

        :return: Status label variable.
        :rtype: tk.StringVar
        """
        return self.status_label_var

    def reset(self):
        """Reset the file loader."""
        self.listbox.delete(0, tk.END)
        self.status_label_var.set("Nothing Loaded")
        self.status_label.config(fg="red")
