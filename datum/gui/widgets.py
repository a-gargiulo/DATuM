"""Define program specific widget classes."""

import platform
import tkinter as tk
from typing import Any, Callable, Dict, Optional

from ...utility.configure import STYLES

is_mac = platform.system() == "Darwin"
if is_mac:
    from tkmacosx import Button as BaseButton
else:
    BaseButton = tk.Button


class Button(BaseButton):
    def __init__(self, parent: tk.Widget, text: str, command: Callable[[], None], **kwargs):
        super().__init__(
            parent,
            text=text,
            command=command,
            bg=STYLES["colors"]["accent"],
            fg=STYLES["colors"]["s1_content"],
            font=(STYLES["font"], STYLES["font_sizes"]["regular"], "bold"),
            **kwargs,
        )
        if is_mac:
            self.config(borderless=1)


class Checkbutton(tk.Checkbutton):
    def __init__(self, parent: tk.Widget, category: int, **kwargs):
        self.checkbox_var = tk.IntVar()
        super().__init__(
            parent, variable=self.checkbox_var, bg=STYLES["colors"][f"s{category}_content"], fg="white", **kwargs
        )

    def get_var(self) -> tk.IntVar:
        return self.checkbox_var


class Frame(tk.Frame):
    def __init__(self, parent: tk.Widget, category: int, **kwargs):
        super().__init__(parent, bg=STYLES["colors"][f"s{category}_content"], **kwargs)


class Label(tk.Label):
    def __init__(self, parent: tk.Widget, text: str, category: int, **kwargs):
        super().__init__(parent, text=text, bg=STYLES["colors"][f"s{category}_content"], fg="white", **kwargs)


class Entry(tk.Entry):
    def __init__(self, parent: tk.Widget, category: int, **kwargs):
        super().__init__(
            parent,
            bg="white",
            bd=1,
            relief="solid",
            highlightthickness=0,
            highlightbackground=STYLES["colors"][f"s{category}_content"],
            **kwargs,
        )


class Section(tk.Frame):
    def __init__(self, parent: tk.Frame, title: str, category: int, **kwargs: Dict[str, Any]):
        super().__init__(parent, bg=STYLES["colors"][f"s{category}_content"], bd=1, relief="solid", **kwargs)

        self.label = tk.Label(
            self,
            text=title,
            bg=STYLES["colors"][f"s{category}_header"],
            fg="white",
            bd=1,
            relief="solid",
        )
        self.content = tk.Frame(self, bd=0, bg=STYLES["colors"][f"s{category}_content"])

        self.label.grid(
            row=0,
            column=0,
            padx=STYLES["pad"]["small"],
            pady=STYLES["pad"]["small"],
            ipady=STYLES["pad"]["small"],
            sticky="ew",
        )
        self.content.grid(row=1, column=0, padx=STYLES["pad"]["small"], pady=STYLES["pad"]["small"], sticky="nsew")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)


class ScrollableCanvas:
    def __init__(
        self,
        parent: tk.Frame,
        vertical: bool = True,
        horizontal: bool = True,
        layout: Optional[Callable[[tk.Canvas, Optional[tk.Scrollbar], Optional[tk.Scrollbar]], None]] = None,
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
        self.canvas = tk.Canvas(self.parent, bg=STYLES["colors"]["base"], bd=0, highlightthickness=0, **canvas_kwargs)

        self.v_scrollbar = None
        self.h_scrollbar = None

        if vertical:
            self.v_scrollbar = tk.Scrollbar(
                self.parent, orient=tk.VERTICAL, command=self.canvas.yview, **scrollbar_kwargs
            )
            self.canvas.configure(yscrollcommand=self.v_scrollbar.set)

        if horizontal:
            self.h_scrollbar = tk.Scrollbar(
                self.parent, orient=tk.HORIZONTAL, command=self.canvas.xview, **scrollbar_kwargs
            )
            self.canvas.configure(xscrollcommand=self.h_scrollbar.set)

        if layout:
            layout(self.canvas, self.v_scrollbar, self.h_scrollbar)
        else:
            self._default_layout()

        self.frame = tk.Frame(self.canvas, bg=STYLES["colors"]["base"], **frame_kwargs)
        self.frame_window = self.canvas.create_window((0, 0), window=self.frame, anchor="nw")

        self.frame.bind("<Configure>", self.update_scroll_region)

        if vertical:
            self.canvas.bind("<MouseWheel>", self._on_vertical_scroll)
        if horizontal:
            self.canvas.bind("<Shift-MouseWheel", self._on_horizontal_scroll)

    def _default_layout(self):
        """Default layout if no custom layout is provided."""
        self.canvas.grid(row=0, column=0, sticky="nsew")
        if self.v_scrollbar:
            self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        if self.h_scrollbar:
            self.h_scrollbar.grid(row=1, column=0, sticky="ew")
        self.parent.grid_columnconfigure(0, weight=1)
        self.parent.grid_rowconfigure(0, weight=1)

    def _update_scroll_region(self, event=None):
        """Update the scroll region to include the entire canvas area."""
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def _on_vertical_scroll(self, event):
        """Scroll vertically based on mouse wheel."""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_horizontal_scroll(self, event):
        """Scroll horizontally when Shift + MouseWheel is used."""
        self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

    def configure_frame(self):
        """Update frame dimensions based on canvas and scrollbar sizes."""
        self.frame.update_idletasks()
        self.canvas.itemconfig(
            self.frame_window,
            width=self.canvas.winfo_width() - (self.v_scrollbar.winfo_width() if self.v_scrollbar else 0),
            height=self.frame.winfo_height() - (self.h_scrollbar.winfo_height() if self.h_scrollbar else 0),
        )
        self._update_scroll_region()

    def get_frame(self) -> tk.Frame:
        """
        Return the scrollable main frame.

        :return: The scrollable main frame.
        """
        return self.frame
