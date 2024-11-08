import platform
import tkinter as tk
from typing import Any, Callable, Dict, Optional, Tuple

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
        frame: tk.Frame,
        vertical: bool = True,
        horizontal: bool = True,
        layout: Optional[Callable[[tk.Canvas, Optional[tk.Scrollbar], Optional[tk.Scrollbar]], None]] = None,
        canvas_kwargs: Optional[Dict[str, Any]] = None,
        scrollbar_kwargs: Optional[Dict[str, Any]] = None,
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

        self.frame = frame
        self.canvas = tk.Canvas(self.frame, bd=0, highlightthickness=0, **canvas_kwargs)

        self.v_scrollbar = None
        self.h_scrollbar = None

        if vertical:
            self.v_scrollbar = tk.Scrollbar(
                self.frame, orient=tk.VERTICAL, command=self.canvas.yview, **scrollbar_kwargs
            )
            self.canvas.configure(yscrollcommand=self.v_scrollbar.set)

        if horizontal:
            self.h_scrollbar = tk.Scrollbar(
                self.frame, orient=tk.HORIZONTAL, command=self.canvas.xview, **scrollbar_kwargs
            )
            self.canvas.configure(xscrollcommand=self.h_scrollbar.set)

        if layout:
            layout(self.canvas, self.v_scrollbar, self.h_scrollbar)
        else:
            self.default_layout()

    def default_layout(self):
        """Default layout if no custom layout is provided."""
        self.canvas.grid(row=0, column=0, sticky="nsew")
        if self.v_scrollbar:
            self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        if self.h_scrollbar:
            self.h_scrollbar.grid(row=1, column=0, sticky="ew")
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_rowconfigure(0, weight=1)

    def get_widgets(self) -> Tuple[tk.Canvas, Optional[tk.Scrollbar], Optional[tk.Scrollbar]]:
        """
        Return the canvas and scrollbar widgets.
        :return: Tuple containing the canvas, vertical scrollbar (if any), and horizontal scrollbar (if any).
        """
        return self.canvas, self.v_scrollbar, self.h_scrollbar
