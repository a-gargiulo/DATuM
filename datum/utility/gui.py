import tkinter as tk
from typing import Any, Dict, Callable, Optional, Tuple

from .configure import default_font, default_colors


def create_scrollable_canvas(
    frame: tk.Frame,
    vertical: bool = True,
    horizontal: bool = True,
    layout: Optional[Callable[[tk.Canvas, Optional[tk.Scrollbar], Optional[tk.Scrollbar]], None]] = None,
    canvas_kwargs: Optional[dict] = None,
    scrollbar_kwargs: Optional[dict] = None,
) -> Tuple[tk.Canvas, Optional[tk.Scrollbar], Optional[tk.Scrollbar]]:
    """
    Create a scrollable canvas with optional horizontal and vertical scrollbars.

    :param frame: The parent frame where the canvas and scrollbars are added.
    :param vertical: Whether to add a vertical scrollbar. Defaults to True.
    :param horizontal: Whether to add a horizontal scrollbar. Defaults to True.
    :param layout: A custom layout function to position the canvas and scrollbars.

    :return: Tuple of the canvas and the created scrollbars (or None if not created).
    """
    canvas_kwargs = canvas_kwargs or {}
    scrollbar_kwargs = scrollbar_kwargs or {}

    canvas = tk.Canvas(frame, bd=0, highlightthickness=0, **canvas_kwargs)

    v_scrollbar = None
    h_scrollbar = None

    if vertical:
        v_scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview, **scrollbar_kwargs)
        canvas.configure(yscrollcommand=v_scrollbar.set)

    if horizontal:
        h_scrollbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL, command=canvas.xview, **scrollbar_kwargs)
        canvas.configure(xscrollcommand=h_scrollbar.set)

    if layout:
        layout(canvas, v_scrollbar, h_scrollbar)
    else:
        canvas.grid(row=0, column=0, sticky="nsew")
        if vertical:
            v_scrollbar.grid(row=0, column=1, sticky="ns")
        if horizontal:
            h_scrollbar.grid(row=1, column=0, sticky="ew")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=1)

    return canvas, v_scrollbar, h_scrollbar

