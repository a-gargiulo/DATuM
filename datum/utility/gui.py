import tkinter as tk
from typing import Callable, Optional, Tuple


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


def create_section(
    frame: tk.Frame,
    title: str,
    position: dict,
    content_column_span: int = 1,
    section_kwargs: Optional[dict] = None,
    section_title_kwargs: Optional[dict] = None,
    section_content_kwargs: Optional[dict] = None,
) -> Tuple[tk.Frame, tk.Frame]:

    section_kwargs = section_kwargs or {}
    section_title_kwargs = section_title_kwargs or {}

    section_frame = tk.Frame(frame, bd=1, relief="solid", **section_kwargs)

    section_title = tk.Label(section_frame, text=title, bd=1, relief="solid", **section_title_kwargs)

    section_content_frame = tk.Frame(frame, bd=0, **section_content_kwargs)

    # Layout
    section_frame.grid(**position)
    section_frame.grid_columnconfigure(0, weight=1)
    section_frame.grid_rowconfigure(1, weight=1)

    section_title.grid(row=0, column=0, padx=5, pady=5, ipady=5, sticky="ew")

    section_content_frame.grid(row=1, column=0, columnspan=content_column_span, padx=5, pady=5, sticky="nsew")

    return section_frame, section_content_frame
