import platform

system = platform.system()

# Buttons
if system == "Darwin":
    from tkmacosx import Button
else:
    from tkinter import Button

# Fonts
if system == "Darwin":
    default_font = ("Avenir", 14)
elif system == "Windows":
    default_font = ("Franklin Gothic Book", 14)

colors = {
    "base": "#262626",
    "f1_header": "#2a262f",
    "f1_content": "#413d46",
    "f2_header": "#1e1e1e",
    "f2_content": "#373737",
    "accent": "#bb86fc",
}
