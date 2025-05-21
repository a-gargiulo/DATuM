import os
import platform
import subprocess
import sys

system = platform.system()

def install_dependencies():
    print("Installing dependencies...")
    if system == "Darwin":
        reqs = "datum/resources/requirements_mac.txt"
    else:
        reqs = "datum/resources/requirements.txt"
    subprocess.check_call(["pip", "install", "-r", reqs])

def create_run_py():
    run_py_content = """import tkinter as tk
import datum

if __name__ == "__main__":
    root = tk.Tk()
    app = datum.DatumWindow(root)
    root.mainloop()
"""
    with open("run.py", "w") as f:
        f.write(run_py_content)
    print("run.py has been created.")

def clean():
    if os.path.exists("run.py"):
        os.remove("run.py")
        print("Removed run.py.")

    print("Uninstalling dependencies...")
    try:
        if system == "Darwin":
            reqs = "datum/resources/requirements_mac.txt"
        else:
            reqs = "datum/resources/requirements.txt"
        subprocess.check_call(["pip", "uninstall", "-r", reqs, "-y"])
    except subprocess.CalledProcessError as e:
        print(f"Error during uninstallation: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        clean()
    else:
        install_dependencies()
        if not os.path.exists("run.py"):
            create_run_py()
