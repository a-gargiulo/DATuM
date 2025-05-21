# Table of Contents

- [Quick Start](#quick-start)
    - [Prerequisites](#prerequisites)
    - [Install and Run](#install-and-run)
        - [On Windows](#on-windows)
        - [On MacOS](#on-macos-arm64---apple-silicon)

# Quick Start 

## Prerequisites 
1. Tecplot
2. Any other required package will be installed automatically via the provided `install.py` script.

## Install and Run

### On Windows
Using a terminal (for example, PowerShell or CMD)

1. Create a `python` virtual environment and activate. Install DATuM via the provided `install.py`.

```powershell
python -m venv .venv
.venv\Scripts\activate
python install.py
```

2. Run DATuM.
```powershell
python run.py
```

3. Deactivate the virtual environment when done.

```bash
deactivate
```

### On MacOS (ARM64 - Apple Silicon)
1. Install and run an Intel-based Python version and Rosetta + Homebrew (if not installed already).

```bash
softwareupdate --install-rosetta
arch -x86_64 /bin/zsh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
/usr/local/bin/brew install python-tk@3.11
```

2. Create a `python` virtual environment and activate. Install DATuM via the provided `install.py`.

```bash
/usr/local/bin/python3.11 -m venv .venv
source .venv/bin/activate
python install.py
```

3. Run DATuM.

```bash
/Applications/Tecplot\ 360\ EX\ 2021\ R1/bin/tec360-env -- python run.py
```
4. Deactivate the virtual environment when done.

```bash
deactivate
```
