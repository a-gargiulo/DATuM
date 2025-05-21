# Quick Start 

## Requirements
1. Tecplot
2. All other required packages will be installed automatically via install.py

## MacOS
1. On ARM64 machines (Apple Silicon), install and run an Intel-based Python version using Rosetta and Homebrew.

Install Rosetta and Homebrew (if not done already) + Intel-based Python:
```bash
softwareupdate --install-rosetta
arch -x86_64 /bin/zsh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
/usr/local/bin/brew install python-tk@3.11
```

2. Create a virtual environment for DATuM, activate the environment, and install DATuM via install.py.

```bash
/usr/local/bin/python3.11 -m venv .venv
source .venv/bin/activate
python install.py
```

3. To run DATuM (requires Tecplot) use: 

```bash
/Applications/Tecplot\ 360\ EX\ 2021\ R1/bin/tec360-env -- python run.py
```
4. Deactivate the virtual environment when done.

```bash
deactivate
```

## On Windows 
Using your favorite terminal, e.g., PowerShell or CMD

1. Create a virtual environment for DATuM, activate the environment, and install DATuM via install.py.

```powershell
python -m venv .venv
.venv\bin\activate
python install.py
```

2. Run DATuM.
```powershell
python run.py
```
