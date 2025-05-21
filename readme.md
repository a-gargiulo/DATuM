# Quick Start 

1) If you are running on an ARM64 machine, install an Intel-based Python version

First, install Rosetta, Intel Homebrew, and intel Python (with Tkinter (tk) support), if you have not done it yet. Intel homebrew is typically installed in `/usr/local/` as opposed to (regular) ARM64 homebrew, which is installed in `/opt`.

```bash
softwareupdate --install-rosetta
arch -x86_64 /bin/zsh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
/usr/local/bin/brew install python-tk@3.11
```

2) You can now open a regular ARM64 terminal session and don't need to worry about switching architecture. Create a virtual environment for DATuM and install DATUM and required packages by running the provided install.py.

```bash
/usr/local/bin/python3.11 -m venv .venv
source .venv/bin/activate
python install.py
```

3) when running the generate run.py script, do so through the tecplot environment setup script

```bash
/Applications/Tecplot\ 360\ EX\ 2021\ R1/bin/tec360-env -- python run.py
```
4) Deactivate virtual environment when done

```bash
deactivate
```
