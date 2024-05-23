# Direct Analyzer of Turbulence Models

This repository houses an analyzer utilizing particle image velocimetry (PIV) and supplementary laser Doppler velocimetry (LDV) and computational fluid dynamics (CFD) data to directly investigate constitutive relations for turbulent aerodynamic flows.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Structure of data folder 

VERY IMPORTANT:

1) The PIV data root folder MUST be structured as indicated below. 

2) The raw data files (.mat) must contain the following suffixes.
 - mean_velocity
 - reynolds_stress
 - instantaneous_frame
 - turbulence_dissipation

3) Calibration image, coordinate transformation parameters, and pose measurement files must be included in the corresponding piv plane's folder.

Data Root Folder
    ├── plane1
    │   ├── 250k_FS
    │   │   ├── plane1_250k_FS_mean_velocity.mat
    │   │   ├── plane1_250k_FS_reynolds_stress.mat
    │   │   ├── plane1_250k_FS_instantaneous_frame.mat
    │   │   └── plane1_250k_FS_turbulence_dissipation.mat
    │   ├── 250k_SS
    │   │   └── ...
    │   ├── 650k_FS
    │   │   └── ...
    │   ├── calibration_image.dat
    │   ├── global_mapping.txt
    │   └── position_measurements.txt
    ├── plane2
    │   ├── 250k_FS
    │  ...  └── ...
   ...


Important notes:
- Each plane folder must also contain a calibration image, a mapping file, and position measurement file. The naming of these files can be specified in the input file.

- The raw data files (.mat) can be named as you like. They should, however, contain the following suffixes
 - mean_velocity
 - reynolds_stress
 - instantaneous_frame
 - turbulence_dissipation


## Installation

- Clone project from Github

- Create and activate a virtual environment (recommended)
```
bash

python -m venv venv
source venv/bin/activate
```

- Install required packages
```
bash

pip install matplotlib
pip install scipy
pip install "trimesh[all]"
pip install plotly
pip install pandas
```

A full list of all the dependencies that these packages install
is provided in requirements.txt.

## Pre-requisites
An input.txt file must exist and be placed in the same folder of main.py.

### Structure of input.txt
...

### PIV data root folder structure and file naming convention
Program handles OS specific path conventions automatically
./Plane
meanVel
reStress
instFrame

Explain the available options for x3 locations!


## Modules
Json
Numpy
Pyglet
Trimesh

## Example Files
To run the example files
