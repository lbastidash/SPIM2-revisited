# SPIM2 Light Sheet Microscope Control Code

## Introduction

This repository contains Python code for controlling the SPIM2 Light Sheet Microscope at Uniandes. The code leverages dependencies and functions defined in `auxiliarySPIM2.py` and `slmpy.py` to facilitate communication with the microscope hardware and perform various tasks. Additionally, the code relies on pycromanager for microscopy control functionalities.

## Dependencies

- **pycromanager**: Python interface for controlling microscopy hardware, including the SPIM2 Light Sheet Microscope.
- **auxiliarySPIM2.py**: Contains auxiliary functions and classes necessary for the operation of other programs in the repository.
- **slmpy.py**: Provides functions for controlling the spatial light modulator (SLM) component of the SPIM2 microscope.
- **AOtools**: Adaptive optics library from M. J. Townson, O. J. D. Farley, G. O. de Xivry, J. Osborn, and A. P. Reeves, “AOtools: a Python package for adaptive optics modeling and analysis,” Opt. Express, OE, vol. 27, no. 22, pp. 31316–31329, Oct. 2019, doi: 10.1364/OE.27.031316. 


## Programs

1. **autofocusSPIM2.py**:
   - Description: Runs an Autofocus algorithm to align the illumination plane with the SPIM2 focal plane.
   - Dependencies: `pycromanager`, `auxiliarySPIM2.py`

2. **slmCenteringWithGraphsSPIM2.py**:
   - Description: Projects a blur phase mask in different coordinates of the SLM to find the best position to center the phase mask and graphs the results.
   - Dependencies: `pycromanager`, `slmpy.py`, `auxiliarySPIM2.py`
     
2. **slm Aberration Correction**:


## Usage

To use the code in this repository, follow these steps:

1. Install the required dependencies (`pycromanager`, etc.) using pip or your preferred package manager.
2. Clone this repository to your local machine.
3. Open and run the desired Python script (e.g., `autofocusSPIM2.py` or `slmCenteringWithGraphsSPIM2.py`).
4. Follow the on-screen instructions or refer to the comments within the code for guidance on usage.

## License

This code is provided under the [MIT License](LICENSE).
