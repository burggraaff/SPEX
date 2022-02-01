"""
Demodulation pipeline for groundSPEX data
"""

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from sys import argv
import groundspex as G

texp = 200.

# Get data folder from command-line input
data_folder = Path(argv[1])
print(f"Loading data from {data_folder.absolute()}")

# Load spectra
data, data_dark, data_timestamps = G.load_data_folder(data_folder)
print("Data loaded")

# Dark subtraction
data = G.correct_darkcurrent(data, data_dark)
print("Dark subtraction done")

# Wavelength calibration
wavelengths, data = G.correct_wavelengths(data)
print("Wavelength correction done")

# Differential transmission correction
data = G.correct_transmission(data)
print("Transmission correction done")

# Efficiency correction

# Demodulation


# Smooth polarisation

# Save
