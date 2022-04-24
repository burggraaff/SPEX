"""
Demodulation pipeline for groundSPEX data
"""
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from sys import argv
import groundspex

texp = 200.

# Get data folder from command-line input
data_folder = Path(argv[1])
print(f"Loading data from {data_folder.absolute()}")

# Load spectra
data, data_dark, data_timestamps = groundspex.load_data_folder(data_folder)
print("Data loaded")

# Plot the raw data
plt.plot(data[0,...].T, c="k", alpha=0.1, rasterized=True)
plt.xlabel("Spectral pixel")
plt.ylabel("Radiance [ADU]")
plt.title(data_folder.stem)
plt.grid(ls="--")
plt.savefig(f"results/{data_folder.stem}_allspectra.png", dpi=300)
plt.close()
print("Saved raw data plot")

# Plot the median and outliers
plt.plot(np.nanmedian(data[0], axis=0), c="k", rasterized=True)
plt.plot(data[0,data[0,:,2000].argmax()], c="b", rasterized=True)
plt.xlabel("Spectral pixel")
plt.ylabel("Radiance [ADU]")
plt.title(data_folder.stem)
plt.grid(ls="--")
plt.savefig(f"results/{data_folder.stem}_outlier.png", dpi=300)
plt.close()
print("Saved outlier plot")

# Dark subtraction
data = groundspex.data_processing.correct_darkcurrent(data, data_dark)
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
