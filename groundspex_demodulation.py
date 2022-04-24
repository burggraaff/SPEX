"""
Demodulation pipeline for groundSPEX data
"""
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from spectacle.plot import RGB_OkabeIto as RGB
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
plt.plot(data[...,0,:].T, c="k", alpha=0.1, rasterized=True)
plt.xlabel("Spectral pixel")
plt.ylabel("Radiance [ADU]")
plt.title(data_folder.stem)
plt.grid(ls="--")
plt.savefig(f"results/{data_folder.stem}_allspectra_raw.png", dpi=300)
plt.close()
print("Saved raw data plot")

# Dark subtraction
data = groundspex.data_processing.correct_darkcurrent(data, data_dark)
print("Dark subtraction done")

# Wavelength calibration
wavelengths, data = groundspex.data_processing.correct_wavelengths(data)
print("Wavelength correction done")

# Differential transmission correction
data = groundspex.data_processing.correct_transmission(data)
print("Transmission correction done")

# Plot the corrected data
plt.plot(wavelengths, data[...,0,:].T, c="k", alpha=0.1, rasterized=True)
plt.xlim(wavelengths[0], wavelengths[-1])
plt.xlabel("Wavelength [nm]")
plt.ylabel("Radiance [ADU]")
plt.title(data_folder.stem)
plt.grid(ls="--")
plt.savefig(f"results/{data_folder.stem}_allspectra_corrected.png", dpi=300)
plt.close()
print("Saved corrected data plot")

# Plot the median and outliers
outlier = data[...,0,2000].argmax()
median = np.nanmedian(data, axis=(0,1))
plt.plot(wavelengths, median, c="k")
for channel, colour in zip(data[outlier], RGB[::2]):
    plt.plot(wavelengths, channel, c=colour)
plt.xlim(wavelengths[0], wavelengths[-1])
plt.xlabel("Wavelength [nm]")
plt.ylabel("Radiance [ADU]")
plt.title(data_folder.stem)
plt.grid(ls="--")
plt.savefig(f"results/{data_folder.stem}_outlier.png", dpi=300)
plt.close()
print("Saved outlier plot")


# Efficiency correction

# Demodulation


# Smooth polarisation

# Save
