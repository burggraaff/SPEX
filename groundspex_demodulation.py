"""
Demodulation pipeline for groundSPEX data
"""
from pathlib import Path
from sys import argv
import numpy as np
from matplotlib import pyplot as plt, patheffects as pe
from spectacle.general import gauss1d
from spectacle.plot import RGB_OkabeIto as RGB
import groundspex

texp = 200.

# Get data folder from command-line input
data_folder = Path(argv[1])
print(f"Loading data from {data_folder.absolute()}")

# Load spectra
data, data_dark, data_timestamps = groundspex.load_data_folder(data_folder)
print("Data loaded")

# Plot the raw data
groundspex.plot.plot_spectrum_stack_dualchannel(groundspex.instrument.PIXEL_ARRAY_DUALCHANNEL[0], data, xlabel="Pixel number", title=data_folder.stem, saveto=f"results/{data_folder.stem}_allspectra_raw.png")
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
groundspex.plot.plot_spectrum_stack_dualchannel(wavelengths, data, title=data_folder.stem, saveto=f"results/{data_folder.stem}_allspectra_corrected.png")
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

# Demodulation
# Crop wavelength range to 420 -- 850 nm
wavelengths, data = groundspex.demodulation.crop_spectra(wavelengths, data)

# Single iteration for now
data_demod = data[outlier].copy()
Stokes_I = np.nansum(data_demod, axis=0)
data_normalised = data_demod / Stokes_I
data_difference = np.squeeze(np.diff(data_demod, axis=0))  # Written in this roundabout way because it will generalise more easily later
data_fraction = data_difference/Stokes_I

# Loop over wavelengths
spectral_resolutions = groundspex.demodulation.spectral_resolution_polarimetric(wavelengths)
spectral_windows = [np.where((wavelengths >= l-r*0.5) & (wavelengths <= l+r*0.5))[0] for l, r in zip(wavelengths, spectral_resolutions)]
popt, pcov = zip(*[groundspex.demodulation.fit_modulation_single(wavelengths[ind], data_normalised[1,ind]) for ind in spectral_windows])
popt = np.array(popt)
pcov = np.array(pcov)
dolp, aolp, offsets, retardance_fit = popt.T
dolp_uncertainty, aolp_uncertainty, offsets_uncertainty, retardance_fit_uncertainty = np.sqrt(np.diagonal(pcov, axis1=-2, axis2=-1)).T

# Smooth polarisation
dolp_smooth = gauss1d(dolp, sigma=25)
dolp_uncertainty_smooth = gauss1d(dolp_uncertainty, sigma=25)
aolp_smooth = gauss1d(aolp, sigma=25)
aolp_uncertainty_smooth = gauss1d(aolp_uncertainty, sigma=25)
retardance_smooth = gauss1d(retardance_fit, sigma=100)
retardance_uncertainty_smooth = gauss1d(retardance_fit_uncertainty, sigma=100)

# Plot and save
fig, ax1 = plt.subplots(figsize=(5.3, 2.4), tight_layout=True)
ax1.plot(wavelengths, Stokes_I, c='k')
ax1.plot(wavelengths, data_demod[0], c=RGB[0])
ax1.plot(wavelengths, data_demod[1], c=RGB[2])
ax1.set_xlabel("Wavelength [nm]")
ax1.set_ylabel("Radiance [ADU]")
ax1.set_xlim(wavelengths[0], wavelengths[-1])
ax1.set_ylim(0, 10000)
ax1.set_yticks(np.arange(0,10001,2000))
ax1.set_title("No debris")
ax1.grid(ls="--")

ax2 = ax1.twinx()
ax2.plot(wavelengths, dolp_smooth, c=RGB[1], path_effects=[pe.withStroke(linewidth=3, foreground='black')])
ax2.fill_between(wavelengths, (dolp_smooth-dolp_uncertainty_smooth), (dolp_smooth+dolp_uncertainty_smooth), facecolor=RGB[1], alpha=0.5)
ax2.set_ylabel("Degree of Linear\nPolarisation $P_L$", color=RGB[1])
ax2.set_ylim(0, 0.25)
ax2.set_yticks(np.arange(0,0.26,0.05))

plt.savefig(f"results/{data_folder.stem}_demodulated_example.pdf")
plt.close()
print("Saved demodulated plot")
