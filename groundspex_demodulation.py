"""
Demodulation pipeline for groundSPEX data
"""
from pathlib import Path
from sys import argv
import numpy as np
from matplotlib import pyplot as plt
from spectacle.general import gauss1d
from spectacle.plot import RGB_OkabeIto as RGB
from scipy.optimize import curve_fit
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

# Demodulation

def calculate_psi(wavelengths, aolp, retardance):
    """
    Calculate the modulation phase as a function of wavelength.
    """
    return 2*np.pi*retardance/wavelengths + 2*aolp


def modulation(wavelengths, intensity, dolp, aolp, retardance_mor=groundspex.MOR_RETARDANCE_NOMINAL):
    """
    From the unpolarised intensity and the degree and angle of linear polarisation, generate the modulated spectra in both channels of a dual-channel spectorpolarimeter.
    retardance_mor is the retardance of the multi-order retarder in the same units as wavelengths (not normalised!)
    All quantities are assumed to be wavelength-dependent.
    """
    psi = calculate_psi(wavelengths, aolp, retardance_mor)  # Helper variable: phase
    Splus = 0.5*intensity*(1 + dolp*np.cos(psi))
    Smin = 0.5*intensity*(1 - dolp*np.cos(psi))

    return Splus, Smin


def modulation_normalised(wavelengths, dolp, aolp, retardance_mor=groundspex.MOR_RETARDANCE_NOMINAL):
    """
    From the degree and angle of linear polarisation, generate the modulation fraction (I+ - I-)/(I+ + I-).
    """
    psi = calculate_psi(wavelengths, aolp, retardance_mor)  # Helper variable: phase
    fraction = dolp*np.cos(psi)

    return fraction


def modulation_to_fit(wavelengths, dolp, aolp, offset, retardance_mor=groundspex.MOR_RETARDANCE_NOMINAL):
    """
    Variation of modulation_normalised with an additional offset parameter, and with a fixed retardance.
    """
    psi = calculate_psi(wavelengths, aolp, retardance_mor)  # Helper variable: phase
    fraction = offset + 0.5*dolp*np.cos(psi)

    return fraction


def spectral_resolution_polarimetric(wavelengths, retardance_mor=groundspex.MOR_RETARDANCE_NOMINAL):
    """
    Calculate the polarimetric spectral resolution, which is roughly equal to the local modulation period.
    Make sure that `wavelengths` and `retardance_mor` are in the same units, for example nm. The result will also be in those units.
    """
    return wavelengths**2 / retardance_mor


# Crop wavelength range to 420 -- 850 nm
crop = np.where((wavelengths >= 420) & (wavelengths <= 850))[0]
wavelengths = wavelengths[crop]
data = data[...,crop]

# Single iteration for now
data_demod = data[outlier].copy()
Stokes_I = np.nansum(data_demod, axis=0)
data_normalised = data_demod / Stokes_I
data_difference = np.squeeze(np.diff(data_demod, axis=0))  # Written in this roundabout way because it will generalise more easily later
data_fraction = data_difference/Stokes_I

# Loop over wavelengths
spectral_resolutions = spectral_resolution_polarimetric(wavelengths)
spectral_windows = [np.where((wavelengths >= l-r*0.5) & (wavelengths <= l+r*0.5))[0] for l, r in zip(wavelengths, spectral_resolutions)]
popt, pcov = zip(*[curve_fit(modulation_to_fit, wavelengths[ind], data_normalised[1,ind], p0=[0.1, 0.1, 0.5, groundspex.MOR_RETARDANCE_NOMINAL], bounds=([0., 0., -1., groundspex.MOR_RETARDANCE_NOMINAL*0.9], [1., 2*np.pi, 1., groundspex.MOR_RETARDANCE_NOMINAL*1.1])) for ind in spectral_windows])
popt = np.array(popt)
pcov = np.array(pcov)
dolp, aolp, offsets, retardance_fit = popt.T

# Smooth polarisation
dolp_smooth = gauss1d(dolp, sigma=25)
aolp_smooth = gauss1d(aolp, sigma=25)

# Plot and save
fig, ax1 = plt.subplots()
ax1.plot(wavelengths, Stokes_I, c='k')
ax1.set_xlabel("Wavelength [nm]")
ax1.set_ylabel("Radiance [ADU]")
ax1.set_xlim(wavelengths[0], wavelengths[-1])
ax1.set_ylim(ymin=0)
ax1.set_title(data_folder.stem)

ax2 = ax1.twinx()
ax2.plot(wavelengths, 100*dolp_smooth, c=RGB[1])
ax2.set_ylabel("Degree of Linear Polarisation [%]", color=RGB[1])
ax2.set_ylim(0, 30)

plt.savefig(f"results/{data_folder.stem}_demodulated_example.png", dpi=300)
plt.close()
print("Saved demodulated plot")
