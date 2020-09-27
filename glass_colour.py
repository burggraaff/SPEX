import numpy as np
from matplotlib import pyplot as plt
from spex import stokes, elements
from spectacle.general import gauss1d, gaussMd
from spectacle import spectral

# Load the input spectrum
input_spectrum = np.loadtxt("input_data/lamp_spectrum.txt", skiprows=14)

# Create the source spectrum, fully unpolarised
wavelength_step = 0.3
wavelengths = np.arange(385, 800, wavelength_step)
source_wavelengths = input_spectrum[:,0]
source_intensity = gauss1d(input_spectrum[:,1], sigma=5)
source_intensity = source_intensity / np.nanmax(source_intensity)
source_intensity = np.interp(wavelengths, source_wavelengths, source_intensity)
source = stokes.Stokes_nm(source_intensity, 0, 0, 0)

# Create the input and output polarisers at 0 and 90 degrees, respectively
polariser_0 = elements.Linear_polarizer_general(0.9, 0.005, 0)
polariser_90 = elements.Linear_polarizer_general(0.9, 0.005, 90)

# Propagate the source light through the input polariser
after_polariser_0 = np.einsum("ij,wi->wj", polariser_0, source)

# Retardances to loop over
retardances_relative = np.linspace(0, 5, 250)
retardances_absolute = retardances_relative * 560.

# Integral of RGB intensities
integral_SRF = np.zeros((3,len(retardances_relative)))

# Create foils and propagate through them
foils = np.stack([elements.Retarder_wavelengths(retardance_absolute, 45, wavelengths) for retardance_absolute in retardances_absolute])
after_foils = np.einsum("rwij,wi->rwj", foils, after_polariser_0)

# Propagate through the output polariser
after_polariser_90 = np.einsum("ij,rwj->rwi", polariser_90, after_foils)

# Get only the total intensity spectra
intensities = after_polariser_90[...,0]

# Interpolate the intensity spectra to the CIE XYZ wavelengths
intensity_interpolated = spectral.interpolate_spectral_data(wavelengths, intensities, spectral.cie_wavelengths)

# Convert the intensity spectra to CIE XYZ
intensity_XYZ = np.einsum("xw,rw->xr", spectral.cie_xyz, intensity_interpolated) / len(spectral.cie_wavelengths)

# Plot XYZ as a function of retardance
for XYZ, label in zip(intensity_XYZ, "XYZ"):
    plt.plot(retardances_relative, XYZ, label=label, lw=3)
plt.xlabel("Retardance in $\lambda$ at 560 nm")
plt.ylabel("CIE XYZ")
plt.legend(loc="best")
plt.savefig("retardance_XYZ.pdf", bbox_inches="tight")
plt.show()
plt.close()

# Convert to xy chromaticity
intensity_xy = (intensity_XYZ / intensity_XYZ.sum(axis=0))[:2]

# Plot xy as a function of retardance
for xy, label in zip(intensity_xy, "xy"):
    plt.plot(retardances_relative, xy, label=label, lw=3)
plt.xlabel("Retardance in $\lambda$ at 560 nm")
plt.ylabel("CIE xy (normalised)")
plt.legend(loc="best")
plt.savefig("retardance_xy.pdf", bbox_inches="tight")
plt.show()
plt.close()
