"""
Create a simple plot for the thesis introduction, showing how SPEX observes an incoming Stokes vector.
"""
from colorio._tools import plot_flat_gamut
from matplotlib import pyplot as plt
import numpy as np
from spectacle import spectral
from spectacle.general import gauss1d
from spectacle.plot import RGB_OkabeIto
from spex import stokes, elements
from groundspex import demodulation

# Variables
retardance_nm = 550*15

## Create the Stokes vector
# Load source spectrum
input_spectrum = np.loadtxt("input_data/cie_d65.txt", skiprows=1, usecols=[0,1])  # Daylight

# Create intensity vector
nr_samples = 1000
wavelengths = np.linspace(390, 700, nr_samples)
source_wavelengths = input_spectrum[:,0]
source_intensity = input_spectrum[:,1].copy()
source_intensity = np.interp(wavelengths, source_wavelengths, source_intensity, left=0, right=0)
source_intensity = gauss1d(source_intensity, sigma=5)
source_intensity = source_intensity / np.nanmax(source_intensity)

# Create DoLP/AoLP vectors
sigmoid = lambda x: 1 / (1 + np.exp(-x))
dolp = np.linspace(0, 1, len(wavelengths))
aolp = np.pi*sigmoid(np.linspace(-6, 6, len(wavelengths)))
aolp_deg = np.rad2deg(aolp) % 180

# Calculate modulated spectrum
S1, S2 = demodulation.modulation(wavelengths, source_intensity, dolp, aolp, retardance_mor=retardance_nm)

# Make plots
figsize = (3, 2)

# Plot input spectrum
fig, ax1 = plt.subplots(figsize=figsize)
ax1.plot(wavelengths, source_intensity, c='k', label="$I$")
ax1.plot(wavelengths, dolp, c=RGB_OkabeIto[1], label="$P_L$")
ax1.text(490, 0.90, "$I$", color='k', ha="right", va="top")
ax1.text(475, 0.4, "$P_L$", color=RGB_OkabeIto[1], ha="right", va="top")
ax1.set_xlabel("Wavelength [nm]")
ax1.set_xlim(wavelengths.min(), wavelengths.max())
ax1.set_xticks(np.arange(400,701,100))
ax1.set_ylabel("$I$, $P_L$")
ax1.set_ylim(0, 1.02)
ax1.set_yticks(np.arange(0,1.01,0.25))
ax1.grid(ls="--")

aolp_colour = RGB_OkabeIto[2]
ax2 = ax1.twinx()
ax2.plot(wavelengths, aolp_deg, c=aolp_colour, label=r"$\phi_L$")
ax2.text(540, 73, r"$\phi_L$", color=RGB_OkabeIto[2], ha="left", va="top")
ax2.set_ylabel("$\phi_L$ [$\degree$]", color=aolp_colour)
ax2.set_ylim(0, 1.02*180)
ax2.tick_params(axis="y", labelcolor=aolp_colour)
ax2.set_yticks(np.arange(0,181,45))

fig.tight_layout()
plt.savefig("spex_example_IPphi.pdf")
plt.show()
plt.close()

# Plot SPEX spectrum
fig, ax1 = plt.subplots(figsize=figsize)
ax1.plot(wavelengths, source_intensity, c='k', label="$I$")
ax1.plot(wavelengths, S1, c=RGB_OkabeIto[0], label="$S_1$")
ax1.plot(wavelengths, S2, c=RGB_OkabeIto[2], label="$S_2$")
ax1.text(565, 0.19, "$S_1$", color=RGB_OkabeIto[0], ha="right", va="top")
ax1.text(600, 0.13, "$S_2$", color=RGB_OkabeIto[2], ha="right", va="top")
ax1.text(580, 0.82, "$I = S_1 + S_2$", color='k', ha="left", va="bottom")
ax1.set_xlabel("Wavelength [nm]")
ax1.set_xlim(wavelengths.min(), wavelengths.max())
ax1.set_xticks(np.arange(400,701,100))
ax1.set_ylabel("Radiance")
ax1.set_ylim(0, 1.02)
ax1.set_yticks(np.arange(0,1.01,0.25))
ax1.grid(ls="--")
# ax1.legend(loc="best", framealpha=1, edgecolor='k')

fig.tight_layout()
plt.savefig("spex_example_SPEX.pdf")
plt.show()
plt.close()
