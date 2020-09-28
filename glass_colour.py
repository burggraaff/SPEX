import numpy as np
from matplotlib import pyplot as plt
from spex import stokes, elements
from spectacle.general import gauss1d, gaussMd
from spectacle import spectral
from colorio._tools import plot_flat_gamut

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

# Convert to sRGB
M_rgb_to_xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                         [0.2126729, 0.7151522, 0.0721750],
                         [0.0193339, 0.1191920, 0.9503041]])

M_xyz_to_rgb = np.linalg.inv(M_rgb_to_xyz)

intensity_XYZ_normalised = intensity_XYZ * 7
intensity_sRGB = M_xyz_to_rgb @ intensity_XYZ_normalised

# Plot sRGB as a function of retardance
for sRGB, label in zip(intensity_sRGB, "rgb"):
    plt.plot(retardances_relative, sRGB, label=label, lw=3, c=label)
plt.xlabel("Retardance in $\lambda$ at 560 nm")
plt.ylabel("sRGB")
plt.legend(loc="best")
plt.savefig("retardance_sRGB.pdf", bbox_inches="tight")
plt.show()
plt.close()

# Clip sRGB and plot as a function of retardance
intensity_sRGB_clip = np.clip(intensity_sRGB, 0, 1)
for sRGB, label in zip(intensity_sRGB_clip, "rgb"):
    plt.plot(retardances_relative, sRGB, label=label, lw=3, c=label)
plt.xlabel("Retardance in $\lambda$ at 560 nm")
plt.ylabel("sRGB")
plt.legend(loc="best")
plt.savefig("retardance_sRGB_clipped.pdf", bbox_inches="tight")
plt.show()
plt.close()

for i, (retardance_relative, retardance_absolute, intensity, xy, sRGB) in enumerate(zip(retardances_relative, retardances_absolute, intensities, intensity_xy.T, intensity_sRGB_clip.T)):
    # Status indicator
    label = f"Retardance: {retardance_absolute:5.0f} nm ; {retardance_relative:4.1f} $\lambda$"

    # Make plots
    fig, axs = plt.subplots(ncols=3, figsize=(8,3))

    # First panel: intensity spectrum
    axs[0].plot(wavelengths, intensity)
    axs[0].set_xlim(380, 720)
    axs[0].set_ylim(-0.02, 0.5)
    axs[0].set_xlabel("Wavelength [nm]")
    axs[0].set_ylabel("Intensity [a.u.]")
    axs[0].set_title(f"Retardance:\n{retardance_absolute:5.0f} nm    {retardance_relative:4.1f} $\lambda$")

    # Second panel: gamut
    plt.sca(axs[1])
    plot_flat_gamut(plot_planckian_locus=False, axes_labels=("", ""))
    plt.plot(*intensity_xy[:,:i+1], c='k')
    plt.scatter(*xy, c='k')
    axs[1].set_title(f"$(x, y) = ({xy[0]:.2f}, {xy[1]:.2f})$")

    # Third panel: colour
    rect = plt.Rectangle((0, 0), 1, 1, facecolor=sRGB)
    axs[2].add_patch(rect)
    axs[2].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    axs[2].set_aspect("equal")
    axs[2].set_title(f"$(r, g, b) = ({sRGB[0]:.2f}, {sRGB[1]:.2f}, {sRGB[2]:.2f})$")


    plt.savefig(f"animation/gamut_{retardance_absolute:07.2f}.png")
    plt.show()
    plt.close()

    print(label)
