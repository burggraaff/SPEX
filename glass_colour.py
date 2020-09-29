import numpy as np
from matplotlib import pyplot as plt
from spex import stokes, elements
from spectacle.general import gauss1d, gaussMd
from spectacle import spectral
from colorio._tools import plot_flat_gamut

nr_retardances = 250

# Load the input spectrum
# input_spectrum = np.loadtxt("input_data/lamp_spectrum.txt", skiprows=14)  # Halogen lamp
# input_spectrum = np.loadtxt("input_data/cie_d65.txt", skiprows=1, usecols=[0,1])  # Daylight
input_spectrum = np.stack([np.linspace(380, 800, 1500), np.ones(1500)]).T  # Equal-energy spectrum

# Create the source spectrum, fully unpolarised
wavelength_step = 0.3
wavelengths = np.arange(385, 800, wavelength_step)
source_wavelengths = input_spectrum[:,0]
source_intensity = input_spectrum[:,1].copy()
source_intensity = source_intensity / np.nanmax(source_intensity)
source_intensity = np.interp(wavelengths, source_wavelengths, source_intensity, left=0, right=0)
source_intensity = gauss1d(source_intensity, sigma=5)
source = stokes.Stokes_nm(source_intensity, 0, 0, 0)

# Create the input and output polarisers at 0 and 90 degrees, respectively
polariser_0 = elements.Linear_polarizer_general(0.9, 0.005, 0)
polariser_90 = elements.Linear_polarizer_general(0.9, 0.005, 90)

# Propagate the source light through the input polariser
after_polariser_0 = np.einsum("ij,wj->wi", polariser_0, source)

# Retardances to loop over
retardances_relative = np.linspace(0, 5, nr_retardances)
retardances_absolute = retardances_relative * 560.

# Integral of RGB intensities
integral_SRF = np.zeros((3,len(retardances_relative)))

# Create foils and propagate through them
foils = np.stack([elements.Retarder_wavelengths(retardance_absolute, 45, wavelengths) for retardance_absolute in retardances_absolute])
after_foils = np.einsum("rwij,wj->rwi", foils, after_polariser_0)

# Propagate through the output polarisers (orthogonal, parallel)
after_polariser_orthogonal = np.einsum("ij,rwj->rwi", polariser_90, after_foils)
after_polariser_parallel = np.einsum("ij,rwj->rwi", polariser_0, after_foils)
polariser_labels = ["Orthogonal", "Parallel"]

# Get only the total intensity spectra
intensities_orthogonal = after_polariser_orthogonal[...,0]
intensities_parallel = after_polariser_parallel[...,0]

# Interpolate the intensity spectra to the CIE XYZ wavelengths
intensity_orthogonal_interpolated = spectral.interpolate_spectral_data(wavelengths, intensities_orthogonal, spectral.cie_wavelengths)
intensity_parallel_interpolated = spectral.interpolate_spectral_data(wavelengths, intensities_parallel, spectral.cie_wavelengths)

# Convert the intensity spectra to CIE XYZ
intensity_orthogonal_XYZ = np.einsum("xw,rw->xr", spectral.cie_xyz, intensity_orthogonal_interpolated) / len(spectral.cie_wavelengths)
intensity_parallel_XYZ = np.einsum("xw,rw->xr", spectral.cie_xyz, intensity_parallel_interpolated) / len(spectral.cie_wavelengths)

# Plot XYZ as a function of retardance
fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True)
for ax, intensity_XYZ, pol_label in zip(axs, [intensity_orthogonal_XYZ, intensity_parallel_XYZ], polariser_labels):
    for XYZ, label in zip(intensity_XYZ, "XYZ"):
        ax.plot(retardances_relative, XYZ, label=label, lw=3)
    ax.set_ylabel(f"CIE XYZ ({pol_label})")
    ax.legend(ncol=3, loc="lower right")
    ax.grid(ls="--")
axs[1].set_xlabel("Retardance in $\lambda$ at 560 nm")
plt.savefig("retardance_XYZ.pdf", bbox_inches="tight")
plt.show()
plt.close()

# Convert to xy chromaticity
intensity_orthogonal_xy = (intensity_orthogonal_XYZ / intensity_orthogonal_XYZ.sum(axis=0))[:2]
intensity_parallel_xy = (intensity_parallel_XYZ / intensity_parallel_XYZ.sum(axis=0))[:2]

# Plot xy as a function of retardance
fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True)
for ax, intensity_xy, pol_label in zip(axs, [intensity_orthogonal_xy, intensity_parallel_xy], polariser_labels):
    for xy, label in zip(intensity_xy, "xy"):
        ax.plot(retardances_relative, xy, label=label, lw=3)
    ax.set_ylabel(f"CIE xy ({pol_label})")
    ax.grid(ls="--")
    ax.legend(ncol=2, loc="lower right")
axs[1].set_xlabel("Retardance in $\lambda$ at 560 nm")
plt.savefig("retardance_xy.pdf", bbox_inches="tight")
plt.show()
plt.close()

# Convert to sRGB
M_rgb_to_xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                         [0.2126729, 0.7151522, 0.0721750],
                         [0.0193339, 0.1191920, 0.9503041]])

M_xyz_to_rgb = np.linalg.inv(M_rgb_to_xyz)

normalisation = 7.  # Arbitrary, controls brightness of resulting colours
intensity_orthogonal_sRGB = M_xyz_to_rgb @ intensity_orthogonal_XYZ * normalisation
intensity_parallel_sRGB = M_xyz_to_rgb @ intensity_parallel_XYZ * normalisation

# Plot sRGB as a function of retardance
fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True)
for ax, intensity_sRGB, pol_label in zip(axs, [intensity_orthogonal_sRGB, intensity_parallel_sRGB], polariser_labels):
    for sRGB, label in zip(intensity_sRGB, "rgb"):
        ax.plot(retardances_relative, sRGB, label=label, lw=3, c=label)
    ax.set_ylabel(f"sRGB ({pol_label})")
    ax.grid(ls="--")
    ax.legend(ncol=3, loc="lower right")
axs[1].set_xlabel("Retardance in $\lambda$ at 560 nm")
plt.savefig("retardance_sRGB.pdf", bbox_inches="tight")
plt.show()
plt.close()

# Clip sRGB and plot as a function of retardance
intensity_orthogonal_sRGB_clip = np.clip(intensity_orthogonal_sRGB, 0, 1)
intensity_parallel_sRGB_clip = np.clip(intensity_parallel_sRGB, 0, 1)
fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True)
for ax, intensity_sRGB, pol_label in zip(axs, [intensity_orthogonal_sRGB_clip, intensity_parallel_sRGB_clip], polariser_labels):
    for sRGB, label in zip(intensity_sRGB, "rgb"):
        ax.plot(retardances_relative, sRGB, label=label, lw=3, c=label)
    ax.set_ylabel(f"sRGB ({pol_label})")
    ax.grid(ls="--")
    ax.legend(ncol=3, loc="lower right")
axs[1].set_xlabel("Retardance in $\lambda$ at 560 nm")
plt.savefig("retardance_sRGB_clipped.pdf", bbox_inches="tight")
plt.show()
plt.close()

# Create spectrum
fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(8,4), gridspec_kw={"hspace": 0.05, "wspace": 0})
for ax, intensity_sRGB, pol_label in zip(axs, [intensity_orthogonal_sRGB_clip, intensity_parallel_sRGB_clip], polariser_labels):
    rectangles = [plt.Rectangle((d,0), 5/nr_retardances, 1, color=sRGB) for d, sRGB in zip(retardances_relative, intensity_sRGB.T)]
    for rect in rectangles:
        ax.add_patch(rect)
    ax.tick_params(axis="y", labelleft=False, left=False)
    ax.set_ylabel(pol_label)
    ax.set_xlim(0, 5)
    ax.set_xlabel("Retardance ($\lambda$ at 560 nm)")
axs[0].tick_params(axis="x", labelbottom=False, bottom=False, labeltop=True, top=True)
axs[0].xaxis.set_label_position("top")
plt.savefig("retardance_spectrum.pdf", bbox_inches="tight")
plt.show()
plt.close()

for i, (retardance_relative, retardance_absolute, intensity_orthogonal, intensity_parallel, xy_orthogonal, xy_parallel, sRGB_orthogonal, sRGB_parallel) in enumerate(zip(retardances_relative, retardances_absolute, intensities_orthogonal, intensities_parallel, intensity_orthogonal_xy.T, intensity_parallel_xy.T, intensity_orthogonal_sRGB_clip.T, intensity_parallel_sRGB_clip.T)):
    # Status indicator
    label = f"Retardance: {retardance_absolute:5.0f} nm ; {retardance_relative:4.1f} $\lambda$"

    # Make plots
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(8,6), sharex="col", sharey="col")

    # First panel: intensity spectrum
    for ax, intensity, pol_label in zip(axs[:,0], [intensity_orthogonal, intensity_parallel], polariser_labels):
        ax.plot(wavelengths, intensity)
        ax.set_xlim(380, 720)
        ax.set_ylim(-0.02, 0.5)
        ax.set_ylabel(f"{pol_label}\nIntensity [a.u.]")

    axs[0,0].set_title(f"Retardance:\n{retardance_absolute:5.0f} nm    {retardance_relative:4.1f} $\lambda$")
    axs[1,0].set_xlabel("Wavelength [nm]")

    # Second panel: gamut
    for ax, intensity_xy, xy in zip(axs[:,1], [intensity_orthogonal_xy, intensity_parallel_xy], [xy_orthogonal, xy_parallel]):
        plt.sca(ax)
        plot_flat_gamut(plot_planckian_locus=False, axes_labels=("", ""))
        plt.plot(*intensity_xy[:,:i+1], c='k')
        plt.scatter(*xy, c='k')
        ax.set_title(f"$(x, y) = ({xy[0]:.2f}, {xy[1]:.2f})$")

    # Third panel: colour
    for ax, sRGB in zip(axs[:,2], [sRGB_orthogonal, sRGB_parallel]):
        rect = plt.Rectangle((0, 0), 1, 1, facecolor=sRGB)
        ax.add_patch(rect)
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        ax.set_aspect("equal")
        ax.set_title(f"$(r, g, b) = ({sRGB[0]:.2f}, {sRGB[1]:.2f}, {sRGB[2]:.2f})$")

    plt.savefig(f"animation/gamut_{retardance_absolute:07.2f}.png")
    plt.show()
    plt.close()

    print(label)
