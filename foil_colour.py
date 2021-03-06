import numpy as np
from matplotlib import pyplot as plt
from spex import stokes, elements
from spectacle.general import gauss1d, gaussMd

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

# Load the SRFs
SRF_path = r"C:\Users\Burggraaff\SPECTACLE_data\iPhone_SE\calibration\iPhone_SE_spectral_response.csv"
SRF = np.loadtxt(SRF_path, delimiter=",")
SRF_wavelengths = SRF[:,0]
SRF_RGB = SRF[:,1:4]

# Interpolate the SRFs to the source spectrum wavelengths
SRF_RGB_interp = np.stack([np.interp(wavelengths, SRF_wavelengths, SRF, left=0, right=0) for SRF in SRF_RGB.T])

# Create the input and output polarisers at 0 and 90 degrees, respectively
polariser_0 = elements.Linear_polarizer_general(0.9, 0.005, 0)
polariser_90 = elements.Linear_polarizer_general(0.9, 0.005, 90)

# Propagate the source light through the input polariser
after_polariser_0 = np.einsum("ij,wi->wj", polariser_0, source)

# Retardances to loop over
retardances_relative = np.linspace(0, 10, 250)
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

for i, (retardance_relative, retardance_absolute, intensity) in enumerate(zip(retardances_relative, retardances_absolute, intensities)):
    # Status indicator
    label = f"Retardance: {retardance_absolute:5.0f} nm ; {retardance_relative:4.1f} $\lambda$"

    # Make plots
    plt.figure(figsize=(5,4))
    plt.plot(wavelengths, intensity)
    plt.xlim(380, 720)
    plt.ylim(-0.02, 0.5)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Intensity [a.u.]")
    plt.annotate(label, xy=(0.1,0.9), xycoords="figure fraction")
    plt.savefig(f"animation/raw_spectrum_{retardance_absolute:07.2f}.png")
    plt.close()

    print(label)

# Integrate over the RGB bands
source_RGB = np.einsum("w,jw->j", source_intensity, SRF_RGB_interp) * wavelength_step
integral_SRF = np.einsum("rw,jw->jr", intensities, SRF_RGB_interp) * wavelength_step
integral_SRF_relative = integral_SRF / source_RGB[:,np.newaxis]

for j, c in enumerate("rgb"):
    plt.plot(retardances_relative, integral_SRF_relative[j], c=c)
plt.xlabel("Retardance in $\lambda$ at 560 nm")
plt.ylabel("RGB value (relative)")
plt.savefig("retardance_RGB.pdf", bbox_inches="tight")
plt.show()
plt.close()

plt.plot(retardances_relative, np.rad2deg(np.arctan2(integral_SRF_relative[0], integral_SRF_relative[2])))
plt.xlabel("Retardance in $\lambda$ at 560 nm")
plt.ylabel("R/B hue angle [degrees]")
plt.savefig("retardance_hue.pdf", bbox_inches="tight")
plt.show()
plt.close()

intensities_smooth = gaussMd(intensities, sigma=(0,10))
derivative = np.diff(intensities_smooth, axis=1)

def stat(spectrum):
    inds = np.where((wavelengths >= 420) & (wavelengths <= 800))
    low = np.where(spectrum[inds] < 0.02)[0]
    steps = np.diff(low)
    nr_minima = len(np.where(steps > 5)[0])
    return nr_minima

statistic = [stat(spectrum) for spectrum in intensities_smooth]

plt.plot(retardances_relative, statistic)
plt.xlabel("Retardance in $\lambda$ at 560 nm")
plt.ylabel("Statistic")
plt.savefig("retardance_stat.pdf", bbox_inches="tight")
plt.show()
plt.close()
