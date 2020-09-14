import numpy as np
from matplotlib import pyplot as plt
from spex import stokes, elements

# Load the input spectrum
input_spectrum = np.loadtxt("input_data/lamp_spectrum.txt", skiprows=14)
input_spectrum = input_spectrum[input_spectrum[:,0] >= 300]
# input_spectrum = input_spectrum[input_spectrum[:,0] <= 720]

# Create the source spectrum, fully unpolarised
wavelengths = input_spectrum[:,0]
source_intensity = input_spectrum[:,1] / np.nanmax(input_spectrum[:,1])
source = stokes.Stokes_nm(source_intensity, 0, 0, 0)

# Create the input and output polarisers at 0 and 90 degrees, respectively
polariser_0 = elements.Linear_polarizer_0
polariser_90 = elements.Linear_polarizer_degrees(90)

# Propagate the source light through the input polariser
after_polariser_0 = np.einsum("ij,wj->wi", polariser_0, source)

for retardance_560 in np.linspace(0, 10, 250):
    # Create the foil
    retardance = retardance_560*560. # nm
    label = f"Retardance: {retardance:.0f} nm ; {retardance_560:4.1f} $\lambda$"
    foil = elements.Retarder_wavelengths(retardance, 45, wavelengths)

    # Propagate the light through the foil
    after_foil = np.einsum("wij,wj->wi", foil, after_polariser_0)

    # Propagate the light through the output polariser
    after_polariser_90 = np.einsum("ij,wj->wi", polariser_90, after_foil)

    # Get the unpolarised intensity
    intensity = after_polariser_90[:,0]

    plt.figure(figsize=(5,4))
    plt.plot(wavelengths, intensity)
    plt.xlim(380, 720)
    plt.ylim(-0.02, 0.6)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Intensity [a.u.]")
    plt.annotate(label, xy=(0.1,0.9), xycoords="figure fraction")
    plt.savefig(f"animation/raw_spectrum_{retardance:07.2f}.png")
    plt.close()

    print(retardance, "nm")
