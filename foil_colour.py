import numpy as np
from matplotlib import pyplot as plt
from spex import stokes, elements

# Create the source spectrum, fully unpolarised
wavelengths = np.arange(380, 720, 0.3)
source = stokes.Stokes_nm(np.ones_like(wavelengths), 0, 0, 0)

# Create the input and output polarisers at 0 and 90 degrees, respectively
polariser_0 = elements.Linear_polarizer_0
polariser_90 = elements.Linear_polarizer_degrees(90)

# Propagate the source light through the input polariser
after_polariser_0 = np.einsum("ij,wj->wi", polariser_0, source)

# Create the foil
retardance = 10*560. #nm
foil = elements.Retarder_wavelengths(retardance, 45, wavelengths)

# Propagate the light through the foil
after_foil = np.einsum("wij,wj->wi", foil, after_polariser_0)

# Propagate the light through the output polariser
after_polariser_90 = np.einsum("ij,wj->wi", polariser_90, after_foil)

plt.plot(wavelengths, after_polariser_90[:,0])
plt.xlabel("Wavelength [nm]")
plt.ylabel("Intensity [a.u.]")
plt.title(f"Retardance: {retardance:.0f} nm, {retardance/560:.1f} $\lambda$ at 560 nm")
plt.show()
plt.close()
