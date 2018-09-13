import pol, spex
import numpy as np
from matplotlib import pyplot as plt

wavelengths = np.arange(297, 802, 0.3)
source = pol.Stokes_nm(np.ones_like(wavelengths), 0, 1, 0)
I, Q, U, V = spex.simulate_iSPEX(wavelengths, source, MOR1_t = 45, MOR2_t = 45)

pure_Q = pol.Stokes_nm(np.ones_like(wavelengths), 1.0, 0, 0)
I_Q, *_ = spex.simulate_iSPEX(wavelengths, pure_Q)

fig, ax = plt.subplots(figsize=(10,5), tight_layout=True)
ax.plot(wavelengths, I , c='k')
ax.set_xlim(300, 800)
ax.set_ylim(0, 1)
ax.set_xlabel("$\lambda$ (nm)")
ax.set_ylabel("Stokes $I$")
ax.grid()
plt.show()

DoLP_real = pol.DoLP(*source[0]) ; AoLP_real = pol.AoLP_deg(*source[0])
print(f"Real: DoLP = {DoLP_real:.2f}, AoLP = {AoLP_real:.1f} degrees")
DoLP, AoLP = spex.retrieve_DoLP(wavelengths, source, I)
print(f"Optimal: DoLP = {DoLP:.2f}, AoLP = {AoLP:.1f} degrees")

def window_width(wavelength, retardance=4480.):
    return wavelength**2 / (retardance * (1 + wavelength**2 / (4 * retardance**2)))

widths = window_width(wavelengths)
indices = [np.where(np.abs(wavelengths - wvl) <= width) for wvl, width in zip(wavelengths, widths)]
spectrum = np.array([np.mean(I[ind]) for ind in indices])
I_new = I / spectrum

fig, ax = plt.subplots(figsize=(10,5), tight_layout=True)
ax.plot(wavelengths, I_new , c='k')
ax.set_xlim(300, 800)
ax.set_ylim(0, 2)
ax.set_xlabel("$\lambda$ (nm)")
ax.set_ylabel("Stokes $I$ (normalised)")
ax.grid()
plt.show()

DoLP_initial = np.array([(I_new[ind].max() - I_new[ind].min())/2 for ind in indices])

def AoLP_estimate(wavelengths, I, I_reference, central_wavelength):
    wvlstep = wavelengths[1] - wavelengths[0]
    width = window_width(central_wavelength)
    indices = np.where(np.abs(wavelengths - central_wavelength) <= width)
    correlations = np.correlate(I[indices], I_reference[indices], mode="full")
    lags = np.arange(-len(indices[0]) + 1, len(indices[0]))
    lags_nm = lags * wvlstep
#    return lags_nm, correlations
    shift = lags_nm[correlations.argmax()]
    AoLP = 180. * shift / width
    return AoLP

AoLP_initial = np.array([AoLP_estimate(wavelengths, I_new, I_Q, wvl) for wvl in wavelengths])