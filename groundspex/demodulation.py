"""
groundSPEX spectropolarimetric demodulation code
"""
import numpy as np
from scipy.optimize import curve_fit
from .instrument import MOR_RETARDANCE_NOMINAL


def crop_spectra(wavelengths, data, wavelength_limits=(419, 851)):
    """
    Crop given spectra and wavelengths to be within the `wavelength_limits`.
    """
    crop = np.where((wavelengths >= wavelength_limits[0]) & (wavelengths <= wavelength_limits[1]))[0]
    wavelengths_crop = wavelengths[crop]
    data_crop = data[...,crop]

    return wavelengths_crop, data_crop


def spectral_resolution_polarimetric(wavelengths, retardance_mor=MOR_RETARDANCE_NOMINAL):
    """
    Calculate the polarimetric spectral resolution, which is roughly equal to the local modulation period.
    Make sure that `wavelengths` and `retardance_mor` are in the same units, for example nm. The result will also be in those units.
    """
    return wavelengths**2 / retardance_mor


def calculate_psi(wavelengths, aolp, retardance):
    """
    Calculate the modulation phase as a function of wavelength.
    """
    return 2*np.pi*retardance/wavelengths + 2*aolp


def modulation(wavelengths, intensity, dolp, aolp, retardance_mor=MOR_RETARDANCE_NOMINAL):
    """
    From the unpolarised intensity and the degree and angle of linear polarisation, generate the modulated spectra in both channels of a dual-channel spectorpolarimeter.
    retardance_mor is the retardance of the multi-order retarder in the same units as wavelengths (not normalised!)
    All quantities are assumed to be wavelength-dependent.
    """
    psi = calculate_psi(wavelengths, aolp, retardance_mor)  # Helper variable: phase
    Splus = 0.5*intensity*(1 + dolp*np.cos(psi))
    Smin = 0.5*intensity*(1 - dolp*np.cos(psi))

    return Splus, Smin


def modulation_normalised(wavelengths, dolp, aolp, retardance_mor=MOR_RETARDANCE_NOMINAL):
    """
    From the degree and angle of linear polarisation, generate the modulation fraction (I+ - I-)/(I+ + I-).
    """
    psi = calculate_psi(wavelengths, aolp, retardance_mor)  # Helper variable: phase
    fraction = dolp*np.cos(psi)

    return fraction


def modulation_to_fit(wavelengths, dolp, aolp, offset, retardance_mor=MOR_RETARDANCE_NOMINAL):
    """
    Variation of modulation_normalised with an additional offset parameter, and with a fixed retardance.
    """
    psi = calculate_psi(wavelengths, aolp, retardance_mor)  # Helper variable: phase
    fraction = offset + 0.5*dolp*np.cos(psi)

    return fraction


def fit_modulation_single(wavelengths, ydata, func=modulation_to_fit, p0=[0.1, 0.1, 0.5, MOR_RETARDANCE_NOMINAL], bounds=([0., 0., -1., MOR_RETARDANCE_NOMINAL*0.9], [1., 2*np.pi, 1., MOR_RETARDANCE_NOMINAL*1.1]), **kwargs):
    """
    Fit the modulation of a single spectrum.
    Catches RuntimeErrors when there is no solution, and returns NaN in that case.
    """
    try:
        popt, pcov = curve_fit(func, wavelengths, ydata, p0=p0, bounds=bounds)
    except RuntimeError:
        popt = np.tile(np.nan, 4)
        pcov = np.tile(np.nan, (4, 4))
    return popt, pcov
