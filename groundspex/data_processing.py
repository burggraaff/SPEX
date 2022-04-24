"""
groundSPEX data processing (not demodulation) code
"""
from functools import partial
import numpy as np
from numpy.polynomial.polynomial import polyval, polyval2d
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from .instrument import PIXEL_ARRAY_DUALCHANNEL
from . import io

# Wavelength coefficients determined by Gerard van Harten, Avantes, Jos de Boer, respectively
wavelength_coeffs_GvH = np.array([[355.688, 0.167436, -2.93242e-06, -2.22549e-10], [360.071, 0.165454, -3.35036e-06, -1.88750e-10]])
wavelength_coeffs_Avantes = np.array([[356.377,0.167307,-2.97917e-6, -2.14825e-10], [360.610,0.165407,-3.46191e-6,-1.71402e-10]])
wavelength_coeffs_JdB = np.array([[356.058,0.167297,-2.88384e-6,-2.28596e-10], [360.120,0.165363,-3.33891e-6,-1.90909e-10]])


def correct_darkcurrent(data, data_dark, darkmap=None, texp=200., temperature=26.):
    """
    Apply a dark current correction to given data.
    If no darkmap is given, load one from file.
    """
    # Load darkmap from file if none was given
    if darkmap is None:
        darkmap = io.read_darkmap()

    # Ensure the axes are in the right order for numpy broadcasting
    # New order: [5, 5, nr_channels, nr_pixels]
    polynomial_coeffs_darkpixels = np.moveaxis(darkmap.darkmodblack, (2, 3), (0, 1))
    polynomial_coeffs_spectrum = np.moveaxis(darkmap.darkmodspec, (2, 3), (0, 1))

    # Function to apply the 2D polynomial to each pixel
    # apply_polynomial = lambda x, y, c: np.moveaxis(np.diagonal(polyval2d(x, y, c), axis2=-1), 0, 2)
    apply_polynomial = lambda x, y, c: polyval2d(x, y, c)

    # Apply the polynomials
    darkcurrent_darkpixels = apply_polynomial(texp, temperature, polynomial_coeffs_darkpixels)
    darkcurrent_spectrum = apply_polynomial(texp, temperature, polynomial_coeffs_spectrum)

    # Apply the correction
    correction_darkpixels = np.nanmean(data_dark - darkcurrent_darkpixels, axis=2)
    data_corrected = data - darkcurrent_spectrum - correction_darkpixels[...,np.newaxis]

    return data_corrected


def generate_wavelengths(x=PIXEL_ARRAY_DUALCHANNEL, coeffs=wavelength_coeffs_GvH):
    """
    Calculate the wavelengths corresponding to each pixel.
    """
    # Calculate the wavelengths corresponding to each pixel
    # Assumes both channels have the same number of pixels (which they do)
    wavelengths = polyval(x[0], coeffs.T)

    return wavelengths


def correct_wavelengths(data, wavelengths=None):
    """
    Correct the wavelength ranges to be equal.
    Interpolates channel 0 (which is wider) to channel 1's wavelengths.
    """
    if wavelengths is None:
        wavelengths = generate_wavelengths()

    # Generate a function that does the interpolation
    # This probably breaks down if you have a lot of exposures in data
    interpolation_function = interp1d(wavelengths[0], data[:,0])

    # Evaluate this function at the new wavelengths
    data_corrected = data.copy()
    data_corrected[:,0] = interpolation_function(wavelengths[1])

    # Return the wavelengths (same for both channels) and result
    return wavelengths[1], data_corrected


def correct_transmission(data, transmission_correction_data=None):
    """
    Apply a transmission correction to given data.
    If no correction data are given, load from file.
    """
    if transmission_correction_data is None:
        transmission_correction_data = io.read_transmission_correction()

    # Correct the data - divide channel 1 by the correction data
    data_corrected = data.copy()
    data_corrected[:,1] /= transmission_correction_data

    return data_corrected


def correct_efficiency(data, wavelengths=None, efficiency_data=None):
    """
    Correct for polarimetric efficiency.
    """
    if wavelengths is None:
        wavelengths = generate_wavelengths()

    if efficiency_data is None:
        efficiency_data = io.read_efficiency()

    # Indices closest to 660 and 672 nm
    ind_660 = np.nanargmin(np.abs(efficiency_data[0,5] - 660))
    ind_672 = np.nanargmin(np.abs(efficiency_data[0,5] - 672))
    wvlrange = np.s_[ind_660:ind_672]
