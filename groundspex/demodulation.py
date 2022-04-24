"""
groundSPEX spectropolarimetric demodulation code
"""
import numpy as np

def demodulate(data, wavelengths=None, wavelength_range=[370., 800.]):
    """
    Main demodulation routine.
    """
    if wavelengths is None:
        wavelengths = generate_wavelengths()

    # Select the relevant wavelengths
    wvl = np.where((wavelengths > wavelength_range[0]) & (wavelengths < wavelength_range[1]))[0]
    wavelengths, data = wavelengths[wvl], data[...,wvl]
