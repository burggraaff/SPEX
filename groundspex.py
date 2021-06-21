"""
Helper functions for groundSPEX
"""

import numpy as np
from numpy.polynomial.polynomial import polyval, polyval2d
from matplotlib import pyplot as plt
from pathlib import Path
from scipy.io import readsav

# Wavelength coefficients determined by Gerard van Harten, Avantes, Jos de Boer, respectively
wavelength_coeffs_GvH = np.array([[355.688, 0.167436, -2.93242e-06, -2.22549e-10], [360.071, 0.165454, -3.35036e-06, -1.88750e-10]])
wavelength_coeffs_Avantes = np.array([[356.377,0.167307,-2.97917e-6, -2.14825e-10], [360.610,0.165407,-3.46191e-6,-1.71402e-10]])
wavelength_coeffs_JdB = np.array([[356.058,0.167297,-2.88384e-6,-2.28596e-10], [360.120,0.165363,-3.33891e-6,-1.90909e-10]])

# Array with pixels
nr_pixels_Avantes = 3648
spectrum_pixels = np.tile(np.arange(nr_pixels_Avantes), (2,1))


def get_filenames(folder):
    """
    Get the filenames for both spectrometers from a given folder.
    """
    # Ensure the folder is a Path object
    folder = Path(folder)

    # Helper function for sorting files (make sure 10 comes after 9, not before 1)
    sorter = lambda p: int(p.stem.split("_")[2])

    # Find all corresponding files
    data_filenames1 = sorted(folder.glob("Spectrometer_1105161U2_*_pix.txt"), key=sorter)
    data_filenames2 = sorted(folder.glob("Spectrometer_1105162U2_*_pix.txt"), key=sorter)

    return data_filenames1, data_filenames2


def load_data_file(filename):
    """
    Load a groundSPEX spectrum from file.
    """
    counts = np.genfromtxt(filename, delimiter=",")[:-1]

    return counts


def load_data_file_dark(filename):
    """
    Load a dark pixel file from a spectrum filename.
    """
    filename_dark = filename.with_suffix(".dark13.txt")
    dark_data = load_data_file(filename_dark)
    return dark_data


def load_data_file_timestamp(filename):
    """
    Load a timestamp data file from a spectrum filename.
    """
    filename_timestamp = filename.with_suffix(".timestamps.txt")
    timestamp = np.loadtxt(filename_timestamp, dtype=np.int32)
    return timestamp


def load_data_bulk(filenames1, filenames2, load=load_data_file):
    """
    Load data in bulk.
    `load` can be any loading function.
    """
    # Load all the data - shape will be [2, N, ...] with N the number of files
    data = np.array([[load(f) for f in filenames] for filenames in [filenames1, filenames2]])

    # Let N be the first axis: [N, 2, ...]
    data = np.moveaxis(data, 1, 0)

    return data


def load_data_folder(folder):
    """
    Load all groundSPEX data from a folder.
    """
    # Get the filenames
    data_filenames1, data_filenames2 = get_filenames(folder)

    # Load the data into an array of shape [2, N, 3648] with N the number of files
    data = load_data_bulk(data_filenames1, data_filenames2)
    data_dark = load_data_bulk(data_filenames1, data_filenames2, load_data_file_dark)
    data_timestamps = load_data_bulk(data_filenames1, data_filenames2, load_data_file_timestamp)

    return data, data_dark, data_timestamps


def read_darkmap(filename="pipeline_GvH/darkmap.sav"):
    """
    Load a darkmap from a .sav file.
    """
    darkmap = readsav(filename)
    return darkmap


def correct_darkcurrent(data, data_dark, darkmap=None):
    """
    Apply a dark current correction to given data.
    If no darkmap is given, load one from file.
    """
    # Hard code metadata for now because files were missing
    texp = np.tile(1000., (len(data),2))
    temperature = np.tile([25.,26], (len(data), 1))

    # Load darkmap from file if none was given
    if darkmap is None:
        darkmap = read_darkmap()

    # Ensure the axes are in the right order for numpy broadcasting
    # New order: [5, 5, nr_channels, nr_pixels]
    polynomial_coeffs_darkpixels = np.moveaxis(darkmap.darkmodblack, (2, 3), (0, 1))
    polynomial_coeffs_spectrum = np.moveaxis(darkmap.darkmodspec, (2, 3), (0, 1))

    # Function to apply the 2D polynomial to each pixel
    # Moveaxis and diagonal are necessary to only get the useful elements, and have
    # them in the right places
    apply_polynomial = lambda x, y, c: np.moveaxis(np.diagonal(polyval2d(x, y, c), axis2=3), 0, 2)

    # Apply the polynomials
    darkcurrent_darkpixels = apply_polynomial(texp, temperature, polynomial_coeffs_darkpixels)
    darkcurrent_spectrum = apply_polynomial(texp, temperature, polynomial_coeffs_spectrum)

    # Apply the correction
    correction_darkpixels = np.nanmean(data_dark - darkcurrent_darkpixels, axis=2)
    data_corrected = data - darkcurrent_spectrum - correction_darkpixels[...,np.newaxis]

    return data_corrected


def wavelengths(x=spectrum_pixels, coeffs=wavelength_coeffs_GvH):
    """
    Calculate the wavelengths corresponding to each pixel.
    """
    # Calculate the wavelengths corresponding to each pixel
    # Assumes both channels have the same number of pixels (which they do)
    wavelengths = polyval(x[0], coeffs.T)

    return wavelengths


def read_transmission_correction(filename="pipeline_GvH/transmission.sav"):
    """
    Load the transmission correction from a .sav file.
    """
    correction = readsav(filename)["t2"]
    return correction


def correct_transmission(data, transmission_correction_data=None):
    """
    Apply a transmission correction to given data.
    If no correction data are given, load from file.
    """
    if transmission_correction_data is None:
        transmission_correction_data = read_transmission_correction()

    # Correct the data - divide channel 1 by the correction data
    data_corrected = data.copy()
    data_corrected[:,1] /= transmission_correction_data

    return data_corrected


def read_efficiency(filename="pipeline_GvH/efficiency.sav"):
    """
    Read the polarisation efficiency data from a .sav file.
    """
    data = readsav(filename)["fitout"]
    return data
