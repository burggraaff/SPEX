"""
Helper functions for groundSPEX
"""

import numpy as np
from numpy.polynomial.polynomial import polyval2d
from matplotlib import pyplot as plt
from pathlib import Path
from scipy.io import readsav


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
    texp = np.tile(1000., len(data))
    temperature = np.tile(25., (len(data), 2))

    # Load darkmap from file if none was given
    if darkmap is None:
        darkmap = read_darkmap()

    # Ensure the axes are in the right order for numpy broadcasting
    # New order: [5, 5, nr_exposures, nr_pixels]
    poly_darkpixels = np.moveaxis(darkmap.darkmodblack, (2, 3), (0, 1))
    poly_spectrum = np.moveaxis(darkmap.darkmodspec, (2, 3), (0, 1))

    # Apply the polynomials per channel (cannot broadcast fully, sadly)
    darkblack = np.array([polyval2d(texp, temperature[:,j], poly_darkpixels)[j] for j in range(2)])
    darkblack = np.moveaxis(darkblack, 2, 0)

    darkspec = np.array([polyval2d(texp, temperature[:,j], poly_spectrum)[j] for j in range(2)])
    darkspec = np.moveaxis(darkspec, 2, 0)

    # Apply the correction
    correction_darkpixels = np.nanmean(data_dark - darkblack, axis=2)
    data_corrected = data - darkspec - correction_darkpixels[...,np.newaxis]

    return data_corrected
