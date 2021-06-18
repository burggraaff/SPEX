"""
Helper functions for groundSPEX
"""

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path


def get_filenames(folder):
    """
    Get the filenames for both spectrometers from a given folder.
    """
    # Ensure the folder is a Path object
    folder = Path(folder)

    # Find all corresponding files
    data_filenames1 = list(folder.glob("Spectrometer_1105161U2_*_pix.txt"))
    data_filenames2 = list(folder.glob("Spectrometer_1105162U2_*_pix.txt"))

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


def load_data_folder(folder):
    """
    Load all groundSPEX data from a folder.
    """
    # Get the filenames
    data_filenames1, data_filenames2 = get_filenames(folder)

    # Load the data into an array of shape [2, N, 3648] with N the number of files
    data = np.array([[load_data_file(f) for f in filenames] for filenames in [data_filenames1, data_filenames2]])
    data_dark = np.array([[load_data_file_dark(f) for f in filenames] for filenames in [data_filenames1, data_filenames2]])
    data_timestamps = np.array([[load_data_file_timestamp(f) for f in filenames] for filenames in [data_filenames1, data_filenames2]])

    return data, data_dark, data_timestamps
