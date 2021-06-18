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
    data_filenames1 = folder.glob("Spectrometer_1105161U2_*_pix.txt")
    data_filenames2 = folder.glob("Spectrometer_1105162U2_*_pix.txt")

    return data_filenames1, data_filenames2


def load_data_file(filename):
    """
    Load a groundSPEX spectrum from file.
    """
    counts = np.genfromtxt(filename, delimiter=",")[:-1]

    return counts


def load_data_folder(folder):
    """
    Load all groundSPEX data from a folder.
    """
    # Get the filenames
    data_filenames1, data_filenames2 = get_filenames(folder)

    # Load the data into an array of shape [2, N, 3648] with N the number of files
    data = np.array([[load_data_file(f) for f in filenames] for filenames in [data_filenames1, data_filenames2]])

    return data
