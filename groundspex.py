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


def load_data(filename):
    """
    Load a groundSPEX spectrum from file.
    """
    counts = np.genfromtxt(filename, delimiter=",")[:-1]

    return counts
