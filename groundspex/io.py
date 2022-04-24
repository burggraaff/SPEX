"""
groundSPEX input/output functions
"""
from functools import partial
from pathlib import Path
import numpy as np
from scipy.io import readsav
from .instrument import PIXEL_NUMBER_AVANTES

load_csv = partial(np.genfromtxt, delimiter=",")

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
    counts = load_csv(filename)[:-1]  # Remove the last element which is always empty

    # groundSPEX data files have all spectra concatenated, so if we have multiple spectra in this file, split them
    if len(counts) > PIXEL_NUMBER_AVANTES:
        counts = counts.reshape((-1, PIXEL_NUMBER_AVANTES))

    return counts


def load_data_file_dark(filename):
    """
    Load a dark pixel file from a spectrum filename.
    """
    filename_dark = filename.with_suffix(".dark13.txt")
    counts_dark = load_csv(filename_dark)[...,:13]  # Remove the 14th element in each row, which is always empty
    return counts_dark


def load_data_file_timestamp(filename):
    """
    Load a timestamp data file from a spectrum filename.
    """
    filename_timestamp = filename.with_suffix(".timestamps.txt")
    timestamp = np.loadtxt(filename_timestamp, dtype=np.int64)
    return timestamp


def load_data_bulk(filenames1, filenames2, load=load_data_file):
    """
    Load data in bulk.
    `load` can be any loading function.
    """
    # Load all the data - shape will be [2, nr_files, nr_spectra, nr_pixels]
    data = np.array([[load(f) for f in filenames] for filenames in [filenames1, filenames2]])

    # Reshape to [nr_files, nr_spectra, 2, nr_pixels]
    data = np.moveaxis(data, (1, 2), (0, 1))

    # If N = 1, remove that axis
    data = np.squeeze(data)

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


def read_transmission_correction(filename="pipeline_GvH/transmission.sav"):
    """
    Load the transmission correction from a .sav file.
    """
    correction = readsav(filename)["t2"]
    return correction


def read_efficiency(filename="pipeline_GvH/efficiency.sav"):
    """
    Read the polarisation efficiency data from a .sav file.
    """
    data = readsav(filename)["fitout"]
    return data
