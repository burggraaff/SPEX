"""
Demodulation pipeline for groundSPEX data
"""

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from sys import argv
import groundspex as G

# Get data folder from command-line input
data_folder = Path(argv[1])
print(f"Loading data from {data_folder.absolute()}")

# Load spectra
data_filenames1, data_filenames2 = G.get_filenames(data_folder)

data = np.array([[G.load_data(f) for f in filenames] for filenames in [data_filenames1, data_filenames2]])
