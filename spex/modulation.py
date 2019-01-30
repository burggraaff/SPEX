import numpy as np

def Modulation_matrix(polarimeter):
    # take first row of each Mueller matrix in `polarimeter'
    pass

def Demodulation_matrix(modulation_matrix):
    return np.linalg.pinv(modulation_matrix)
