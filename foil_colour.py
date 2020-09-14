import numpy as np
from matplotlib import pyplot as plt
import pol, spex

retardance = 4000. #nm

wavelengths = np.arange(380, 720, 0.3)
source = pol.Stokes_nm(np.ones_like(wavelengths), 0, 0, 0)

