"""
groundSPEX instrument properties
"""
import numpy as np

# Array with pixels
PIXEL_NUMBER_AVANTES = 3648
PIXEL_ARRAY_DUALCHANNEL = np.tile(np.arange(PIXEL_NUMBER_AVANTES), (2,1))
