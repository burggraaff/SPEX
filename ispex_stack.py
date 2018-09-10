import pol
import numpy as np
from matplotlib import pyplot as plt

wavelengths = np.arange(297, 802, 2)
source = pol.Stokes_nm(np.ones_like(wavelengths), 1, 0, 0)

QWP = pol.Retarder_wavelengths(140, 0, wavelengths)
MOR1= pol.Retarder_wavelengths(2240, 45, wavelengths)
MOR2= pol.Retarder_wavelengths(2240, 45, wavelengths)
POL = pol.LinPol_deg(0)

after_QWP = np.einsum("wij,wj->wi", QWP, source)
after_MOR1= np.einsum("wij,wj->wi", MOR1, after_QWP)
after_MOR2= np.einsum("wij,wj->wi", MOR2, after_MOR1)
after_POL = np.einsum( "ij,wj->wi", POL, after_MOR2)

plt.plot(wavelengths, after_POL[:,0])
plt.xlim(300, 800)
plt.ylim(0, 1)
plt.show()