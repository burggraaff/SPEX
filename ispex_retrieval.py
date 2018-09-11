import pol, spex
import numpy as np
from matplotlib import pyplot as plt

wavelengths = np.arange(297, 802, 0.3)
source = pol.Stokes_nm(np.ones_like(wavelengths), 0.7, 0., 0)

I, Q, U, V = spex.simulate_iSPEX(wavelengths, source)

fig, ax = plt.subplots(figsize=(10,5), tight_layout=True)
ax.plot(wavelengths, I , c='k')
ax.set_xlim(300, 800)
ax.set_ylim(0, 1)
ax.set_xlabel("$\lambda$ (nm)")
ax.set_ylabel("Stokes $I$")
plt.show()

DoLP_real = pol.DoLP(*source[0]) ; AoLP_real = pol.AoLP_deg(*source[0])
print(f"Real: DoLP = {DoLP_real:.2f}, AoLP = {AoLP_real:.1f} degrees")
DoLP, AoLP = spex.retrieve_DoLP(wavelengths, source, I)
print(f"Optimal: DoLP = {DoLP:.2f}, AoLP = {AoLP:.1f} degrees")