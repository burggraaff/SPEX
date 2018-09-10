import pol
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

wavelengths = np.arange(297, 802, 0.3)
source = pol.Stokes_nm(np.ones_like(wavelengths), 0.7, 0., 0)

def simulate_iSPEX(wavelengths, source, QWP_d=140., QWP_t=0., QWP_a=False, MOR1_d=2240., MOR1_t=45., MOR2_d=2240., MOR2_t=45., POL_t=0.):
    QWP = pol.Retarder_wavelengths(QWP_d, QWP_t, wavelengths)
    MOR1= pol.Retarder_wavelengths(MOR1_d, MOR1_t, wavelengths)
    MOR2= pol.Retarder_wavelengths(MOR2_d, MOR2_t, wavelengths)
    POL = pol.LinPol_deg(POL_t)

    after_QWP = np.einsum("wij,wj->wi", QWP , source    )
    after_MOR1= np.einsum("wij,wj->wi", MOR1, after_QWP )
    after_MOR2= np.einsum("wij,wj->wi", MOR2, after_MOR1)
    after_POL = np.einsum( "ij,wj->wi", POL , after_MOR2)

    return after_POL.T  # transpose to split I, Q, U, V if wanted

I, Q, U, V = simulate_iSPEX(wavelengths, source)
I2,Q2,U2,V2= simulate_iSPEX(wavelengths, source, POL_t=25)

fig, ax = plt.subplots(figsize=(10,5), tight_layout=True)
ax.plot(wavelengths, I , c='k')
ax.plot(wavelengths, I2, c='k', ls='--')
ax.set_xlim(300, 800)
ax.set_ylim(0, 1)
ax.set_xlabel("$\lambda$ (nm)")
ax.set_ylabel("Stokes $I$")
plt.show()

mod_source = lambda wvl, DoLP, AoLP: pol.modulation(wvl, source[:,0], DoLP, AoLP, 4480)

popt, pcov = curve_fit(mod_source, wavelengths, I, bounds=([0, 0], [1, 180]), p0=[0.5,90])
DoLP, AoLP = popt
print(f"Optimal: DoLP = {DoLP:.2f}, AoLP = {AoLP:.1f} degrees")

popt2, pcov2 = curve_fit(mod_source, wavelengths, I2, bounds=([0, 0], [1, 180]), p0=[0.5,90])
DoLP2, AoLP2 = popt2
print(f"Real: DoLP = {DoLP2:.2f}, AoLP = {AoLP2:.1f} degrees")