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

def retrieve_DoLP(wavelengths, source, I, delta=4480):
    mod_source = lambda wvl, DoLP, AoLP: pol.modulation(wvl, source[:,0], DoLP, AoLP, delta)
    popt, pcov = curve_fit(mod_source, wavelengths, I, bounds=([0, 0], [1, 180]), p0=[0.5,90])
    DoLP, AoLP = popt
    return DoLP, AoLP

I, Q, U, V = simulate_iSPEX(wavelengths, source)

QWP_ds = np.arange(-20, 20, 0.5)
Is = np.array([simulate_iSPEX(wavelengths, source, QWP_d=140.+Q_diff)[0] for Q_diff in QWP_ds])
DoLPs = np.array([])

raise Exception