import pol
import numpy as np
from inspect import signature
from scipy.optimize import curve_fit

def simulate_iSPEX(wavelengths, source, QWP_d=140., QWP_t=0., MOR1_d=2240., MOR1_t=-45., MOR2_d=2240., MOR2_t=-45., POL_t=0.):
    QWP = pol.Retarder_wavelengths(QWP_d, QWP_t, wavelengths)
    MOR1= pol.Retarder_wavelengths(MOR1_d, MOR1_t, wavelengths)
    MOR2= pol.Retarder_wavelengths(MOR2_d, MOR2_t, wavelengths)
    POL = pol.LinPol_deg(POL_t)

    after_QWP = np.einsum("wij,wj->wi", QWP , source    )
    after_MOR1= np.einsum("wij,wj->wi", MOR1, after_QWP )
    after_MOR2= np.einsum("wij,wj->wi", MOR2, after_MOR1)
    after_POL = np.einsum( "ij,wj->wi", POL , after_MOR2)

    return after_POL.T  # transpose to split I, Q, U, V if wanted

def simulate_iSPEX_error(wavelengths, source, parameter, prange):
    default = signature(simulate_iSPEX).parameters[parameter].default
    kwargs_list = [{parameter: default + p} for p in prange]
    Is = np.array([simulate_iSPEX(wavelengths, source, **kwargs)[0] for kwargs in kwargs_list])
    return Is

def retrieve_DoLP(wavelengths, source, I, delta=4480):
    mod_source = lambda wvl, DoLP, AoLP: pol.modulation(wvl, source[:,0], DoLP, AoLP, delta)
    popt, pcov = curve_fit(mod_source, wavelengths, I, bounds=([0, -90], [1, 90]), p0=[0.5,0])
    DoLP, AoLP = popt
    return DoLP, AoLP

def retrieve_DoLP_many(wavelengths, source, Is, **kwargs):
    DoLPs, AoLPs = np.array([retrieve_DoLP(wavelengths, source, I, **kwargs) for I in Is]).T
    return DoLPs, AoLPs