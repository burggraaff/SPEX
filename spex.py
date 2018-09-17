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

def simulate_iSPEX2(wavelengths, source, QWP_d=140., QWP_t=0., MOR1_d=2240., MOR1_t=-45., MOR2_d=2240., MOR2_t=-45., POL0_t=0., POL90_t=90.):
    QWP  = pol.Retarder_wavelengths(QWP_d, QWP_t, wavelengths)
    MOR1 = pol.Retarder_wavelengths(MOR1_d, MOR1_t, wavelengths)
    MOR2 = pol.Retarder_wavelengths(MOR2_d, MOR2_t, wavelengths)
    POL0 = pol.LinPol_deg(POL0_t )
    POL90= pol.LinPol_deg(POL90_t)

    after_QWP  = np.einsum("wij,wj->wi", QWP , source    )
    after_MOR1 = np.einsum("wij,wj->wi", MOR1, after_QWP )
    after_MOR2 = np.einsum("wij,wj->wi", MOR2, after_MOR1)
    after_POL0 = np.einsum( "ij,wj->wi", POL0 , after_MOR2)
    after_POL90= np.einsum( "ij,wj->wi", POL90, after_MOR2)

    return after_POL0[:,0], after_POL90[:,0]  # transpose to split I, Q, U, V if wanted

def simulate_iSPEX_error(wavelengths, source, parameter, prange):
    default = signature(simulate_iSPEX).parameters[parameter].default
    kwargs_list = [{parameter: default + p} for p in prange]
    Is = np.array([simulate_iSPEX(wavelengths, source, **kwargs)[0] for kwargs in kwargs_list])
    return Is

def simulate_iSPEX2_error(wavelengths, source, parameter, prange):
    default = signature(simulate_iSPEX2).parameters[parameter].default
    kwargs_list = [{parameter: default + p} for p in prange]
    I0s, I90s = zip(*[simulate_iSPEX2(wavelengths, source, **kwargs) for kwargs in kwargs_list])
    I0s = np.array(I0s) ; I90s = np.array(I90s)
    return I0s, I90s

def retrieve_DoLP(wavelengths, source, I, delta=4480):
    mod_source = lambda wvl, DoLP, AoLP: pol.modulation(wvl, source[:,0], DoLP, AoLP, delta)
    dolp_init = I.max() - I.min()
    popt, pcov = curve_fit(mod_source, wavelengths, I, bounds=([0, -90], [1, 90]), p0=[dolp_init,0])
    DoLP, AoLP = popt
    if AoLP <= -89.9:
        AoLP += 180
    return DoLP, AoLP

def retrieve_DoLP_many(wavelengths, source, Is, **kwargs):
    DoLPs, AoLPs = np.array([retrieve_DoLP(wavelengths, source, I, **kwargs) for I in Is]).T
    return DoLPs, AoLPs

def _correct_AoLP90(A90):
    A = np.array(A90)
    A += 90
    A[A > 90] -= 180
    return A

def _merge_DoLP_AoLP(D0, D90, A0, A90):
    A = np.stack([A0, A90]).mean(axis=0)
    D = np.stack([D0, D90]).mean(axis=0)
    use_0 = np.where((A0 > -5) & (A0 <  5))
    use_90= np.where((A90> 85) | (A90<-85))
    A[use_0] = A0 [use_0]
    D[use_0] = D0 [use_0]
    A[use_90]= A90[use_90]
    D[use_90]= D90[use_90]
    return D, A

def retrieve_DoLP_many2(wavelengths, source, I0s, I90s, **kwargs):
    DoLPs0 , AoLPs0  = np.array([retrieve_DoLP(wavelengths, source, I, **kwargs) for I in I0s ]).T
    DoLPs90, AoLPs90 = np.array([retrieve_DoLP(wavelengths, source, I, **kwargs) for I in I90s]).T
    AoLPs90 = _correct_AoLP90(AoLPs90)
    D, A = _merge_DoLP_AoLP(DoLPs0, DoLPs90, AoLPs0, AoLPs90)
    return D, A