import pol
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from inspect import signature

wavelengths = np.arange(297, 802, 0.3)
source = pol.Stokes_nm(np.ones_like(wavelengths), 0.7, 0., 0)

def simulate_iSPEX(wavelengths, source, QWP_d=140., QWP_t=0., MOR1_d=2240., MOR1_t=45., MOR2_d=2240., MOR2_t=45., POL_t=0.):
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
    popt, pcov = curve_fit(mod_source, wavelengths, I, bounds=([0, 0], [1, 180]), p0=[0.5,90])
    DoLP, AoLP = popt
    return DoLP, AoLP

def retrieve_DoLP_many(wavelengths, source, I, **kwargs):
    DoLPs, AoLPs = np.array([retrieve_DoLP(wavelengths, source, I, **kwargs) for I in Is]).T
    return DoLPs, AoLPs

def plot(x, DoLPs, AoLPs, xlabel):
    fig, ax = plt.subplots(figsize=(10,5), tight_layout=True)
    ax2 = ax.twinx()
    ax.plot(x, DoLPs, c='k')
    ax2.plot(x, AoLPs, c='k', ls="--")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-0.01, 1.01)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax2.set_ylim(-1, 181)
    ax2.set_yticks(np.arange(0, 200, 20))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("DoLP")
    ax2.set_ylabel("AoLP")
    ax.set_title("Solid: DoLP ; Dashed: AoLP")
    return fig

I, Q, U, V = simulate_iSPEX(wavelengths, source)
print("Simulated ideal conditions")

QWP_ds = np.arange(-20, 20.5, 0.5)
Is = simulate_iSPEX_error(wavelengths, source, "QWP_d", QWP_ds)
print("Simulated QWP \u0394d")
DoLPs, AoLPs = retrieve_DoLP_many(wavelengths, source, Is)
fig = plot(QWP_ds, DoLPs, AoLPs, r"$\Delta d$ on QWP (nm)")
plt.show()

QWP_ts = np.arange(-25, 25, 0.5)
Is = simulate_iSPEX_error(wavelengths, source, "QWP_t", QWP_ts)
print("Simulated QWP \u0394\u03B1")
DoLPs, AoLPs = retrieve_DoLP_many(wavelengths, source, Is)
fig = plot(QWP_ts, DoLPs, AoLPs, r"$\Delta \alpha$ on QWP (degrees)")
plt.show()

QWP_ts = np.arange(-25, 25, 0.5)
Is = simulate_iSPEX_error(wavelengths, source, "QWP_t", QWP_ts)
print("Simulated QWP \u0394\u03B1")
DoLPs, AoLPs = retrieve_DoLP_many(wavelengths, source, Is)
fig = plot(QWP_ts, DoLPs, AoLPs, r"$\Delta \alpha$ on QWP (degrees)")
plt.show()