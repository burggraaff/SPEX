import pol
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import spex
import spex_errors as spex_E

wavelengths = np.arange(450, 700, 0.3)
D = "\u0394"
a = "\u03B1"

steps = 35

QWP_d  = np.tile(np.nan, (steps, steps))
QWP_t  = QWP_d.copy()
MOR1_d = QWP_d.copy()
MOR1_t = QWP_d.copy()
POL0_t = QWP_d.copy()
POL90_t= QWP_d.copy()

Qrange = np.linspace(-1, 1, steps)
Urange = Qrange.copy()

Usq, Qsq = np.meshgrid(Urange, Qrange)
Dsq = pol.DoLP(1, Qsq, Usq, 0)
Asq = pol.AoLP_deg(1, Qsq, Usq, 0)

def plot(data, label=None, **kwargs):
    plt.figure(figsize=(6,5))
    plt.imshow(data, origin="lower", extent=(-1,1,-1,1), **kwargs)
    plt.xlabel("$U/I$")
    plt.ylabel("$Q/I$")
    plt.title(f"{label}: minimum {np.nanmin(data):.1f}")
    plt.colorbar(aspect=10)
    plt.tight_layout()
    if label:
        plt.savefig(f"margins_{label}.png")
    plt.close()

def plot_DA(D, A, data, label=None, **kwargs):
    plt.figure(figsize=(6,5))
    plt.contourf(D, A, data)
    plt.xlabel("DoLP")
    plt.ylabel("AoLP (degrees)")
    plt.xlim(0, 1)
    plt.ylim(-90, 90)
    plt.title(f"{label}: minimum {np.nanmin(data):.1f}")
    plt.colorbar(aspect=10)
    plt.tight_layout()
    if label:
        plt.savefig(f"margins_AD_{label}.png")
    plt.close()

for i,Q in enumerate(Qrange):
    for j,U in enumerate(Urange):
        perc = 100 * (steps * i + j) / steps**2
        print(f"{perc:.1f}", end="     ")
        if not 0 < Q**2 + U**2 <= 1:
            print("")
            continue
        print(f"Q = {Q:.2f}, U = {U:.2f}")
        source = pol.Stokes_nm(np.ones_like(wavelengths), Q, U, 0.)

        I, *_ = spex.simulate_iSPEX(wavelengths, source)
        DoLP_real = pol.DoLP(*source[0]) ; AoLP_real = pol.AoLP_deg(*source[0])
        print(f"Real: DoLP = {DoLP_real:.2f}, AoLP = {AoLP_real:.1f} degrees")
        DoLP, AoLP = spex.retrieve_DoLP(wavelengths, source, I)
        print(f"Optimal: DoLP = {DoLP:.2f}, AoLP = {AoLP:.1f} degrees")

        QWP_ds = np.linspace(-15, 15, 100)
        I0, I90 = spex.simulate_iSPEX2_error(wavelengths, source, "QWP_d", QWP_ds)
        DoLPs, AoLPs = spex.retrieve_DoLP_many2(wavelengths, source, I0, I90)
        QWP_d[i,j] = spex_E.margin(QWP_ds, DoLPs, AoLPs, DoLP_real, AoLP_real)

        QWP_ts = np.linspace(-12, 12, 100)
        I0, I90 = spex.simulate_iSPEX2_error(wavelengths, source, "QWP_t", QWP_ts)
        DoLPs, AoLPs = spex.retrieve_DoLP_many2(wavelengths, source, I0, I90)
        QWP_t[i,j] = spex_E.margin(QWP_ts, DoLPs, AoLPs, DoLP_real, AoLP_real)

        MOR1_ds = np.linspace(-30, 30, 100)
        I0, I90 = spex.simulate_iSPEX2_error(wavelengths, source, "MOR1_d", MOR1_ds)
        DoLPs, AoLPs = spex.retrieve_DoLP_many2(wavelengths, source, I0, I90)
        MOR1_d[i,j] = spex_E.margin(MOR1_ds, DoLPs, AoLPs, DoLP_real, AoLP_real)

        MOR1_ts = np.linspace(-9, 9, 100)
        I0, I90 = spex.simulate_iSPEX2_error(wavelengths, source, "MOR1_t", MOR1_ts)
        DoLPs, AoLPs = spex.retrieve_DoLP_many2(wavelengths, source, I0, I90)
        MOR1_t[i,j] = spex_E.margin(MOR1_ts, DoLPs, AoLPs, DoLP_real, AoLP_real)

        POL_ts = np.linspace(-9, 9, 100)
        I0, I90 = spex.simulate_iSPEX2_error(wavelengths, source, "POL0_t" , POL_ts)
        DoLPs, AoLPs = spex.retrieve_DoLP_many2(wavelengths, source, I0, I90)
        POL0_t [i,j] = spex_E.margin(POL_ts, DoLPs, AoLPs, DoLP_real, AoLP_real)

        I0, I90 = spex.simulate_iSPEX2_error(wavelengths, source, "POL90_t", POL_ts)
        DoLPs, AoLPs = spex.retrieve_DoLP_many2(wavelengths, source, I0, I90)
        POL90_t[i,j] = spex_E.margin(POL_ts, DoLPs, AoLPs, DoLP_real, AoLP_real)

for arr, label in zip([QWP_d, QWP_t, MOR1_d, MOR1_t, POL0_t, POL90_t], ["QWP_d", "QWP_t", "MOR1_d", "MOR1_t", "POL0_t", "POL90_t"]):
    plot(arr, label)
    plot_DA(Dsq, Asq, arr, label)
    np.save(f"margins_{label}.npy", arr)
