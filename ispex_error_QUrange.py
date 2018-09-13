import pol
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import spex

wavelengths = np.arange(450, 700, 0.3)
D = "\u0394"
a = "\u03B1"

steps = 30

QWP_d  = np.tile(np.nan, (steps, steps))
QWP_t  = QWP_d.copy()
MOR1_d = QWP_d.copy()
MOR1_t = QWP_d.copy()
POL_t  = QWP_d.copy()

def margin(x, DoLPs, AoLPs, real_DoLP, real_AoLP, Dlim=0.03, Alim=5):
    D_err = np.abs(DoLPs / real_DoLP - 1)
    A_err = np.abs(AoLPs - real_AoLP)

    D_ind = np.where(D_err > Dlim)
    A_ind = np.where(A_err > Alim)

    try:
        D_min = np.abs(x[D_ind]).min()
    except ValueError:
        D_min = np.abs(x).max()
    try:
        A_min = np.abs(x[A_ind]).min()
    except ValueError:
        A_min = np.abs(x).max()

    x_min = np.min([D_min, A_min])

    return x_min

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

for i,Q in enumerate(np.linspace(-1, 1, steps)):
    for j,U in enumerate(np.linspace(-1, 1, steps)):
        if not 0 < Q**2 + U**2 <= 1:
            continue
        print(f"Q = {Q:.2f}, U = {U:.2f}")
        source = pol.Stokes_nm(np.ones_like(wavelengths), Q, U, 0.)

        I, *_ = spex.simulate_iSPEX(wavelengths, source)
        DoLP_real = pol.DoLP(*source[0]) ; AoLP_real = pol.AoLP_deg(*source[0])
        print(f"Real: DoLP = {DoLP_real:.2f}, AoLP = {AoLP_real:.1f} degrees")
        DoLP, AoLP = spex.retrieve_DoLP(wavelengths, source, I)
        print(f"Optimal: DoLP = {DoLP:.2f}, AoLP = {AoLP:.1f} degrees")

        QWP_ds = np.linspace(-15, 15, 100)
        Is = spex.simulate_iSPEX_error(wavelengths, source, "QWP_d", QWP_ds)
        DoLPs, AoLPs = spex.retrieve_DoLP_many(wavelengths, source, Is)
        QWP_d[i,j] = margin(QWP_ds, DoLPs, AoLPs, DoLP_real, AoLP_real)

        QWP_ts = np.linspace(-12, 12, 100)
        Is = spex.simulate_iSPEX_error(wavelengths, source, "QWP_t", QWP_ts)
        DoLPs, AoLPs = spex.retrieve_DoLP_many(wavelengths, source, Is)
        QWP_t[i,j] = margin(QWP_ts, DoLPs, AoLPs, DoLP_real, AoLP_real)

        MOR1_ds = np.linspace(-30, 30, 100)
        Is = spex.simulate_iSPEX_error(wavelengths, source, "MOR1_d", MOR1_ds)
        DoLPs, AoLPs = spex.retrieve_DoLP_many(wavelengths, source, Is)
        MOR1_d[i,j] = margin(MOR1_ds, DoLPs, AoLPs, DoLP_real, AoLP_real)

        MOR1_ts = np.linspace(-9, 9, 100)
        Is = spex.simulate_iSPEX_error(wavelengths, source, "MOR1_t", MOR1_ts)
        DoLPs, AoLPs = spex.retrieve_DoLP_many(wavelengths, source, Is)
        MOR1_t[i,j] = margin(MOR1_ts, DoLPs, AoLPs, DoLP_real, AoLP_real)

        POL_ts = np.linspace(-9, 9, 100)
        Is = spex.simulate_iSPEX_error(wavelengths, source, "POL_t", POL_ts)
        DoLPs, AoLPs = spex.retrieve_DoLP_many(wavelengths, source, Is)
        POL_t[i,j] = margin(POL_ts, DoLPs, AoLPs, DoLP_real, AoLP_real)

for arr, label in zip([QWP_d, QWP_t, MOR1_d, MOR1_t, POL_t], ["QWP_d", "QWP_t", "MOR1_d", "MOR1_t", "POL_t"]):
    plot(arr, label)
    np.save(f"margins_{label}.npy", arr)
