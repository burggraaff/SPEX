import pol
import numpy as np
from matplotlib import pyplot as plt
import spex
from sys import argv

Q_in = float(argv[1])
try:
    U_in = float(argv[2])
except:
    U_in = 0.

wavelengths = np.arange(450, 700, 0.3)
source = pol.Stokes_nm(np.ones_like(wavelengths), Q_in, U_in, 0.)
D = "\u0394"
a = "\u03B1"

def _lims(ax, ax2, x):
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-0.01, 1.01)
    ax2.set_ylim(-91.8, 91.8)

def _ticks(ax, ax2):
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], minor=True)
    ax.set_yticks(np.arange(0, 1.1, 0.1), minor=False)
    ax.tick_params(axis="y", which="minor", length=0)
    ax2.set_yticks(np.arange(-90, 90+15, 15))
    ax.yaxis.grid(True, which="minor")
    ax.xaxis.grid(True)

def plot_dolp_aolp(x, DoLPs, AoLPs, xlabel, title="Solid: DoLP ; Dashed: AoLP"):
    fig, ax = plt.subplots(figsize=(6,4), tight_layout=True)
    ax2 = ax.twinx()
    ax.plot(x, DoLPs, c='k')
    ax2.plot(x, AoLPs, c='k', ls="--")
    _lims(ax , ax2, x)
    _ticks(ax, ax2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("DoLP")
    ax2.set_ylabel("AoLP (degrees)")
    ax.set_title(title)
    return fig

def _lims2(ax, ax2, x):
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-0.12, 0.12)
    ax2.set_ylim(-20, 20)

def _ticks2(ax, ax2):
    ax.set_yticks(np.arange(-0.12, 0.14, 0.03), minor=False)
    ax.tick_params(axis="y", which="minor", length=0)
    ax2.set_yticks(np.arange(-20, 25, 5))
    ax.yaxis.grid(True, which="major")
    ax.xaxis.grid(True)

def _D_err(D, D_real):
    return D/D_real - 1

def _A_err(A, A_real):
    diff = A - A_real
    if A_real >= 0:
        diff[A < 0] = A[A < 0] + 180 - A_real
    else:
        diff[A > 0] = A[A > 0] - 180 - A_real
    return diff

def plot_errors(x, DoLPs, AoLPs, real_DoLP, real_AoLP, xlabel, Dlim=0.03, title="Solid: DoLP ; Dashed: AoLP"):
    fig, ax = plt.subplots(figsize=(6,4), tight_layout=True)
    ax2 = ax.twinx()
    Derr = _D_err(DoLPs, real_DoLP)
    Aerr = _A_err(AoLPs, real_AoLP)
    ax.plot(x, Derr, c='k')
    ax.axhspan(-Dlim, Dlim, color='0.5', alpha=0.5)
    ax2.plot(x, Aerr, c='k', ls="--")
    _lims2(ax , ax2, x)
    _ticks2(ax, ax2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("DoLP Error (%)")
    ax2.set_ylabel("AoLP Error (degrees)")
    ax.set_title(title)
    return fig

def margin(x, DoLPs, AoLPs, real_DoLP, real_AoLP, Dlim=0.03, Alim=5):
    D_err = _D_err(DoLPs, real_DoLP)
    A_err = _A_err(AoLPs, real_AoLP)

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

I0, I90 = spex.simulate_iSPEX2(wavelengths, source)
DoLP_real = pol.DoLP(*source[0]) ; AoLP_real = pol.AoLP_deg(*source[0])
print(f"Real: DoLP = {DoLP_real:.2f}, AoLP = {AoLP_real:.1f} degrees")
DoLP0 , AoLP0  = spex.retrieve_DoLP(wavelengths, source, I0 )
DoLP90, AoLP90 = spex.retrieve_DoLP(wavelengths, source, I90)
AoLP90 = spex._correct_AoLP90(AoLP90)
print(f"+Q: DoLP = {DoLP0 :.2f}, AoLP = {AoLP0 :.1f} degrees")
print(f"-Q: DoLP = {DoLP90:.2f}, AoLP = {AoLP90:.1f} degrees")

title = f"DoLP (solid, {DoLP_real:.2f}) and AoLP (dashed, {AoLP_real:.1f})"

print("Simulated ideal conditions")
QWP_ds = np.linspace(-20, 20, 100)
I0, I90 = spex.simulate_iSPEX2_error(wavelengths, source, "QWP_d", QWP_ds)
print(f"Simulated QWP {D}d")
DoLPs, AoLPs = spex.retrieve_DoLP_many2(wavelengths, source, I0, I90)
fig = plot_dolp_aolp(QWP_ds, DoLPs, AoLPs, r"$\Delta d$ on QWP (nm)", title=title)
plt.show()
fig = plot_errors(QWP_ds, DoLPs, AoLPs, DoLP_real, AoLP_real, r"$\Delta d$ on QWP (nm)")
fig.savefig(f"single_margin_Q{Q_in:.1f}_U{U_in:.1f}_QWPd.png")
plt.show()

QWP_ts = np.linspace(-25, 25, 100)
I0, I90 = spex.simulate_iSPEX2_error(wavelengths, source, "QWP_t", QWP_ts)
print(f"Simulated QWP {D}{a}")
DoLPs, AoLPs = spex.retrieve_DoLP_many2(wavelengths, source, I0, I90)
fig = plot_dolp_aolp(QWP_ts, DoLPs, AoLPs, r"$\Delta \alpha$ on QWP (degrees)", title=title)
plt.show()
fig = plot_errors(QWP_ts, DoLPs, AoLPs, DoLP_real, AoLP_real, r"$\Delta \alpha$ on QWP (degrees)")
fig.savefig(f"single_margin_Q{Q_in:.1f}_U{U_in:.1f}_QWPt.png")
plt.show()

MOR1_ds = np.linspace(-150, 150, 150)
I0, I90 = spex.simulate_iSPEX2_error(wavelengths, source, "MOR1_d", MOR1_ds)
print(f"Simulated MOR 1 {D}d")
DoLPs, AoLPs = spex.retrieve_DoLP_many2(wavelengths, source, I0, I90)
fig = plot_dolp_aolp(MOR1_ds, DoLPs, AoLPs, r"$\Delta d$ on MOR 1 (nm)", title=title)
plt.show()
fig = plot_errors(MOR1_ds, DoLPs, AoLPs, DoLP_real, AoLP_real, r"$\Delta d$ on MOR 1 (nm)")
fig.savefig(f"single_margin_Q{Q_in:.1f}_U{U_in:.1f}_MOR1d.png")
plt.show()

MOR1_ts = np.linspace(-25, 25, 100)
I0, I90 = spex.simulate_iSPEX2_error(wavelengths, source, "MOR1_t", MOR1_ts)
print(f"Simulated MOR 1 {D}{a}")
DoLPs, AoLPs = spex.retrieve_DoLP_many2(wavelengths, source, I0, I90)
fig = plot_dolp_aolp(MOR1_ts, DoLPs, AoLPs, r"$\Delta \alpha$ on MOR 1 (degrees)", title=title)
plt.show()
fig = plot_errors(MOR1_ts, DoLPs, AoLPs, DoLP_real, AoLP_real, r"$\Delta \alpha$ on MOR 1 (degrees)")
fig.savefig(f"single_margin_Q{Q_in:.1f}_U{U_in:.1f}_MOR1t.png")
plt.show()

POL_ts = np.linspace(-15, 15, 100)
I0, I90 = spex.simulate_iSPEX2_error(wavelengths, source, "POL0_t", POL_ts)
print(f"Simulated POL0 {D}{a}")
DoLPs, AoLPs = spex.retrieve_DoLP_many2(wavelengths, source, I0, I90)
fig = plot_dolp_aolp(POL_ts, DoLPs, AoLPs, r"$\Delta \alpha$ on POL0 (degrees)", title=title)
plt.show()
fig = plot_errors(POL_ts, DoLPs, AoLPs, DoLP_real, AoLP_real, r"$\Delta \alpha$ on POL0 (degrees)")
fig.savefig(f"single_margin_Q{Q_in:.1f}_U{U_in:.1f}_POL0t.png")
plt.show()

I0, I90 = spex.simulate_iSPEX2_error(wavelengths, source, "POL90_t", POL_ts)
print(f"Simulated POL90 {D}{a}")
DoLPs, AoLPs = spex.retrieve_DoLP_many2(wavelengths, source, I0, I90)
fig = plot_dolp_aolp(POL_ts, DoLPs, AoLPs, r"$\Delta \alpha$ on POL90 (degrees)", title=title)
plt.show()
fig = plot_errors(POL_ts, DoLPs, AoLPs, DoLP_real, AoLP_real, r"$\Delta \alpha$ on POL90 (degrees)")
fig.savefig(f"single_margin_Q{Q_in:.1f}_U{U_in:.1f}_POL90t.png")
plt.show()
