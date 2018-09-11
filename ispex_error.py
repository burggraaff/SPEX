import pol
import numpy as np
from matplotlib import pyplot as plt
import spex

wavelengths = np.arange(297, 802, 0.3)
source = pol.Stokes_nm(np.ones_like(wavelengths), 0., 1., 0.)
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
    ax2.set_ylabel("AoLP")
    ax.set_title(title)
    return fig

def _lims2(ax, ax2, x):
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-0.2, 0.2)
    ax2.set_ylim(-20, 20)

def _ticks2(ax, ax2):
    ax.set_yticks(np.arange(-0.2, 0.25, 0.05), minor=False)
    ax.tick_params(axis="y", which="minor", length=0)
    ax2.set_yticks(np.arange(-20, 25, 5))
    ax.yaxis.grid(True, which="major")
    ax.xaxis.grid(True)

def plot_errors(x, DoLPs, AoLPs, real_DoLP, real_AoLP, xlabel, Dlim=0.05, title="Solid: DoLP ; Dashed: AoLP"):
    fig, ax = plt.subplots(figsize=(6,4), tight_layout=True)
    ax2 = ax.twinx()
    ax.plot(x, DoLPs - real_DoLP, c='k')
    ax.axhspan(-Dlim, Dlim, color='0.5', alpha=0.5)
    ax2.plot(x, AoLPs - real_AoLP, c='k', ls="--")
    _lims2(ax , ax2, x)
    _ticks2(ax, ax2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("DoLP Error")
    ax2.set_ylabel("AoLP Error")
    ax.set_title(title)
    return fig

I, Q, U, V = spex.simulate_iSPEX(wavelengths, source)
DoLP_real = pol.DoLP(*source[0]) ; AoLP_real = pol.AoLP_deg(*source[0])
print(f"Real: DoLP = {DoLP_real:.2f}, AoLP = {AoLP_real:.1f} degrees")
DoLP, AoLP = spex.retrieve_DoLP(wavelengths, source, I)
print(f"Optimal: DoLP = {DoLP:.2f}, AoLP = {AoLP:.1f} degrees")

title = f"DoLP (solid, {DoLP_real:.2f}) and AoLP (dashed, {AoLP_real:.1f})"

print("Simulated ideal conditions")
QWP_ds = np.linspace(-20, 20, 100)
Is = spex.simulate_iSPEX_error(wavelengths, source, "QWP_d", QWP_ds)
print(f"Simulated QWP {D}d")
DoLPs, AoLPs = spex.retrieve_DoLP_many(wavelengths, source, Is)
fig = plot_dolp_aolp(QWP_ds, DoLPs, AoLPs, r"$\Delta d$ on QWP (nm)", title=title)
plt.show()
fig = plot_errors(QWP_ds, DoLPs, AoLPs, DoLP_real, AoLP_real, r"$\Delta d$ on QWP (nm)")
plt.show()

QWP_ts = np.linspace(-25, 25, 100)
Is = spex.simulate_iSPEX_error(wavelengths, source, "QWP_t", QWP_ts)
print(f"Simulated QWP {D}{a}")
DoLPs, AoLPs = spex.retrieve_DoLP_many(wavelengths, source, Is)
fig = plot_dolp_aolp(QWP_ts, DoLPs, AoLPs, r"$\Delta \alpha$ on QWP (degrees)", title=title)
plt.show()
fig = plot_errors(QWP_ts, DoLPs, AoLPs, DoLP_real, AoLP_real, r"$\Delta \alpha$ on QWP (degrees)")
plt.show()

MOR1_ds = np.linspace(-150, 150, 150)
Is = spex.simulate_iSPEX_error(wavelengths, source, "MOR1_d", MOR1_ds)
print(f"Simulated MOR 1 {D}d")
DoLPs, AoLPs = spex.retrieve_DoLP_many(wavelengths, source, Is)
fig = plot_dolp_aolp(MOR1_ds, DoLPs, AoLPs, r"$\Delta d$ on MOR 1 (nm)", title=title)
plt.show()
fig = plot_errors(MOR1_ds, DoLPs, AoLPs, DoLP_real, AoLP_real, r"$\Delta d$ on MOR 1 (nm)")
plt.show()

MOR1_ts = np.linspace(-25, 25, 100)
Is = spex.simulate_iSPEX_error(wavelengths, source, "MOR1_t", MOR1_ts)
print(f"Simulated MOR 1 {D}{a}")
DoLPs, AoLPs = spex.retrieve_DoLP_many(wavelengths, source, Is)
fig = plot_dolp_aolp(MOR1_ts, DoLPs, AoLPs, r"$\Delta \alpha$ on MOR 1 (degrees)", title=title)
plt.show()
fig = plot_errors(MOR1_ts, DoLPs, AoLPs, DoLP_real, AoLP_real, r"$\Delta \alpha$ on MOR 1 (degrees)")
plt.show()

POL_ts = np.linspace(-15, 15, 100)
Is = spex.simulate_iSPEX_error(wavelengths, source, "POL_t", POL_ts)
print(f"Simulated POL {D}{a}")
DoLPs, AoLPs = spex.retrieve_DoLP_many(wavelengths, source, Is)
fig = plot_dolp_aolp(POL_ts, DoLPs, AoLPs, r"$\Delta \alpha$ on POL (degrees)", title=title)
plt.show()
fig = plot_errors(POL_ts, DoLPs, AoLPs, DoLP_real, AoLP_real, r"$\Delta \alpha$ on POL (degrees)")
plt.show()
