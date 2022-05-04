"""
groundSPEX plotting functions
"""
import functools
from matplotlib import pyplot as plt
import numpy as np
import spectacle
from spectacle.plot import RGB_OkabeIto as RGB, save_or_show


def new_or_existing_figure(func):
    """
    Decorator that handles the choice between creating a new figure or plotting in an existing one.
    Checks if an Axes object was given - if yes, use that - if no, create a new one.
    In the "no" case, save/show the resulting plot at the end.
    This was copypasted in from `smartphone-water-colour` and will be refactored into SPECTACLE soon.
    """
    @functools.wraps(func)
    def newfunc(*args, ax=None, title=None, figsize=(5, 3), figure_kwargs={}, saveto=None, dpi=300, bbox_inches="tight", **kwargs):
        # If no Axes object was given, make a new one
        if ax is None:
            newaxes = True
            plt.figure(figsize=figsize, **figure_kwargs)
            ax = plt.gca()
        else:
            newaxes = False

        # Plot everything as normal
        func(*args, ax=ax, **kwargs)

        # If this is a new plot, add a title and save/show the result
        if newaxes:
            ax.set_title(title)
            save_or_show(saveto, dpi=dpi, bbox_inches=bbox_inches)

    return newfunc


@new_or_existing_figure
def plot_spectrum_stack(wavelengths, spectra, colour="k", alpha=0.1, rasterized=True, xlabel="Wavelength [nm]", ylabel="Radiance [ADU]", title=None, ax=None, saveto=None):
    """
    Plot a stack of spectra.
    `wavelengths` can also be pixels or something similar.
    This can be done into an existing Axes object (`ax=...`) or a new one.
    """
    # Plot the data
    ax.plot(wavelengths, spectra.T, c=colour, alpha=alpha, rasterized=rasterized)

    # Plot settings
    ax.set_xlim(wavelengths[0], wavelengths[-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(ls="--")


def plot_spectrum_stack_dualchannel(wavelengths, spectra, title=None, saveto=None, dpi=300, **kwargs):
    """
    Plot two stacks of spectra, one for each groundSPEX channel.
    This simply repeats `plot_spectrum_stack` twice, so please see its documentation.
    """
    # Create a new figure
    fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(5, 5), tight_layout=True)

    # Plot the data
    for j, ax in enumerate(axs):
        plot_spectrum_stack(wavelengths, spectra[...,j,:], ax=ax, **kwargs)
        ax.set_title(f"Channel {j}")

    # Plot settings
    axs[0].set_xlabel(None)
    fig.suptitle(title)

    # Save or show the result
    save_or_show(saveto, dpi=dpi)
