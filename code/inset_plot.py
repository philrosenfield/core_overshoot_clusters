"""plotting functions for 1 main panel and 2 inset panels"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np


def add_rect(ax, xlim, ylim, kw=None):
    """caller for ax.add_patch"""
    default = {'fill': False, 'color': 'grey', 'lw': 2, 'zorder': 1000}
    kw = kw or {}
    default.update(kw)
    rect = plt.Rectangle((xlim[0], ylim[0]), np.diff(xlim), np.diff(ylim),
                         **default)
    ax.add_patch(rect)
    return


def setup_zoomgrid():
    """
    Set up a 6 panel plot, 2x2 grid is one main axes and the other 2 are
    zoom-ins of the main axes
    """
    fig = plt.figure(figsize=(8, 6.5))
    # cmd grid is one square taking up 4 of the 6 axes
    ax = plt.subplot2grid((2,3), (0,0), rowspan=2, colspan=2)
    # for HB probably
    ax2 = plt.subplot2grid((2,3), (0,2))
    # for MSTO probably
    ax3 = plt.subplot2grid((2,3), (1,2))

    plt.subplots_adjust(wspace=0.15, right=0.88)
    ax.tick_params(right=False, top=False, bottom=True, left=True)
    for ax_ in [ax2, ax3]:
        ax_.tick_params(labelright=True, labelleft=False,
                        left=False, top=False)
    return fig, (ax, ax2, ax3)


def adjust_zoomgrid(ax, ax2, ax3, zoom1_kw=None, zoom2_kw=None, reversey=True):
    """
    zoom grid is just a fancy call to set_[x,y]lim and plot a
    rectangle.
    """
    def adjust(ax, axz, zoom, reversey=True):
        add_rect(ax, **zoom)
        axz.set_xlim(zoom['xlim'])
        ylim = np.sort(zoom['ylim'])
        if reversey:
            ylim = np.sort(zoom['ylim'])[::-1]
        axz.set_ylim(ylim)
        return ax

    default1 = {'xlim': [0.5, 1.],
                'ylim': [19.6, 21.7]}
    zoom1_kw = zoom1_kw or default1

    adjust(ax, ax2, zoom1_kw, reversey=reversey)
    if zoom2_kw is not None:
        adjust(ax, ax3, zoom2_kw, reversey=reversey)

    for ax_ in [ax2, ax3]:
        ax_.locator_params(axis='x', nbins=4)
        ax_.locator_params(axis='y', nbins=6)

    return [ax, ax2, ax3]
