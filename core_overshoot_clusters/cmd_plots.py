"""Functions supporting CMD related plots."""
from __future__ import absolute_import, print_function

import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from .config import FIGEXT, cmd_limits
from .inset_plot import adjust_zoomgrid, setup_zoomgrid


def parse_pipeline(filename):
    """Find target and filters from the filename."""
    import re
    name = os.path.split(filename)[1].upper()

    # filters are assumed to be F???W
    starts = np.array([m.start() for m in re.finditer('_F', name)])
    starts += 1
    if len(starts) == 1:
        starts = np.append(starts, starts + 6)
    filters = [name[s: s + 5] for s in starts]

    # the target name is assumed to be before the filters in the filename
    pref = name[:starts[0] - 1]
    for pre in pref.split('_'):
        if pre == 'IR':
            continue
        try:
            # this could be the proposal ID
            int(pre)
        except ValueError:
            # a mix of str and int should be the target
            target = pre
    return target, filters


def load_obs(filename, filter1, filter2, xyfile=None, fextra='VEGA',
             crowd=None):
    """
    Load observations.

    Parameters
    ----------
    filename : string
        path to observation file.
        Can be binary fits table (.fits),
        asteca membership probability file (.dat or .asteca), or
        match photometry file (.match)
    filter1, filter2 : string, string (needed for fits reader)
        column name of mag1, mag2
    fextra : string (needed for fits reader)
        use if column name of mag1, mag2 is prefix. For example,
        fextra='VEGA' if F475W_VEGA is the mag1 column name but
        F475W_CROWD or other columns exist.
    xyfile : string
        optional file with x,y or ra, dec information if not in observation
    crowd : float or None (only for fits reader)
        cull observation file to this value of crowd or below.

    Returns
    -------
    color, mag : np.array, np.array
        mag1-mag2, mag1
    color_err, mag_err : np.array, np.array
        quadriture mag1, mag2 err, mag1_err
    good : np.array
        indices of color, mag where color, mag < 30.
    x, y : np.array, np.array
        x, y from the file
    """
    if xyfile is not None:
        _, _, x, y = np.loadtxt(xyfile, unpack=True)
    else:
        x = np.array([])
        y = np.array([])

    if filename.endswith('fits'):
        try:
            data = fits.getdata(filename)
            keyfmt = '{}_{}'
            errfmt = '{}_ERR'
            if crowd is not None:
                crd = keyfmt.format(filter2, 'CROWD')
                crd1 = keyfmt.format(filter1, 'CROWD')
                inds, = np.nonzero((data[crd] < crowd) & (data[crd1] < crowd))
                data = data[inds]
            mag2 = data[keyfmt.format(filter2, fextra)]
            mag = data[keyfmt.format(filter1, fextra)]
            color = mag - mag2
            mag_err = data[errfmt.format(filter1)]
            color_err = \
                np.sqrt(data[errfmt.format(filter1)] ** 2 + mag_err ** 2)
            x = data.X
            y = data.Y
        except ValueError:
            print('Problem with {}'.format(filename))
            return None, None
    elif filename.endswith('match'):
        mag1, mag2 = np.genfromtxt(filename, unpack=True)
        color = mag1 - mag2
        mag = mag1
        mag_err = None
        color_err = None
    elif filename.endswith('dat'):
        try:
            _, x, y, mag, mag_err, color, color_err, _, _ = \
                np.loadtxt(filename, unpack=True)
        except ValueError:
            print("Can't understand file format {}".format(filename))
            return None, None
    else:
        _, x, y, mag, mag_err, color, color_err = \
            np.loadtxt(filename, unpack=True)

    good, = np.nonzero((np.abs(color) < 30) & (np.abs(mag) < 30))
    return color, mag, color_err, mag_err, good, x, y


def _plot_cmd(color, mag, color_err=None, mag_err=None, inds=None, ax=None,
              plt_kw=None, star_by_starerr=False):
    """
    Plot a cmd with uncertainties.

    Parameters
    ----------
    color, mag : np.arrays
        CMD data
    color_err, mag_err : np.arrays
        CMD uncertainties
    inds: np.array
        slice of color, mag to plot
    ax : plt.axes instance
        axes to plot to
    plt_kw : dict
        optional kwargs passed to plt.plot
    star_by_starerr: bool
        True: plot each star's uncertainty.
        False: plot mean errors as a function of mag. (call median_err)

    Returns
    -------
    ax : plt.axes instance
    """
    if inds is None:
        inds = np.arange(len(mag))

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 12))

    plt_kw = plt_kw or {}
    default = {'color': 'black', 'ms': 4, 'rasterized': True}
    default.update(plt_kw)

    ax.plot(color[inds], mag[inds], '.', **default)

    if color_err is not None and mag_err is not None:
        if star_by_starerr:
            ax.errorbar(color[inds], mag[inds], fmt='none',
                        xerr=color_err[inds], yerr=mag_err[inds],
                        capsize=0, ecolor='gray')
        else:
            median_err(color[inds], mag[inds], color_err[inds], mag_err[inds],
                       ax=ax)
    return ax


def median_err(color, mag, color_err, mag_err, ax=None, dmag=1.,
               cplace=None):
    """
    Calculate mean color and mag errors as a function of mag bin.

    (all colors included)

    Parameters
    ----------
    color, mag : np.arrays
        color and mag arrays
    color_err, mag_err : np.arrays
        color and mag error arrays
    ax : plt.ax instance
        will add error bars to this axis and use axes limits to calculate
        mag, color bounds
    dmag : float
        mag bin width to calculate mean errors

    Returns
    -------
    magbins: mag mins to displace errors
    carr: array
        a constant array (cplace) of len(magbins)
    merrs, cerrs: mean mag, color errors
    """
    if ax is None:
        mmin = np.min(mag)
        mmax = np.max(mag)
        if cplace is None:
            cplace = np.median(color) - 2.5 * np.std(color)
    else:
        mmin = np.min(ax.get_ylim())
        mmax = np.max(ax.get_ylim())
        if cplace is None:
            cplace = ax.get_xlim()[0] + 0.25

    magbins = np.arange(np.floor(mmin), np.ceil(mmax), dmag)
    idix = np.digitize(mag, magbins)
    merrs = np.array([np.mean(mag_err[idix == i])
                      for i, _ in enumerate(magbins)])
    merrs[np.isnan(merrs)] = 0
    cerrs = np.array([np.mean(color_err[idix == i])
                      for i, _ in enumerate(magbins)])
    carr = np.repeat(cplace, len(magbins))
    if ax is not None:
        ax.errorbar(carr, magbins, fmt='none', xerr=cerrs, yerr=merrs, lw=1.4,
                    capsize=0, ecolor='k')
    return magbins, carr, merrs, cerrs


def cmd(obs, filter1, filter2, zoom=False, xlim=None, ylim=None,
        fig=None, axs=None, plt_kw=None, zoom1_kw=None, zoom2_kw=None,
        load_obskw=None, plt_kwz=None):
    """
    Plot cmd of data, with two insets (optional).

    Parameters
    ----------
    obs : string
        filename of observation to plot (passed to load_obs)
    load_obskw : dict
        optional kwargs passed to load_obs
    filter1, filter2 : string, string
        filter names for axes labels and passed to load_obs
    zoom : bool
        make insets
    xlim, ylim : list or tuple
        (main) axes limits
    fig, axs : plt.figure, plt.axes instances
        optional places to plot to
    plt_kw : dict
        optional kwargs passed to plt.plot
    zoom1_kw, zoom2_kw : dict, dict
        xlim, ylim axes limits for insets
    plt_kwz : dict
        optional kwargs passed to plt.plot for the inset plots

    Returns
    -------
    fig, axs : figure and axes
    """
    plt_kw = plt_kw or {}
    plt_kwz = plt_kwz or plt_kw.copy()
    load_obskw = load_obskw or {}
    color, mag, color_err, mag_err, good, x, y = \
        load_obs(obs, filter1, filter2, **load_obskw)

    if axs is None:
        if zoom:
            fig, (ax, ax2, ax3) = setup_zoomgrid()
        else:
            fig, ax = plt.subplots(figsize=(12, 12))
    else:
        if zoom:
            ax, ax2, ax3 = axs
        else:
            ax = axs

    ax = _plot_cmd(color, mag, color_err=color_err, mag_err=mag_err, inds=good,
                   ax=ax, plt_kw=plt_kw)

    ax.set_ylabel(r'${}$'.format(filter1))
    ax.set_xlabel(r'${}-{}$'.format(filter1, filter2))
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(np.sort(ylim)[::-1])
    axs = [ax]

    if zoom:
        ax2 = _plot_cmd(color, mag, color_err=color_err, mag_err=mag_err,
                        inds=good, ax=ax2, plt_kw=plt_kwz)
        ax3 = _plot_cmd(color, mag, color_err=color_err, mag_err=mag_err,
                        inds=good, ax=ax3, plt_kw=plt_kwz)
        axs = adjust_zoomgrid(
            ax, ax2, ax3, zoom1_kw=zoom1_kw, zoom2_kw=zoom2_kw)

    return fig, axs


def cmd_plots(clusters, membs):
    """Produce two cmds overlaid one from clusters, one from membs."""
    clusters = np.atleast_1d(clusters)
    membs = np.atleast_1d(membs)
    for i, _ in enumerate(clusters):
        cluster = clusters[i]
        memb = membs[i]
        assert os.path.isfile(memb), '{0:s} not found'.format(memb)
        assert os.path.isfile(cluster), '{0:s} not found'.format(cluster)

        _, filters = parse_pipeline(memb)
        filter1, filter2 = filters
        targ = os.path.split(cluster)[1].split('_')[1]

        cmd_kw = dict({'zoom': True, 'load_obskw': {'crowd': 1.3}},
                      **cmd_limits(targ))

        fig, axs = cmd(cluster, filter1, filter2, plt_kw={'alpha': 0.3},
                       **cmd_kw)

        fig, axs = cmd(memb, filter1, filter2, axs=axs, fig=fig,
                       plt_kw={'color': 'darkred'}, **cmd_kw)

        axs[1].set_title('${}$'.format(targ))

        [ax.ticklabel_format(useOffset=False) for ax in axs]
        outfile = os.path.join(os.getcwd(), '{0:s}_cmd{1:s}'
                               .format(targ, FIGEXT))
        plt.savefig(outfile)
        print('wrote {}'.format(outfile))
        plt.close()
    return
