"""plotting functions related to PDF plots"""
from __future__ import absolute_import, print_function

import os

import matplotlib.pyplot as plt
import numpy as np

from .config import FIGEXT, key2label
from .ssp import SSP


def unify_axlims(axs, bycolumn=True, x=True, y=False):
    """set all axes limits to largest range of axes"""
    if bycolumn:
        axs = axs.T
    for i in range(len(axs)):
        col = axs[i]
        if x:
            l, h = zip(*[a.get_xlim() for a in col])
            [a.set_xlim(np.min(l), np.max(h)) for a in col]
        if y:
            l, h = zip(*[a.get_ylim() for a in col])
            [a.set_ylim(np.min(l), np.max(h)) for a in col]


def fixcorner(fig, axs, ndim):
    """tweak corner plots, add large ylabel, set number of ticks"""
    labelfmt = r'$\rm{{{}}}$'
    for ax in axs.ravel()[:-1 * ndim]:
        ax.tick_params(labelbottom='off', tickdir='in')
        ax.axes.set_xlabel('')
    [ax.axes.set_ylabel('') for ax in axs[:, 0]]
    # dmod hack:
    [ax.locator_params(axis='x', nbins=4) for ax in axs.ravel()]
    [ax.locator_params(axis='x', nbins=3) for ax in axs.T[2]]
    fig.text(0.02, 0.5, labelfmt.format('\ln\ Probability'), ha='center',
             va='center', rotation='vertical')
    fig.subplots_adjust(hspace=0.15, wspace=0.15, right=0.95, top=0.98,
                        bottom=0.15)

    unify_axlims(axs)
    return fig, axs


def cluster_result_plots(sspfns, oned=True, twod=True, onefig=True,
                         gauss=False, quantile=True, mock=False,
                         ovis5=False):
    """
    PDF plots of the results

    Parameters
    ----------
    sspfns : list of strings
        PDF file names
    oned : bool
        Plot only marginalized PDFs (1D)
    twod : bool
        Make a corner plot
    onefig : bool
        Add all 1D plots to a single figure
    gauss : bool
        Fit a 1D gaussian to the PDF
    quantile : bool
        Add quantiles to the marginalized axes
    mock : bool
        sspfns is/are mock data
    ovis5 : bool
        Marginalize over Lambda_c=0.5
    """
    labelfmt = r'$\rm{{{}}}$'
    mstr = ''
    avoid_list = ['sfr', 'fit', 'dmag_min', 'vstep', 'vistep', 'tbin', 'ssp',
                  'trueov', 'dav']

    # This assures the same order on the plots, though they are default in ssp
    marg_cols = ['Av', 'dmod', 'lage', 'logZ', 'ov']
    if mock or ovis5:
        marg_cols = ['Av', 'dmod', 'lage', 'logZ']

    line = ' & '.join(marg_cols) + '\n'

    if mock:
        mstr = '_test'
        ovs = [0.3, 0.4, 0.5, 0.6]
        ssps = [SSP(sspfns[0], gyr=True, filterby={'trueov': ov})
                for ov in ovs]
        name = os.path.splitext(sspfns[0])[0]
        sspfns = ['{0:s}_trueov{1!s}.csv'.format(name, ov) for ov in ovs]
        cmap = plt.cm.Reds
        truth = {'IMF': 1.35,
                 'dmod': 18.45,
                 'Av': 0.1,
                 'dav': 0.0,
                 'dlogZ': 0.1,
                 'bf': 0.3,
                 'dmag_min': -1.5,
                 'vstep': 0.15,
                 'vistep': 0.05,
                 'logZ': -0.40,
                 'sfr': 8e-4,
                 'lagei': 9.1673,
                 'lagef': 9.1847}
    else:
        truth = {}
        ssps = []
        cmap = plt.cm.Blues
        for sspfn in sspfns:
            if ovis5:
                ssp = SSP(sspfn, gyr=True, filterby={'ov': 0.50})
                cmap = plt.cm.Greens
                mstr = '_ov5'
            else:
                ssp = SSP(sspfn, gyr=True)

            ssps.append(ssp)

    nssps = len(ssps)
    ndim = len(marg_cols)
    fig = None
    axs = [None] * nssps
    if onefig:
        fig, axs = plt.subplots(nrows=nssps, ncols=ndim,
                                figsize=(ndim * 1.7, nssps))

    for i, ssp in enumerate(ssps):
        ssp.uniq_grid(skip_cols=avoid_list)
        sspfn = sspfns[i]
        targ = ssp.name.split('_')[0].upper()
        label = labelfmt.format(targ)
        ylabel = labelfmt.format(targ)
        if onefig or mock:
            label = None

        if oned:
            f, raxs = pdf_plots(ssp, marginals=marg_cols, text=label,
                                axs=axs[i], quantile=True, fig=fig,
                                gauss1D=gauss, truth=truth)
            if 'lagei' in list(truth.keys()):
                j = marg_cols.index('lage')
                agei = truth['lagei']
                agef = truth['lagef']
                if ssp.gyr:
                    agei = 10 ** (agei - 9)
                    agef = 10 ** (agef - 9)
                raxs[j].fill_betweenx(np.linspace(*raxs[j].get_ylim()), agei,
                                      agef, color='darkred', zorder=0)
            if not onefig:
                figname = sspfn.replace('.csv', '_1d.pdf')
                plt.savefig(figname)
                plt.close()
            else:
                if mock:
                    raxs[-1].set_ylabel(r'$\Lambda_c={!s}$'.format(ovs[i]),
                                        color='darkred')
                    targ = '{!s}'.format(ovs[i])
                else:
                    raxs[-1].set_ylabel(ylabel)
                raxs[-1].yaxis.set_label_position("right")

        if twod:
            pdf_plots(ssp, marginals=marg_cols, twod=True, quantile=True,
                      cmap=cmap, gauss1D=gauss)
            figname = os.path.split(sspfn)[1].replace(
                '.csv', '{}{}'.format(mstr, FIGEXT))
            plt.savefig(figname)
            print('wrote {0:s}'.format(figname))
            plt.close()

        gs = [ssp.__getattribute__(k + 'g') for k in marg_cols]
        fmt = r'${0:.3f}^{{+{1:.3f}}}_{{-{2:.3f}}}$'
        line += targ + '& '
        try:
            gs[0].mean
            line += ' &  '.join(
                ['{:.3f} & {:.3f}'.format(g.mean / 1., g.stddev / 2.)
                 for g in gs])
        except (AttributeError, TypeError):
            try:
                gs[0][2]
            except TypeError:
                gs = [ssp.__getattribute__('{0:s}q'.format(q))
                      for q in marg_cols]
            j = marg_cols.index('logZ')
            gs[j] = 0.01524 * 10 ** gs[j]
            line += ' &  '.join([fmt.format(g[2], g[1] -
                                            g[2], g[2] - g[0]) for g in gs])

        line += r'\\'
        line += '\n'

    outtab = 'pdf{}.tex'.format(mstr)
    with open(outtab, 'w') as out:
        out.write(line)
    print('wrote {0:s}'.format(outtab))

    if onefig:
        fig, axs = fixcorner(fig, axs, ndim)
        figname = 'combo_{}_ssp{}s{}'.format(nssps, mstr, FIGEXT)
        plt.savefig(figname)
        plt.close()

    return


def fix_diagonal_axes(raxs, ndim):
    """
    set diagonal xaxis limits to that of the off diagonal xaxis limits.

    With corner_meshgrid, the axes limits for the 2d plots will be slightly
    different than the 1d plots.
    """
    nplots = len(raxs)
    idiag = [i * (ndim + 1) for i in range(nplots // (ndim + 1))]
    [raxs[i].set_xlim(raxs[i + 1].get_xlim()) for i in idiag]
    [raxs[nplots - 1].set_xlim(raxs[ndim - 1].get_ylim()) for i in idiag]
    return


def add_inner_title(ax, title, loc, size=None):
    '''
    add a label to an axes as if it were a legend (loc must be 1-11)
        'upper right'     1
        'upper left'      2
        'lower left'      3
        'lower right'     4
        'right'           5
        'center left'     6
        'center right'    7
        'lower center'    8
        'upper center'    9
        'center'          10
    '''
    from matplotlib.patheffects import withStroke
    from matplotlib.offsetbox import AnchoredText

    if size is None:
        size = dict(size=plt.rcParams['legend.fontsize'])
    anct = AnchoredText(title, loc=loc, prop=size, pad=0., borderpad=0.5,
                        frameon=False)
    ax.add_artist(anct)
    anct.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
    anct.patch.set_alpha(0.5)
    return anct


def corner_setup(ndim):
    """
    Setup a corner plot
    Only keep ticklabels on outter left and bottom plots, only set visible
    lower triangle of the plot grid.

    ndim: int
        number of rows (= number of columns)
    Returns
    fig, axs : output of plt.subplots
    """
    fig, axs = plt.subplots(nrows=ndim, ncols=ndim,
                            figsize=(ndim * 2, ndim * 2))
    # Turn off ticks on diagonal
    offs = ['top', 'right', 'left', 'labelleft']
    lnp_ticks = dict(zip(offs, ['off'] * len(offs)))
    [axs[k, k].tick_params(**lnp_ticks) for k in range(ndim)]

    # Turn off bottom tick labels on all but bottom axes
    [ax.tick_params(labelbottom='off') for ax in axs[:ndim - 1, :].ravel()]

    # Turn off left tick labels on all but left axes
    [ax.tick_params(labelleft='off') for ax in axs[:, 1:].ravel()]

    # Turn off upper triangle axes
    [[ax.set_visible(False) for ax in axs[k, k + 1:]] for k in range(ndim)]

    if ndim > 2:
        fig.subplots_adjust(left=0.1, bottom=0.1, top=0.95, right=0.95)
    return fig, axs


def add_quantiles(SSP, ax, attrs, uvalss=None, probs=None,
                  twod=False, gauss=False):
    """
    Add some lines!

    Parameters
    ----------
    SSP : ssp.SSP instance
    ax : plt.axes instance
        add lines/points to this plot
    attrs: 1 or 2D str array
        (marginalized) SSP.data columns to plot
        NB: attrs must go [xattr, yattr] for 2d plotting
    uvalls: 1 or 2D float array
        unique values of attrs
    probs: 1 or 2D float array
        marginalized probabilities corresponding to attrs
    twod:
        add lines
    gauss: bool (False)
        fit or access gaussian fit of the attribute(s) and add lines
        for the mean and +/- stdev / 2
        or (default)
        add 16 and 84 percentile as well as maximum posterior probability.
        if twod, the mean or max post. prob will be plotted as point.

    Returns
    -------
    ax : plt.axes instance
    (may add attributes to SSP if quantiles or fittgauss1D had not been called)
    """
    attrs = np.atleast_1d(attrs)
    if uvalss is None:
        uvalss = [None] * len(attrs)
    else:
        uvalss = np.atleast_1d(uvalss)

    if probs is None:
        probs = [None] * len(attrs)
    else:
        probs = np.atleast_1d(probs)

    pltkw = {'color': 'k'}
    # why order matteres: xattr yattr will plot vline and hline
    linefuncs = [ax.axvline, ax.axhline]
    # mean or max post. prob to plot as points
    pts = []
    for i, (attr, uvals, prob) in enumerate(zip(attrs, uvalss, probs)):
        if attr is None:
            # for second loop 1D
            continue
        # look up value
        gatr = '{:s}g'.format(attr)
        if not hasattr(SSP, gatr):
            if gauss:
                # fit 1D Gaussian
                g = SSP.fitgauss1D(attr, uvals, prob)
            else:
                # go with quantiles (default 0.16, 0.84)
                # g = SSP.quantiles(attr, uvals, prob, maxp=True, k=1, ax=ax)
                g = SSP.quantiles(attr, uvals, prob, maxp=True, k=1)
        else:
            g = SSP.__getattribute__(gatr)

        if gauss:
            lines = [g.mean, g.mean + g.stddev / 2, g.mean - g.stddev / 2]
        else:
            # if maxp=False when SSP.quantiles called
            # this will raise a value error because g will be length 2.
            lines = [g[2], g[0], g[1]]
        lstys = ['-', '--', '--']

        if twod:
            if gauss:
                lines = [g.mean + g.stddev / 2, g.mean - g.stddev / 2]
                pts.append(g.mean)
            else:
                lines = [g[0], g[1]]
                pts.append(g[2])
            lstys = ['--', '--']

        # plot.
        [linefuncs[i](l, ls=ls, **pltkw) for (l, ls) in zip(lines, lstys)]

    if twod:
        # plot mean or max post prob
        ax.plot(pts[0], pts[1], 'o', color='white', mec='k', mew=1)
    else:
        if gauss:
            # over plot Gaussian fit
            X = uvalss[0]
            prob = probs[0]
            # plot the Gaussian 10 steps beyond the calculated limits.
            dx = 10 * np.diff(X)[0]
            xx = np.linspace(X.min() - dx, X.max() + dx, 100)
            ax.plot(xx, g(xx), color='darkred')
    return ax


def pdf_plot(SSP, xattr, yattr=None, ax=None, sub=None, save=False,
             truth=None, cmap=None, plt_kw=None, X=None, prob=None,
             logp=True, quantile=False, gauss1D=False):
    """Plot -2 ln P vs marginalized attributes

    SSP : SSP class instance

    xattr, yattr : str
        column names to marginalize and plot

    ax : plt.Axes
        add plot to this axis

    sub : str
        if save, add this string to the filename

    save : bool
        save the plot to file with axes lables.

    truth : dict
        truth dictionary with attributes as keys and truth values as values
        overplot as hline and vline on axes.

    cmap : cm.cmap instance
        if yattr isn't None, call plt.pcolor with this cmap.

    plt_kw : dict
        if yattr is None, pass these kwargs to plt.plot

    Returns
    ax : plt.Axes
        new or updated axes instance
    """
    plt_kw = plt_kw or {'lw': 4, 'color': 'k'}
    cmap = cmap or plt.cm.viridis_r
    sub = sub or ''
    truth = truth or {}
    do_cbar = False
    pstr = ''
    if logp:
        pstr = '\ln\ '

    if ax is None:
        _, ax = plt.subplots()
        if yattr is not None:
            do_cbar = True

    if yattr is None:
        # plot type is marginal probability. Attribute vs -2 ln P
        if X is None and prob is None:
            if not SSP.vdict[xattr]:
                return ax
            X, prob = SSP.marginalize(xattr, log=logp)
        line = ax.plot(X, prob, **plt_kw)

        if gauss1D or quantile:
            ax = add_quantiles(SSP, ax, xattr, uvalss=[X], probs=[prob],
                               gauss=gauss1D)
        ax.set_xlim(X.min(), X.max())
        # yaxis max is the larger of 10% higher than the max val or current
        # ylim.
        ymax = np.max([prob.max() + (prob.max() * 0.1), ax.get_ylim()[1]])
        ax.set_ylim(prob.min(), ymax)

        if save:
            ptype = 'marginal'
            # ax.set_ylabel(key2label('fit'))
            ax.set_ylabel(key2label(pstr + 'Probability'))
    else:
        # plot type is joint probability.
        # Attribute1 vs Attribute2 colored by fit
        if not SSP.vdict[xattr] or not SSP.vdict[yattr]:
            return ax
        [X, Y], prob = SSP.marginalize(xattr, yattr=yattr, log=logp)
        line = ax.pcolor(X, Y, prob, cmap=cmap)
        if gauss1D or quantile:
            add_quantiles(SSP, ax, [xattr, yattr], twod=True, gauss=gauss1D)
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())

        if do_cbar:
            cbar = plt.colorbar(line)
            # cbar.set_label(key2label('fit'))
            cbar.set_label(key2label(pstr + 'Probability'))

        if save:
            ptype = '{}_joint'.format(yattr)
            ax.set_ylabel(key2label(yattr, gyr=SSP.gyr))

        if yattr in truth:
            ax.axhline(truth[yattr], color='k', lw=3)

    if xattr in truth:
        ax.axvline(truth[xattr], color='k', lw=3)

    if save:
        ax.set_xlabel(key2label(xattr, gyr=SSP.gyr))
        # add subdirectory to filename
        if len(sub) > 0:
            sub = '_' + sub
        outfmt = '{}_{}{}_{}{}'
        outname = outfmt.format(SSP.name.replace('.csv', ''),
                                xattr, sub, ptype, EXT)
        plt.savefig(outname, bbox_inches='tight')
        print('wrote {}'.format(outname))
        plt.close()
    return ax


def pdf_plots(SSP, marginals=None, sub=None, twod=False, truth=None,
              text=None, cmap=None, fig=None, axs=None,
              logp=True, gauss1D=False, quantile=False):
    """Call pdf_plot for a list of xattr and yattr"""
    text = text or ''
    sub = sub or ''
    truth = truth or {}
    marginals = marginals or SSP._getmarginals()
    pstr = ''
    plkw = {'logp': logp, 'gauss1D': gauss1D, 'quantile': quantile}

    if logp:
        pstr = '\ln\ '

    if not hasattr(SSP, 'vdict'):
        SSP.uniq_grid()
    valid_margs = [k for (k, v) in list(SSP.vdict.items()) if v]
    ndim = len(marginals)
    if ndim != len(valid_margs):
        bad_margs = [m for m in marginals if m not in valid_margs]
        marginals = [m for m in marginals if m in valid_margs]
        print('Warning: {} does not vary and will be skipped.'
              .format(bad_margs))
        ndim = len(marginals)

    raxs = []
    if twod:
        fig, axs = corner_setup(ndim)
        for c, mx in enumerate(marginals):
            for r, my in enumerate(marginals):
                ax = axs[r, c]
                if r == c:
                    # diagonal
                    # my = 'fit'  # my is reset for ylabel call
                    my = pstr + 'Probability'
                    raxs.append(pdf_plot(SSP, mx, ax=ax, truth=truth, **plkw))
                else:
                    # off-diagonal
                    raxs.append(pdf_plot(SSP, mx, yattr=my, ax=ax, truth=truth,
                                         cmap=cmap, **plkw))

                if c == 0:
                    # left most column
                    ax.set_ylabel(key2label(my))

                if r == ndim - 1:
                    # bottom row
                    ax.set_xlabel(key2label(mx))
        [ax.locator_params(axis='y', nbins=6) for ax in axs.ravel()]
        fix_diagonal_axes(raxs, ndim)
    else:
        if fig is None and axs is None:
            fig, axs = plt.subplots(
                ncols=ndim, figsize=(ndim * 3., ndim * 0.6))
        [ax.tick_params(left='off', labelleft='off', right='off', top='off')
         for ax in axs]
        X = None
        prob = None
        for i in marginals:
            ax = axs[marginals.index(i)]
            ax = pdf_plot(SSP, i, truth=truth, ax=ax, X=X, prob=prob, **plkw)
            ax.set_xlabel(key2label(i, gyr=SSP.gyr))
            raxs.append(ax)

        if text:
            add_inner_title(raxs[-1], '${}$'.format(text), 3, size=None)
        fig.subplots_adjust(bottom=0.22, left=0.05)
        raxs[0].set_ylabel(key2label('Probability'))
    [ax.locator_params(axis='x', nbins=5) for ax in axs.ravel()]
    return fig, raxs
