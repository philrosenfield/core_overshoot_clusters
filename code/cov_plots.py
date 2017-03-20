import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from .config import FIGEXT, TRACKS_LOC
from .track import Track
from .inset_plot import adjust_zoomgrid, setup_zoomgrid
from .utils import get_files, get_dirs


def _joint_plot(size=8, ratio=5, space=0.2):
    """
    Hack edits of seaborn.JointGrid so I can access side panel histograms mpl
    style.
    """
    f = plt.figure(figsize=(size, size))
    gs = plt.GridSpec(ratio + 1, ratio + 1)
    ax_joint = f.add_subplot(gs[1:, :-1])
    ax_marg_x = f.add_subplot(gs[0, :-1], sharex=ax_joint)
    ax_marg_y = f.add_subplot(gs[1:, -1], sharey=ax_joint)
    for ax in [ax_marg_x, ax_marg_y]:
        plt.setp(ax.get_xticklabels(), visible=True)
        ax.yaxis.grid(False)
        ax.tick_params(top=False, left=False, right=False, bottom=False,
                       labelleft=False, labelbottom=False)
    ax_joint.tick_params(top=False, left=True, right=True, bottom=True)
    return f, ax_joint, ax_marg_x, ax_marg_y


def fake_cmds(phots, space=0.2):
    """
    Plot CMDs of mock data with top and side histograms in color and mag
    Also plots a summary file with no side histograms.
    """
    # histogram binning
    try:
        from seaborn.distributions import _freedman_diaconis_bins
    except:
        def _freedman_diaconis_bins(arr):
            return 100

    clp = ['darkred', 'orange', 'darkgreen', 'navy', 'purple']
    cov_strs = ['OV0.3', 'OV0.4', 'OV0.5', 'OV0.6']
    # set up mock CMD figures
    f7, ax_joint7, ax_marg_x7, ax_marg_y7 = _joint_plot()
    f8, ax_joint8, ax_marg_x8, ax_marg_y8 = _joint_plot()
    f9, ax_joint9, ax_marg_x9, ax_marg_y9 = _joint_plot()
    fs = [f7, f8, f9]
    ax_joints = [ax_joint7, ax_joint8, ax_joint9]
    ax_marg_xs = [ax_marg_x7, ax_marg_x8, ax_marg_x9]
    ax_marg_ys = [ax_marg_y7, ax_marg_y8, ax_marg_y9]
    # set up summary figure
    f, ax = plt.subplots(figsize=(8,8))

    for p in phots:
        icov, = [j for j, c in enumerate(cov_strs) if c.lower() in p]

        m1, m2 = np.loadtxt(p, unpack=True, usecols=(0,1))
        col = m1 - m2

        # axes limits for oldest population
        faintlim = 40
        xlim = [0.05, 1.1]
        ylim = [20.5, 17]
        idx = 2
        if '7' in p:
            # axes limits for youngest population
            faintlim = 16
            ylim = [faintlim, 10]
            xlim = [-0.3, 2]
            idx = 0
        if '8' in p:
            # axes limits for mid-age population
            faintlim = 18.0
            ylim = [faintlim, 14.25]
            xlim = [-0.1, 1.3]
            idx = 1

        # cut unresolved stars
        inds, = np.nonzero((m2 < 40) & (col < 40))
        # summary plot
        ax.plot(col[inds], m2[inds], 'o', ms=2, mec='none',
                color=clp[icov], zorder=100-icov,
                rasterized=True)
        inds, = np.nonzero((m2 < ylim[0]) & (col < xlim[1]) &
                           (m2 > ylim[1]) & (col > xlim[0]))
        # single burst figure
        ax_joints[idx].plot(col[inds], m2[inds], 'o', ms=2, mec='none',
                      color=clp[icov], zorder=100-icov,
                      rasterized=True)
        # 50 bins or use seaborn's nice binning
        cb = min(_freedman_diaconis_bins(col[inds]), 50)
        mb = min(_freedman_diaconis_bins(m2[inds]), 50)

        # histograms
        ax_marg_xs[idx].hist(col[inds], cb, orientation='vertical',
                       histtype='step', color=clp[icov], lw=1.4)
        ax_marg_ys[idx].hist(m2[inds], mb, orientation='horizontal',
                       histtype='step', color=clp[icov], lw=1.4)
        ax_joints[idx].set_ylim(ylim)
        ax_joints[idx].set_xlim(xlim)

    # main axes limits
    ax.set_xlim(-0.5,2)
    ax.set_ylim(22,10)
    ax.set_xlabel('$F555W-F814W$')
    ax.set_ylabel('$F814W$')

    # tweak plots
    for i in range(len(fs)):
        ax_marg_xs[i].set_yscale('log')
        ax_marg_ys[i].set_xscale('log')
        # fake out legend
        labels = [r'$\Lambda_c=%.1f$' % float(c.replace('OV', ''))
                for c in cov_strs]
        for j in range(len(labels)):
            ax_joints[i].plot(100, 100, 'o', color=clp[j], label=labels[j])
            if i == 0:
                ax.plot(100, 100, 'o', color=clp[j], label=labels[j])
        ax_joints[i].legend(loc='best')
        ax.legend(loc='best')

        ax_joints[i].set_xlabel(r'$F555W-F814W$')
        ax_joints[i].set_ylabel(r'$F814W$')
        fs[i].tight_layout()
        fs[i].subplots_adjust(hspace=space, wspace=space)
        fs[i].savefig('fake_cmds{0:d}{1:s}'.format(i+7, FIGEXT))
        print('wrote fake_cmds{0:d}{1:s}'.format(i+7, FIGEXT))

    f.savefig('fake_cmds{0:s}'.format(FIGEXT))
    print('wrote fake_cmds{0:s}'.format(FIGEXT))
    return


def cov_complifetimes(track_summary, hb=False, both=False):
    """Plot a comparison of the H or He lifetimes vs Mass to OV=0.50"""
    def setup_covlifetimes(data, z=0.008, hb=False, agescale=1e6,
                           intp_masses=None):
        """
        parse track_summary into useful dictionary
        interpolate for smoother plotting
        """
        intp_dict = {}
        if intp_masses is None:
            intp_masses = np.arange(1, 6, 0.02)
        tau = 'tau_H'
        for ov in np.array([0.3, 0.4, 0.5, 0.6]):
            df = data[(data.Z == z) & (data.HB == 0)]
            if hb:
                tau = 'tau_He'
                dfhb = data[(data.Z == z) & (data.HB == 1)]
                dfms = df[df.M > dfhb.M.max()]
                df = dfhb.append(dfms)
            iov = [i for i, l in enumerate(df['fname'])
                   if l.startswith('OV{:.1f}'.format(ov))]
            x = df['M'].iloc[iov]
            y = df[tau].iloc[iov] / agescale  # Units!!

            isort = np.argsort(x)
            fx = interp1d(x.iloc[isort], y.iloc[isort],
                          bounds_error=False)
            intp_dict[ov] = fx(intp_masses)
        return intp_dict

    tau = 'tau_H'
    if hb:
        tau = 'tau_He'
    taustr = r'\{}_{{{}}}'.format(*tau.split('_'))

    data = pd.read_table(track_summary, delim_whitespace=True)

    intp_masses = np.arange(1, 6, 0.02)
    uzs = np.unique(data['Z'])
    uovs = np.array([0.3, 0.4, 0.5, 0.6])

    xlim = (1.15, 5.5)
    linesyles = ['-', '--']
    fig, ax = plt.subplots(figsize=(8, 6))
    clp = ['darkred', 'orange', 'darkgreen', 'navy', 'purple']
    for k, z in enumerate([uzs[0], uzs[-1]]):
        intp_dict = setup_covlifetimes(data, z=z, hb=hb,
                                       intp_masses=intp_masses)
        for i, j in enumerate(uovs):
            x_ = intp_masses
            y_ = intp_dict[j] - intp_dict[0.5]
            if k == 0:
                ax.plot(x_, y_, linesyles[k], color=clp[i],
                        label=r'$\Lambda_c={}$'.format(j))
            else:
                ax.plot(x_, y_, linesyles[k], color=clp[i])

        ax.set_xlabel(r'$\rm{Mass}\ (\rm{M}_\odot)$')
        yfmt = r'$\Delta {}\ (\Lambda_c - \Lambda_{{c=0.50}})\ (\rm{{Myr}})$'
        ylab = yfmt.format(taustr)
        ax.set_ylabel(ylab)
    xticks = ax.xaxis.get_majorticklocs()
    age = np.log10(intp_dict[0.5] * 1e6)
    imass = [np.argmin(np.abs(x - intp_masses)) for x in xticks]
    plt.tick_params(right='off')
    ax.plot(-1, -1, '-', color='k', label='$Z=0.0005$')
    ax.plot(-1, -1, '--', color='k', label='$Z=0.01$')

    ax.set_xlim(xlim)
    if both:
        plt.legend(loc=0)
    ax1 = ax.twiny()
    ax1.plot(age, age, visible=False)
    ax1.set_xlim(age[imass[0]], age[imass[-1]])
    ax1.set_xlabel(r'${:s}\ (\log {{\rm yr}})$'.format(taustr))
    fig.subplots_adjust(left=0.15, top=0.85)
    plt.savefig('COV_{}'.format(tau.split('_')[1]) + FIGEXT)
    if both:
        hb2 = not hb
        ax2 = cov_complifetimes(track_summary, hb=hb2, both=False)
        ax = [ax, ax2]
    return ax


def plot_compare_tracks(Z=0.006, cmd=False):
    """
    Plot tracks of different overshooting values.
    If cmd, HB will be smoothed.
    """
    plt.rcParams['lines.linewidth'] = 1.4
    w = 21
    p = 1
    if not cmd:
        zoomb_kw = {'ylim': [1.75, 2.],
                    'xlim': [3.706, 3.685]}
        zoomt_kw = {'ylim': [1.5, 2.1],
                    'xlim': [3.725, 3.685]}
        xlim = [4.04, 3.66]
        ylim = [0.8, 2.5]
        ylabel = r'$\log L\ \rm{(L_\odot)}$'
        xlabel = r'$\log T_{\rm{eff}}\ \rm{(K)}$'
        xs = [3.9, 4.]
        ys = [1.27, 1.75]
        reversey = False
        ext = 'dat'
    else:
        zoomb_kw = {'ylim': [0.8, 1.15],
                    'xlim': [1.4, 1.54]}
        zoomt_kw = {'ylim': [0.5, 1.5],
                    'xlim': [1.28, 1.5]}
        ylim = [2.85, -1.5]
        xlim = [-0.15, 1.65]

        ylabel = r'$\rm{F475W}$'
        xlabel = r'$\rm{F475W-F814W}$'
        xs = [0.3, 0.1]
        ys = [1.75, 0.3]
        reversey = True
        ext = 'acs_wfc'

    cov_strs = ['OV0.3', 'OV0.4', 'OV0.5', 'OV0.6']
    masses = [1.5, 2.]
    cols = ['darkred', 'orange', 'darkgreen', 'navy', 'purple']
    tracks = []
    for cov in cov_strs:
        for mass in masses:
            track_dir, = get_dirs(TRACKS_LOC, criteria='%s_Z%g_' % (cov, Z))
            track_name = get_files(track_dir, '*M{:.2f}*{:s}'.format(mass, ext))
            track = Track(track_name[0])
            if len(track_name) > 1:
                # slap the PMS and HB tracks together...
                # this won't add MSTO age to HB, only for HRD plotting.
                df1 = pd.DataFrame(track.data)
                track.data = df1.append(pd.DataFrame(Track(track_name[1]).data),
                                        ignore_index=True)
            tracks.append(track)

    fig, (axm, axhbt, axhbb) = setup_zoomgrid()

    for k, t in enumerate(tracks[::-1]):
        if not cmd:
            x = t.data['logT']
            y = t.data['logL']
        else:
            x = t.data['F475W'] - t.data['F814W']
            y = t.data['F475W']

        cov_str = t.base.split('OV')[1].split('_')[0]
        label = (r'$\Lambda_c=%s$' % cov_str).replace('OV', '')
        icol, = [i for i, c in enumerate(cov_strs) if cov_str in c]

        if t.mass == masses[0]:
            # plot hb on lower inset
            ax = axhbb
            # add label once (first mass, not hb)
            if t.hb:
                axm.plot(x[200:], y[200:], color=cols[icol])
            else:
                axm.plot(x[200:], y[200:], label=label, color=cols[icol])
        else:
            axm.plot(x[200:], y[200:], color=cols[icol])
            # plot hb on upper inset
            ax = axhbt

        if t.hb:
            if cmd:
                ax.plot(savgol_filter(x, w, p), savgol_filter(y, w, p),
                        color=cols[icol])
            else:
                ax.plot(x, y, color=cols[icol])
        else:
            if cmd:
                ax.plot(savgol_filter(x[1130:], w, p),
                        savgol_filter(y[1130:], w, p),
                        color=cols[icol])
            else:
                ax.plot(x[1130:], y[1130:], color=cols[icol])

    for ax, m in zip([axhbb, axhbt], masses):
        ax.text(0.9, 0.01, '${}M_\odot$'.format(m), transform=ax.transAxes,
                fontsize=20, ha='right')

    for x, y, m in zip(xs, ys, masses):
        axm.text(x, y, '${}M_\odot$'.format(m), fontsize=20, ha='center')

    adjust_zoomgrid(axm, axhbt, axhbb, zoom1_kw=zoomt_kw,
                    zoom2_kw=zoomb_kw, reversey=reversey)

    [ax.locator_params('x', nbins=3) for ax in [axhbt, axhbb]]
    axm.set_xlabel(xlabel, fontsize=20)
    axm.set_ylabel(ylabel, fontsize=20)

    axm.set_xlim(xlim)
    axm.set_ylim(ylim)

    axm.tick_params(labelsize=16)
    axm.legend(loc=0, frameon=False, fontsize=16)

    axhbt.set_title('$Z={}$'.format(Z), fontsize=20)

    outfile = 'COV_HRD'
    if cmd:
        outfile = 'COV_CMD'
    plt.savefig(outfile + FIGEXT)
    print('wrote {0:s}'.format(outfile + FIGEXT))
    return
