"""Stats and visualization of calcsfh -ssp runs"""
from __future__ import absolute_import, print_function

import os

import numpy as np
import pandas as pd

__all__ = ['SSP']


def get_absprob(data):
    """absprob is the posterior since the fit parameter is -2 ln (posterior)"""
    data['absprob'] = np.exp(0.5 * (data['fit'].min() - data['fit']))
    return data


def quantiles(ux, prob, qs=[0.16, 0.84], res=200, maxp=False,
              ax=None, k=3):
    """
    Calculate quantiles, or lines at qs fraction of total area under the curve.

    Parameters
    ----------
    ux : 1D array
        vector assciated with prob
    prob : 1D array
        probability of ux
    qs : list or array
        quantiles
    res :
        number of resolution points for interpolation
    maxp : bool (False)
        append maximum posterior probability to returned quantile array
    ax : plt.Axes instance
        plot the interpolation
    Returns
    -------
    interpolated ux evaluated at qs
    """
    from scipy.interpolate import splprep, splev
    ((tckp, u), fp, ier, msg) = splprep([ux, prob], k=k, full_output=1)
    iux, iprob = splev(np.linspace(0, 1, res), tckp)

    fac = np.cumsum(iprob) / np.sum(iprob)
    ipts = [np.argmin(np.abs(fac - q)) for q in qs]
    g = iux[ipts]
    if maxp:
        g = np.append(g, iux[np.argmax(iprob)])
    if ax is not None:
        # useful for debugging or by-eye checking of interpolation
        ax.plot(iux, iprob, color='r')
    return g


def fitgauss1D(ux, prob):
    """Fit a 1D Gaussian to a marginalized probability
    Parameters

    xattr : str
        column name (and will be attribute name)
    ux :
        unique xattr values
    prob :
        marginalized probability at ux.

    Returns

    g : astropy.models.Gaussian1D object
    sets g as attribute 'xattr'g
    """
    assert ux is not None, \
        'need to supply values and probability to fitgauss1D'
    from astropy.modeling import models, fitting
    weights = np.ones(len(ux))
    fit_g = fitting.LevMarLSQFitter()
    g_init = models.Gaussian1D(amplitude=1.,
                               mean=np.mean(ux),
                               stddev=np.diff(ux)[0])
    noweight, = np.nonzero(prob == 2 * np.log(1e-323))
    weights[noweight] = 0.
    return fit_g(g_init, ux, prob, weights=weights)


def lnprob(prob):
    """
    ln-ify the probability and set a 0 tolerance
    """
    lenp = len(prob)
    prob[prob == 0] = 1e-323
    prob = 2 * np.log(prob)
    pmin = np.min(np.ravel(prob)[(np.isfinite(np.ravel(prob)))])
    # prob[np.isneginf(prob)] = pmin
    prob -= pmin
    assert len(prob) == lenp, 'lnprob has changed array shape of prob.'
    return prob


def marg(x, z, unx=None, log=True):
    """
    marginalize in 1d.
    Does not normalize probability.
    z should be ssp.absprob or some linear probability
    NOTE: this simple code only works on a grid with equally spaced data
    """
    ux = unx
    if ux is None:
        ux = np.unique(x)
    prob = np.zeros(len(ux))
    for i in range(len(ux)):
        iz, = np.nonzero(x == ux[i])
        prob[i] = np.sum(z.iloc[iz])
    if log:
        prob = lnprob(prob)
    return prob, ux


def marg2d(x, y, z, unx=None, uny=None, log=True):
    """
    marginalize in 2d.
    Does not normalize probability.
    z should be ssp.absprob or some linear probability
    NOTE: this simple code only works on a grid with equally spaced data
    """
    ux = unx
    if ux is None:
        ux = np.unique(x)
    uy = uny
    if uy is None:
        uy = np.unique(y)
    prob = np.zeros(shape=(len(ux), len(uy)))
    models = np.array([])
    for i in range(len(ux)):
        for j in range(len(uy)):
            iz, = np.nonzero((x == ux[i]) & (y == uy[j]))
            models = np.append(models, len(iz))
            if len(iz) > 0:
                prob[i, j] = np.sum(z.iloc[iz])
    unm = np.unique(models)
    if log:
        prob = lnprob(prob)
    return prob, ux, uy


def centered_meshgrid(x, y, unx=None, uny=None):
    """call meshgrid with bins shifted so x, y will be at bin center"""
    X, Y = np.meshgrid(center_grid(x, unx=unx),
                       center_grid(y, unx=uny),
                       indexing="ij")
    return X, Y


def center_grid(a, unx=None):
    """uniquify and shift a uniform array half a bin maintaining its size"""
    x = unx
    if x is None:
        x = np.unique(a)
    dx = np.diff(x)[0]
    x = np.append(x, x[-1] + dx)
    x -= dx / 2
    return x


class SSP(object):
    """
    Class for calcsfh -ssp outputs
    """

    def __init__(self, filename=None, data=None, filterby=None, gyr=False):
        """
        filenames are the calcsfh -ssp terminal or console output.
        They do not need to be stripped of their header or footer or
        be concatenated as is typical in MATCH useage.
        """
        self.gyr = gyr
        self.frompost = False
        if filename is not None:
            data = self.load_ssp(filename)
            if 'post' in filename:
                self.frompost = True

        if data is not None:
            if gyr:
                data['lage'] = (10 ** (np.array(data['lage'],
                                                dtype=float) - 9))

            if filterby is not None:
                # Perhaps this should split into a dict instead of culling...
                for key, val in filterby.items():
                    if len(np.nonzero(data[key] == val)[0]) == 0:
                        print(
                            'can not filter by {0:s}={1:g}: no matching values'
                            .format(key, val))
                        print('available values:', np.unique(data[key]))
                        import sys
                        sys.exit(1)
                    data = data.loc[data[key] == val].copy(deep=True)

            self.data = data

    def uniq_grid(self, skip_cols=None):
        """call unique_ on all columns."""
        skip_cols = skip_cols or ['None']
        cols = [c for c in self.data.columns
                if c not in skip_cols or 'prob' in c]
        [self.unique_(c) for c in cols]

    def _getmarginals(self, avoid_list=['fit']):
        """get the values to marginalize over that exist in the data"""
        # marg = np.array(['Av', 'IMF', 'dmod', 'lage', 'logZ', 'dav',
        # 'ov', 'bf'])
        marg_ = np.array([k for k in self.data.columns if k not in avoid_list])
        inds = [i for i, m in enumerate(marg_) if self._haskey(m)]
        return marg_[inds]

    def load_ssp(self, filename):
        """read table add file base, name to self"""
        self.base, self.name = os.path.split(filename)
        return pd.read_csv(filename)

    def _haskey(self, key):
        """test if the key requested is in self.data.columns"""
        ecode = True
        if key not in self.data.columns:
            ecode = False
        return ecode

    def fitgauss1D(self, xattr, ux, prob):
        """Fit a 1D Gaussian to a marginalized probability
        see fitgauss1D
        sets attribute 'xattr'g
        """
        g = fitgauss1D(ux, prob)
        self.__setattr__('{0:s}g'.format(xattr), g)
        return g

    def quantiles(self, xattr, ux, prob, qs=[0.16, 0.84], res=200, maxp=False,
                  ax=None, k=3):
        """Add quantiles, see quantiles"""
        g = quantiles(ux, prob, qs=qs, res=res, maxp=maxp, ax=ax,
                      k=k)
        self.__setattr__('{0:s}g'.format(xattr), g)
        return g

    def unique_(self, attr, uniq_attr='u{:s}'):
        """
        call np.unique on self.data[attr] if it has not already been called.

        Will store the unique array as an attribute passed with uniq_attr.

        vdict is also added to self. It is a dictionary of attr: bool
        where the bool is True if the attr has more than one value.
        """
        if not hasattr(self, 'vdict'):
            self.vdict = {}
        if self.frompost:
            self.vdict[attr] = True
            return self.data[attr]
        self.vdict[attr] = False
        uatr = uniq_attr.format(attr)
        if not hasattr(self, uatr):
            self.data[attr] = np.array(self.data[attr], dtype=float)
            uns, idx = np.unique(self.data[attr], return_counts=True)
            self.__setattr__(uatr, uns)
        u = self.__getattribute__(uatr)
        if len(u) > 1:
            self.vdict[attr] = True
        return u

    def marginalize(self, xattr, yattr=None, **kwargs):
        """
        Marginalize over one or two quanitities
        xattr, yattr : string, string
            data column to marginalize over

        Returns
        vals : array or list of arrays
            if yattr is passed, vals is output of cantered_meshgrid
            otherwise it's the unique values of data[xattr]
        prob : return from marg or marg2d

        NOTE:
        marg/marg2d only work for values calculated with an equal spaced grid.
        """
        assert self._haskey(xattr), '{} not found'.format(xattr)
        if not self._haskey('absprob'):
            self.data = get_absprob(self.data)

        z = self.data['absprob']
        x = self.data[xattr]
        ux = self.unique_(xattr)

        if yattr is not None:
            assert self._haskey(yattr), '{} not found'.format(yattr)
            y = self.data[yattr]
            uy = self.unique_(yattr)
            prob, ux, uy = marg2d(x, y, z, unx=ux, uny=uy, **kwargs)
            vals = centered_meshgrid(x, y, unx=ux, uny=uy)
        else:
            prob, ux = marg(x, z, unx=ux, **kwargs)
            vals = ux

        return vals, prob
