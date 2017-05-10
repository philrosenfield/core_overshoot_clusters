"""
Configuration realated parameters.

File locations, figure extensions, label strings, etc
"""
import os

FIGEXT = '.pdf'

base = os.path.split(os.path.split(__file__)[0])[0]

ASTECA_LOC = os.path.join(base, 'phot', 'asteca')
PHOT_LOC = os.path.join(base, 'phot', 'obs')
TRACKS_LOC = os.path.join(base, 'stev')
MOCK_LOC = os.path.join(base, 'mock')
PDF_LOC = os.path.join(base, 'pdfs')

for d in [ASTECA_LOC, PHOT_LOC, TRACKS_LOC, MOCK_LOC, PDF_LOC]:
    assert os.path.isdir(
        d), 'directory {} not found, check config.py'.format(d)


def key2label(string, gyr=False):
    """Latex labels for different strings."""
    def_fmt = r'$\rm{{{}}}$'
    convert = {'Av': r'$A_V$',
               'dmod': r'$\mu_0$',
               'lage': r'$\log\ \rm{Age\ (yr)}$',
               'logZ': r'$\log\ \rm{Z}$',
               'fit': r'$-2 \ln\ \rm{P}$',
               'ov': r'$\Lambda_c$'}

    if string not in convert.keys():
        convstr = def_fmt.format(string)
    else:
        convstr = convert[string]

    if gyr and 'age' in string:
        convstr = convstr.replace('yr', 'Gyr').replace(r'\log\ ', '')
    return convstr


def cmd_limits(targ):
    """
    CMD limits for plotting.

    zoom1_kw is the top right axis, HB
    zoom2_kw is the bottom right axis, MSTO
    xlim, ylim is the main axis.
    (ylim order doesn't matter, xlim order does)
    """
    kw = {'HODGE2': {'xlim': [0., 1.99],
                     'ylim': [26, 18.5],
                     'zoom1_kw': {'xlim': [1.29, 1.6],
                                  'ylim': [19.5, 20.67]},
                     'zoom2_kw': {'xlim': [0.3, 0.74],
                                  'ylim': [19.35, 21.1]}},
          'NGC1644': {'xlim': [0., 1.5],
                      'ylim': [26, 18],
                      'zoom1_kw': {'xlim': [0.8, 1.2],
                                   'ylim': [19.7, 18.7]},
                      'zoom2_kw': {'xlim': [0.15, 0.7],
                                   'ylim': [19.5, 21.5]}},
          'NGC1718': {'xlim': [0., 1.99],
                      'ylim': [26, 18.5],
                      'zoom1_kw': {'xlim': [1.5, 1.9],
                                   'ylim': [20.7, 19.7]},
                      'zoom2_kw': {'xlim': [0.6, 1.1],
                                   'ylim': [22.23, 20.73]}},
          'NGC2203': {'xlim': [0., 1.99],
                      'ylim': [26, 18.5],
                      'zoom1_kw': {'xlim': [1.35, 1.62],
                                   'ylim': [19.30, 20.30]},
                      'zoom2_kw': {'xlim': [0.37, 0.85],
                                   'ylim': [19.87, 21.50]}},
          'NGC2213': {'xlim': [0., 1.99],
                      'ylim': [26, 18.5],
                      'zoom1_kw': {'xlim': [1.30, 1.62],
                                   'ylim': [20.20, 19.00]},
                      'zoom2_kw': {'xlim': [0.40, 0.90],
                                   'ylim': [19.80, 21.50]}},
          'NGC1795': {'xlim': [0., 1.5],
                      'ylim': [26, 18],
                      'zoom1_kw': {'xlim': [0.9, 1.3],
                                   'ylim': [18.30, 20.0]},
                      'zoom2_kw': {'xlim': [0.25, 0.7],
                                   'ylim': [19.50, 21.50]}}}
    return kw[targ]
