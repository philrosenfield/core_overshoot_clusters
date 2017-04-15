'''Class to read Padova tracks'''
import os
import numpy as np

# column names
logL = 'logL'
logT = 'logT'
age = 'age'
logg = 'logg'
CO = 'CO'

track_columns = {
    'parsec': [age, 'mass', logT, logL, logg, CO, 'mbol', 'ACS_WFC_F435W',
               'ACS_WFC_F475W', 'ACS_WFC_F550M', 'ACS_WFC_F555W',
               'ACS_WFC_F606W', 'ACS_WFC_F625W', 'ACS_WFC_F658N',
               'ACS_WFC_F660N', 'ACS_WFC_F775W', 'ACS_WFC_F814W',
               'ACS_WFC_F850LP', 'ACS_WFC_F892N'],

    'yy': ['model', 'shells', age, 'XCEN', 'YCEN', 'ZCEN', logL, 'logR',
           logg, logT, 'mcore', 'menvp', 'r_bcz', 'tau_c', 'T_bcz',
           'X_env', 'Z_env', 'logPc', 'logTc', 'logDc', 'beta', 'eta',
           'M_Hsh', 'DM_Hsh', 'M_He', 'M_Tmax', 'U_grav', 'I_TOT', 'I_env',
           'I_core', 'L_ppI', 'L_ppII', 'L_ppIII', 'L_CNO', 'L_3alpha', 'mass',
           'mbol', 'ACS_WFC_F435W', 'ACS_WFC_F475W', 'ACS_WFC_F550M',
           'ACS_WFC_F555W', 'ACS_WFC_F606W', 'ACS_WFC_F625W', 'ACS_WFC_F658N',
           'ACS_WFC_F660N', 'ACS_WFC_F775W', 'ACS_WFC_F814W', 'ACS_WFC_F850LP',
           'ACS_WFC_F892N'],

    'mist': [age, logT, logg, logL, 'Z_surf', 'ACS_WFC_F435W',
             'ACS_WFC_F475W', 'ACS_WFC_F502N', 'ACS_WFC_F550M',
             'ACS_WFC_F555W', 'ACS_WFC_F606W', 'ACS_WFC_F625W',
             'ACS_WFC_F658N', 'ACS_WFC_F660N', 'ACS_WFC_F775W',
             'ACS_WFC_F814W', 'ACS_WFC_F850LP', 'ACS_WFC_F892N', 'phase'],

    'vr': ['model', logL, logT, age, 'HCEN', 'DLdt', 'DTDt', 'mass',
           'mbol', 'ACS_WFC_F435W', 'ACS_WFC_F475W', 'ACS_WFC_F550M',
           'ACS_WFC_F555W', 'ACS_WFC_F606W', 'ACS_WFC_F625W', 'ACS_WFC_F658N',
           'ACS_WFC_F660N', 'ACS_WFC_F775W', 'ACS_WFC_F814W', 'ACS_WFC_F850LP',
           'ACS_WFC_F892N'],

    'dartmouth': [age, logT, logg,  logL, 'ACS_WFC_F435W', 'ACS_WFC_F475W',
                 'ACS_WFC_F555W', 'ACS_WFC_F606W', 'ACS_WFC_F625W',
                 'ACS_WFC_F775W', 'ACS_WFC_F814W', 'ACS_WFC_F850L'],
    }

class Track(object):
    '''Padova stellar track class.'''
    def __init__(self, filename, model='parsec'):
        '''
        load the track
        adds path base, file name and hb (bool) to self.

        Parameters
        ----------
        filename : str
            the path to the track file
        '''
        self.base, self.name = os.path.split(filename)

        self.hb = False
        if 'hb' in self.name.lower():
            self.hb = True

        self.load_track(filename, model=model)

    def load_track(self, filename, model='parsec'):
        '''
        load the PARSEC (interpolated for MATCH) tracks into a record array

        File contains Mbol, but it is converted to logL on read:
        logL = (4.77 - Mbol) / 2.5

        column names: logAge, mass, logT, logL, logg, CO
        CO will be 0 if there are no TPAGB tracks.


        Parameters
        ----------
        filename : str
            the path to the PARSEC track file

        Returns
        -------
        data : recarray
            file contents
        adds to self:
            data : recarry of the file contents
            mass : float
                initial mass of the track (from the first row of data)
        '''
        kw = {}
        if model == 'parsec':
            # Mbol to logL
            kw = {'converters': {3: lambda m: mbol2logl(m)}}

        data = np.genfromtxt(filename, names=track_columns[model], **kw)
        self.col_keys = data.dtype.names
        self.data = data.view(np.recarray)

        if model == 'parsec':
            self.mass = data['mass'][0]
        else:
            self.mass = float(os.path.split(filename)[1].split('M')[1].replace('.dat', ''))


def mbol2logl(m):
    try:
        logl = (4.77 - float(m)) / 2.5
    except TypeError:
        logl = (4.77 - m) / 2.5
    return logl
