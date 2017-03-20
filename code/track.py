'''Class to read Padova tracks'''
import os
import numpy as np

# column names
logL = 'logL'
logT = 'logT'
age = 'age'
mass = 'mass'
logg = 'logg'
CO = 'CO'

class Track(object):
    '''Padova stellar track class.'''
    def __init__(self, filename):
        '''
        load the track
        adds path base, file name and hb (bool) to self.

        Parameters
        ----------
        filename : str
            the path to the PARSEC track file
        '''
        self.base, self.name = os.path.split(filename)

        self.hb = False
        if 'hb' in self.name.lower():
            self.hb = True

        self.load_track(filename)

    def load_track(self, filename):
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
        def mbol2logl(m):
            try:
                logl = (4.77 - float(m)) / 2.5
            except TypeError:
                logl = (4.77 - m) / 2.5
            return logl

        self.col_keys = [age, mass, logT, logL, logg, CO]
        with open(filename, 'r') as inp:
            header = inp.readline()
            col_keys = header.split()
            if len(col_keys) > len(self.col_keys):
                self.col_keys.extend(col_keys[7:])

        data = np.genfromtxt(filename, names=self.col_keys,
                             converters={3: lambda m: mbol2logl(m)})

        self.data = data.view(np.recarray)
        self.mass = self.data[mass][0]
        return data
