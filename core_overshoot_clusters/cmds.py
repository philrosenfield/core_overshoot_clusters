"""main function to make CMD related plots"""
from __future__ import absolute_import
import os
import sys

from .cmd_plots import cmd_plots
from .config import ASTECA_LOC, PHOT_LOC
from .utils import get_files


def main():
    """main caller for CMD plots"""
    membs = get_files(ASTECA_LOC, '*memb*')
    clusters = []
    for m in membs:
        name = os.path.splitext(os.path.split(m)[1])[0]
        prefix = '_'.join(name.split('_')[:2])
        cluster, = get_files(PHOT_LOC, '{}*fits'.format(prefix))
        clusters.append(cluster)
    assert len(clusters) == len(membs), \
        'Something went wrong looking for asteca and cluster files'

    return cmd_plots(clusters, membs)


if __name__ == "__main__":
    sys.exit(main())
