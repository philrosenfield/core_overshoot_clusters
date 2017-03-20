"""Make PDF plots"""
import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from .pdf_plots import cluster_result_plots
from .config import PDF_LOC


def load_files(mock=False):
    """load filenames"""
    if mock:
        filenames = [os.path.join(PDF_LOC, 'mock_full.csv')]
    else:
        fns = ['HODGE2_full.csv',
               'NGC1718_full.csv',
               'NGC2203_full.csv',
               'NGC2213_full.csv',
               'NGC1644_full.csv',
               'NGC1795_full.csv']
        filenames = [os.path.join(PDF_LOC, f) for f in fns]

    for f in filenames:
        assert os.path.isfile(f), '{} not found'.format(f)
    return filenames


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="pdf plots for a calcsfh run")

    parser.add_argument('-t', '--test', action='store_true',
                        help='ssp test')

    parser.add_argument('--ov5', action='store_true',
                        help='marginalize over ov=0.5')

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    filenames = load_files(mock=args.test)
    cluster_result_plots(filenames, mock=args.test, ovis5=args.ov5)

if __name__ == "__main__":
    sys.exit(main())
