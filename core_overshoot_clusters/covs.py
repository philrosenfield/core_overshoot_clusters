"""Make plots related to core overshooting."""
import argparse
import os
import sys

from .config import MOCK_LOC, TRACKS_LOC
from .cov_plots import (add_models, cov_complifetimes, fake_cmds,
                        plot_compare_tracks)
from .utils import get_files


def parse_args(argv=None):
    """Argument parsinge for core overshooting plots."""
    parser = argparse.ArgumentParser(
        description="plots related to core overshooting")

    parser.add_argument('-t', '--tracks', action='store_true',
                        help='tracks plot')

    parser.add_argument('-s', '--stev', action='store_true',
                        help='tracks plot with more stellar evolution codes')

    parser.add_argument('-l', '--lifetimes', action='store_true',
                        help='core burning lifetimes plot')

    parser.add_argument('-m', '--mock_cmds', action='store_true',
                        help='synthetic CMDs plot')

    return parser.parse_args(argv)


def main(argv=None):
    """Main caller for overshooting plots."""
    args = parse_args(argv)
    if args.tracks:
        # track files are searched for in plot_compare_tracks.
        [plot_compare_tracks(cmd=b) for b in [True, False]]
    if args.stev:
        [add_models(cmd=b) for b in [True, False]]
    if args.lifetimes:
        track_summary = os.path.join(TRACKS_LOC, 'track_summary.dat')
        cov_complifetimes(track_summary, both=True)
    if args.mock_cmds:
        # load mock data
        phots = get_files(MOCK_LOC, '*full')
        fake_cmds(phots)


if __name__ == "__main__":
    sys.exit(main())
