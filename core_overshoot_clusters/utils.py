"""common utility functions"""
import os


def get_files(src, search_string):
    """
    Return a list files

    Parameters
    ----------
    src : str
        abs path of directory to search in
    search_string : str
        search criteria, as in *dat for ls *dat

    Returns
    -------
    files : list
        abs path of files found (or empty)
    """
    if not src.endswith('/'):
        src += '/'
    try:
        import glob
        files = glob.glob1(src, search_string)
    except IndexError:
        print('Cannot find {0:s} in {1:s}'.format(search_string, src))
        import sys
        sys.exit(2)
    files = [os.path.join(src, f)
             for f in files if os.path.isfile(os.path.join(src, f))]
    return files


def get_dirs(src, criteria=None):
    """
    Return a list of directories in src, optional simple cut by criteria

    Parameters
    ----------
    src : str
        abs path of directory to search in
    criteria : str
        simple if criteria in d to select within directories in src

    Returns
    -------
    dirs : list
        abs path of directories found (or empty)
    """
    dirs = [os.path.join(src, l)
            for l in os.listdir(src) if os.path.join(src, l)]
    if criteria is not None:
        dirs = [d for d in dirs if criteria in d]
    return dirs
