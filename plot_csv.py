#! /usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from astropy.io import fits
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' #for \text command


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('filename',
                        help="""csv file with mesured parameters""")
    parser.add_argument('-t', '--title', help="Graph title",
                        default='')

    pargs = parser.parse_args(args[1:])
    csvname = pargs.filename
    dirname = csvname.split('/')[-2]
    title = pargs.title or dirname

    meascsv = pd.read_csv(csvname, index_col=0)
    meascsv['position'] = meascsv['position'] / 10.
    mask = meascsv['v_err'] < 100

    fig, ax = plt.subplots(nrows=3, sharex=True, gridspec_kw={'hspace':0})
    ax[0].set_title(title)
    ax[0].errorbar(meascsv['position'][mask], meascsv['velocity'][mask],
                   meascsv['v_err'][mask], marker='.', linestyle='')
    ax[0].set_ylabel(r'$V_{los}, km/s$', fontsize='x-large')
    ax[1].errorbar(meascsv['position'][mask], meascsv['flux'][mask],
                   meascsv['flux_err'][mask], marker='.', linestyle='')
    ax[1].set_ylabel(r'$I, counts$', fontsize='x-large')
    ax[2].errorbar(meascsv['position'][mask], meascsv['sigma_v'][mask],
                   meascsv['sigma_v_err'][mask], marker='.', linestyle='')
    ax[2].set_ylabel(r'$\sigma V_{los}, km/s$', fontsize='x-large')
    ax[2].set_xlabel('$position, "$', fontsize='x-large')
    plt.show()

    return(0)


if __name__ == '__main__':
    import sys
    import argparse
    sys.exit(main(sys.argv))
