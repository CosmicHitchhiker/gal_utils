#! /usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
import pandas as pd
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('csvname',
                        help="""csv file with measured parameters""")
    parser.add_argument('-i', '--fitsname',
                        help="""fits image with correct wcs""")
    # parser.add_argument('-dx', type=float, default=0)
    # parser.add_argument('-dy', type=float, default=0)

    pargs = parser.parse_args(args[1:])
    csvname = pargs.csvname

    data = pd.read_csv(csvname)

    image = fits.open(pargs.fitsname)[0]
    wcs = WCS(image.header)
    slit = SkyCoord(data['RA'], data['DEC'], frame='icrs', unit=(u.hourangle, u.degree))
    slit_pix = wcs.world_to_pixel(slit)

    plt.imshow(image.data, origin='lower')
    plt.plot(slit_pix[0], slit_pix[1], 'ro')
    plt.show()

    return(0)


if __name__ == '__main__':
    import sys
    import argparse
    sys.exit(main(sys.argv))