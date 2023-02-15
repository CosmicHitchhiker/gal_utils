#! /usr/bin/env python3

import numpy as np
from astropy.io import fits
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u


def coords_from_header_values(crpix, crval, cdelt, naxis):
    pix = np.arange(naxis) + 1
    coords = cdelt * (pix - crpix) + crval
    return coords


def bin_coords(bins, coords):
    bins_ = np.array(list(set(bins[0].data)))
    bins_.sort()
    coords_ = np.array([np.mean(coords[bins[0].data == b]) for b in bins_])
    coords = coords_[bins_ != -1]
    return coords


def get_radec(pos, hdr):
    slit_center = SkyCoord(hdr['RA'], hdr['DEC'], unit=(u.hourangle, u.deg))
    slit_frame = slit_center.skyoffset_frame(rotation=hdr['POSANG'] * u.deg)

    rel_lat = pos * u.arcsec
    rel_lon = 0 * pos * u.arcsec
    slit_points = SkyCoord(rel_lon, rel_lat, frame=slit_frame)
    # slit_ra = slit_center.ra + pos * u.arcsec * np.sin(hdr['POSANG'])
    # slit_dec = slit_center.dec + pos * u.arcsec * np.cos(hdr['POSANG'])
    # slit = SkyCoord(slit_ra, slit_dec, frame='icrs')
    slit = slit_points.transform_to('icrs')
    slit_ra = slit.ra.to_string(unit=u.hourangle, sep=':')
    slit_dec = slit.dec.to_string(unit=u.deg, sep=':')
    return slit_ra, slit_dec


def read_scorpio(data, bins, spec_hdr):
    pos = coords_from_header_values(spec_hdr['CRPIX2'], spec_hdr['CRVAL2'],
                                    spec_hdr['CDELT2'], spec_hdr['NAXIS2'])
    coords = bin_coords(bins, pos)
    sortmask = np.argsort(coords)
    result_pd = pd.DataFrame()
    result_pd['position'] = coords[sortmask]
    result_pd['velocity'] = data[0].data[0, 0][sortmask]
    result_pd['v_err'] = data[0].data[1, 0][sortmask]
    result_pd['sigma_v'] = data[0].data[0, 1][sortmask]
    result_pd['sigma_v_err'] = data[0].data[1, 1][sortmask]
    result_pd['flux'] = data[0].data[0, 2][sortmask]
    result_pd['flux_err'] = data[0].data[1, 2][sortmask]
    result_pd['RA'], result_pd['DEC'] = get_radec(coords[sortmask], spec_hdr)

    return result_pd


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('data',
                        help='''fits file with measured velocity, dispersion
                        and flux''')
    parser.add_argument('-b', '--bins', help='fits file with binning map')
    parser.add_argument('-s', '--spectrum',
                        help='fits file with spectrum (to get coordinates)')
    parser.add_argument('-o', '--output', help='full name of result file')
    pargs = parser.parse_args(args[1:])

    data = fits.open(pargs.data)

    if pargs.bins:
        bins = fits.open(pargs.bins)
    else:
        bins = np.arange(data.header[0]['NAXIS1'])

    if pargs.spectrum:
        spec_hdr = fits.getheader(pargs.spectrum)
    else:
        spec_hdr = None

    if pargs.output:
        outname = pargs.output
    else:
        outname = ".".join(pargs.data.split(".")[:-1]) + ".csv"

    res = read_scorpio(data, bins, spec_hdr)
    print('Writing result to ', outname)
    res.to_csv(outname)

    return 0


if __name__ == '__main__':
    import sys
    import argparse
    sys.exit(main(sys.argv))
