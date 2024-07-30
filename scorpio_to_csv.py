#! /usr/bin/env python3

import numpy as np
from astropy.io import fits
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib import pyplot as plt


def coords_from_header_values(crpix, crval, cdelt, naxis):
    """
    Calculate coordinates from header values.

    Parameters
    ----------
    crpix : float
        Reference pixel value.
    crval : float
        Reference value.
    cdelt : float
        Scale factor.
    naxis : int
        Number of pixels.

    Returns
    -------
    numpy.ndarray
        Array of calculated coordinates.

    """
    pix = np.arange(naxis)
    coords = cdelt * (pix - crpix) + crval
    return coords


def bin_coords(bins: fits.HDUList, coords: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    bins : fits.HDUList
        The list of bins.

    coords : numpy.ndarray
        An array of coordinates.

    Returns
    -------
    numpy.ndarray
        An array of coordinates with empty bins removed.
    """
    # list of bin numbers
    bins_ = set(bins[0].data)
    bins_.remove(-1)
    bins_ = np.array(list(bins_))
    bins_.sort()
    # plt.plot(bins[0].data)
    # plt.show()
    # print(bins[0].data)
    # mean coordinate of pixels in every bin
    coords_ = np.array([np.mean(coords[bins[0].data == b]) for b in bins_])
    coords = coords_[bins_ != -1]
    return coords


def bin_spec(bins, specc):
    bins_ = set(bins[0].data)
    bins_.remove(-1)
    bins_ = np.array(list(bins_))
    bins_.sort()
    # plt.figure()
    # plt.imshow(spec)
    # plt.show()
    spec = specc[:,180:]
    spec[spec < 0] = 0
    tflux = [spec[bins[0].data == b] for b in bins_]
    tflux = np.array([np.mean(x) for x in tflux])
    # plt.plot(tflux)
    # plt.show()
    tflux_err = np.sqrt(tflux) * 0
    return tflux, tflux_err


def get_radec(pos, hdr):
    slit_center = SkyCoord(hdr['RA'], hdr['DEC'], unit=(u.hourangle, u.deg))
    slit_frame = slit_center.skyoffset_frame(rotation=hdr['POSANG'] * u.deg)

    rel_lat = pos * u.arcsec
    rel_lon = 0 * pos * u.arcsec
    slit_points = SkyCoord(rel_lon, rel_lat, frame=slit_frame)
    slit = slit_points.transform_to('icrs')
    slit_ra = slit.ra.to_string(unit=u.hourangle, sep=':')
    slit_dec = slit.dec.to_string(unit=u.deg, sep=':')
    return slit_ra, slit_dec


def read_scorpio(data: fits.HDUList, bins: fits.HDUList,
                 spec: fits.ImageHDU, cut: list = None) -> pd.DataFrame:
    """
    Parameters
    ----------
    data : fits.HDUList
        The FITS file object containing the data to be read.

    bins : fits.HDUList
        The FITS file object containing the bin information.

    spec_hdr : fits.Header
        The FITS header object containing the spectral header information.

    Returns
    -------
    result_pd : pd.DataFrame
        The resulting pandas DataFrame containing the extracted data.

    """
    spec_hdr = spec.header
    pos = coords_from_header_values(spec_hdr['CRPIX2'], spec_hdr['CRVAL2'],
                                    spec_hdr['CDELT2'], spec_hdr['NAXIS2'])
    # coordiantes of the centrum of every bin
    coords = bin_coords(bins, pos)
    tot_flux, tot_flux_err = bin_spec(bins, spec.data)
    sortmask = np.argsort(coords)
    result_pd = pd.DataFrame()
    result_pd['position'] = coords[sortmask]
    result_pd['velocity'] = data[0].data[0, 0][sortmask]
    result_pd['velocity_err'] = data[0].data[1, 0][sortmask]
    result_pd['sigma_v'] = data[0].data[0, 1][sortmask]
    result_pd['sigma_v_err'] = data[0].data[1, 1][sortmask]
    result_pd['flux'] = data[0].data[0, 2][sortmask]
    result_pd['flux_err'] = data[0].data[1, 2][sortmask]
    result_pd['total_flux'] = tot_flux[sortmask]
    # plt.plot(tot_flux[sortmask])
    # plt.show()
    result_pd['total_flux_err'] = tot_flux_err[sortmask]
    result_pd['RA'], result_pd['DEC'] = get_radec(coords[sortmask], spec_hdr)

    return result_pd


def main(args=None):
    """
    Parse command-line arguments, start the program and save results.

    Parameters
    ----------
    args : list
        List of command-line arguments. Defaults to None.

    Returns
    -------
    int
        The exit code of the program.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('data',
                        help='''fits file with measured velocity, dispersion
                        and flux''')
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-b', '--bins',
                                help='fits file with binning map',
                                required=True)
    required_named.add_argument('-s', '--spectrum',
                                help='fits file with spectrum',
                                required=True)
    parser.add_argument('-c', '--cut', nargs=4,
                        help='''xmin, xmax, ymin, ymaxl - int coordinates of
                        pixels, None to skip some of the values''')
    parser.add_argument('-o', '--output', help='''full name of result file
                        (default is the name of data with the extention changed
                        to .csv)''')
    pargs = parser.parse_args(args[1:])

    cut = None
    if pargs.cut:
        cut = []
        for i in pargs.cut:
            try:
                cut.append(int(i))
            except ValueError:
                cut.append(None)

    data = fits.open(pargs.data)
    bins = fits.open(pargs.bins)
    spec_hdr = fits.open(pargs.spectrum)[0]

    if pargs.output:
        outname = pargs.output
    else:
        outname = ".".join(pargs.data.split(".")[:-1]) + ".csv"

    res = read_scorpio(data, bins, spec_hdr, cut)
    print('Writing result to ', outname)
    res.to_csv(outname)

    return 0


if __name__ == '__main__':
    import sys
    import argparse
    sys.exit(main(sys.argv))
