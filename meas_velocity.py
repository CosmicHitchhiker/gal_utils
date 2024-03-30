#! /usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from astropy.visualization import simple_norm
from astropy.io import fits
import pandas as pd
from tqdm import tqdm
import scipy.optimize as opt
from vorbin.voronoi_2d_binning import voronoi_2d_binning as vorbin
from astropy.coordinates import SkyCoord, SkyOffsetFrame
import astropy.units as u


def myimshow(img, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    norm = simple_norm(img, 'linear', percent=60.0)
    ax.imshow(img, origin='lower', norm=norm)
    
    
def sum_bins(reg, reg_err, bin_number, y_bar):
    bin_n = list(set(bin_number))
    bin_n.sort()
    binned_reg = np.array([np.sum(reg[bin_number == n], axis=0) for n in bin_n])
    binned_err = np.array([np.sqrt(np.sum(reg_err[bin_number == n] ** 2, axis=0))
                           for n in bin_n])
    sortmask = np.argsort(y_bar)
    binned_reg = binned_reg[sortmask]
    binned_err = binned_err[sortmask]
    return binned_reg, binned_err, sortmask


def mybin(reg: np.ndarray, reg_err: np.ndarray, regpos: np.ndarray, snr=3):
    """Voronoi-binning across y-axis

    Parameters
    ----------
    reg : 2D ndarray
        cut frame (rectangle)
    reg_err : 2D ndarray
        errors for reg
    regpos : 1D ndarray
        slit coordinate of each row in reg
    snr : float
        desired signal-to-noise ratio

    Returns
    -------
    binned_reg : 2D ndarray
        reg (input) binned across y-axis
    binned_err : 2D ndarray
        errors of binned_reg
    y_bar : 1D ndarray
        weighted center of each bin (position along slit)
    n_pixels : 1D ndarray
        number of pixels in each bin
    """
    y = regpos - regpos.min()
    x = np.ones(len(regpos))
    signal = np.sum(reg, axis=1)
    noise = np.sqrt(np.sum(reg_err ** 2, axis=1))

    # bin_number - numbers of bins assigned to each row
    # y_bar - coordinate of the bin signal-weighted center
    # sn - snr for each bin
    # n_pixels - number of pixels in each bin
    bin_number, _, _, _, y_bar, sn, n_pixels, _ = vorbin(x, y, signal, noise, snr)
    return bin_number, y_bar, n_pixels


def fits_to_data(names):
    """Open fits-files and put them in 'data' dict

    Parameters
    ----------
    names : list
        list of fits files to open

    Returns
    -------
    data : dict
        data['data'] - list of ndarrays with object
            frames (first HDU data)
        data['headers'] - list of headers of first HDUs
        data['errors'] - list of ndarrays with errors,
            if no 'error' HDU, then errors = sqrt(data)
        data['mask'] - list of masks showing good pixels

    """
    hduls = [fits.open(name) for name in names]
    frames = [x[0].data for x in hduls]
    headers = [x[0].header for x in hduls]

    data = {'data': frames, 'headers': headers}

    try:
        errors = [x['errors'].data for x in hduls]
        data['errors'] = errors
    except KeyError:
        data['errors'] = np.sqrt(data['data'])
        print('No errors found')

    try:
        mask = [x['mask'].data for x in hduls]
        data['mask'] = mask
    except KeyError:
        print('No mask found')

    return data


def gauss_cont(x, s, x0, amp, k, b):
    return amp * np.exp(-(x - x0) ** 2 / (2 * s ** 2)) + k * x + b

def true_gauss_cont(x, s, x0, amp, k, b):
    return amp / (s * np.sqrt(np.pi * 2)) * np.exp(-(x - x0) ** 2 / (2 * s ** 2)) + k * x + b


# noinspection PyTupleAssignmentBalance
def fit_gauss(data, x, err=None):
    """Fit signal with gaussian + linear continuum

    Parameters
    ----------
    data : 1D ndarray
        signal on given coordinates
    x : 1D ndarray
        coordinates (usually wavelengths)
    err : 1D ndarray
        errors of signal

    Returns
    -------
    gauss_parameters : 1D ndarray of size 5
        [sigma, x0, amplitude, k, b] (linear continuum is k*x + b)
        IMPORTANT: if failed to fit, all parameters are set to 0!
    gauss_errors : 1D ndarray of size 5
        errors for each value in gauss_parameters
        IMPORTANT: if failed to fit, all errors are set to 1000!

    """
    try:
        p0 = [3, x[np.argmax(data)], np.max(data), 0, np.median(data)]
        l_bound = [0, np.min(x), 0, -np.inf, -np.inf]
        h_bound = [np.max(x) - np.min(x), np.max(x), np.max(data) * 3,
                   np.inf, np.inf]
        popt, pcov = opt.curve_fit(true_gauss_cont, x, data, sigma=err,
                                   p0=p0, absolute_sigma=True,
                                   bounds=(l_bound, h_bound))
        return popt, np.sqrt(np.diag(pcov))
    except (RuntimeError, ValueError):
        return np.array([0] * 5), np.array([1000] * 5)


def wl_to_v(wl, ref=6562.81):
    c = 299792.458
    z = (wl - ref) / ref
    return c * z


def wlerr_to_verr(wl, ref=6562.81):
    c = 299792.458
    z = wl / ref
    return c * z


def coords_from_header_values(crpix, crval, cdelt, naxis):
    pix = np.arange(int(naxis)) + 1
    coords = float(cdelt) * (pix - int(crpix)) + float(crval)
    return coords


def get_radec(pos, hdr):
    slit_center = SkyCoord(hdr['RA'], hdr['DEC'], unit=(u.hourangle, u.deg))
    slit_PA = hdr['POSANG'] * u.deg
    # print(slit_center)
    # print(hdr['POSANG'])
    slit_frame = SkyOffsetFrame(origin=slit_center, rotation=slit_PA)
    slit_coords = SkyCoord(lon=np.zeros(len(pos))*u.arcsec, lat=pos*u.arcsec, frame=slit_frame)
    slit_refcoords = slit_coords.transform_to('icrs')

    # slit_ra = slit_center.ra + pos * u.arcsec * np.sin(hdr['POSANG'] * u.deg)
    # print(slit_ra)
    # slit_dec = slit_center.dec + pos * u.arcsec * np.cos(hdr['POSANG'] * u.deg)
    # slit = SkyCoord(slit_ra, slit_dec, frame='icrs')
    slit_ra = slit_refcoords.ra.to_string(unit=u.hourangle, sep=':')
    # print(slit_ra)
    slit_dec = slit_refcoords.dec.to_string(unit=u.deg, sep=':')
    return slit_ra, slit_dec


def meas_velocity(data, xlim, ylim, refwl=6562.81, binning=False):
    """

    Parameters
    ----------
    data
    xlim :
        x-borders of a rectangle region
    ylim :
        y-borders of a rectangle region
    refwl : float, default 6562.81 (Halpha)
        reference wavelenght of an emission line
    binning : bool, default False
        if True, data will be binned across y-axis
        in the given region using voronoi binning

    Returns
    -------

    """
    hdr = data['headers'][0]
    if 'CRDER1' in hdr:
        syserr = hdr['CRDER1']
    else:
        syserr = 0

    wl = coords_from_header_values(hdr['CRPIX1'], hdr['CRVAL1'],
                                   hdr['CDELT1'], hdr['NAXIS1'])
    pos = coords_from_header_values(hdr['CRPIX2'], hdr['CRVAL2'],
                                    hdr['CDELT2'], hdr['NAXIS2'])

    # Cut rectangular region
    reg = data['data'][0][ylim, xlim]
    reg_err = data['errors'][0][ylim, xlim]
    regwl = wl[xlim]
    regpos = pos[ylim]

    # full spectrum limited by y
    fullspec = data['data'][0][ylim]
    fullspec_err = data['errors'][0][ylim]

    # plot chosen region flux and snr
    fig, ax = plt.subplots(1, 2)
    myimshow(reg, ax[0])
    myimshow(reg / reg_err, ax[1])
    plt.show()

    if binning:
        # reg, reg_err, regpos, n_pixels = mybin(reg, reg_err, regpos, snr=binning)
        bin_number, y_bar, n_pixels = mybin(reg, reg_err, regpos, snr=binning)
        reg, reg_err, sortmask = sum_bins(reg, reg_err, bin_number, y_bar)
        fullspec, fullspec_err, sortmask = sum_bins(fullspec, fullspec_err, bin_number, y_bar)
        n_pixels = n_pixels[sortmask]
        regpos = y_bar[sortmask] + regpos.min()

        fig, ax = plt.subplots(1, 2)
        myimshow(reg, ax[0])
        myimshow(reg / reg_err, ax[1])
        plt.show()
        binned = True
    else:
        binned = False
        n_pixels = None

    vals = np.empty((0, 5))
    errs = np.empty((0, 5))

    for i, line in enumerate(tqdm(reg)):
        v, e = fit_gauss(line, regwl, err=reg_err[i])
        vals = np.vstack([vals, v])
        errs = np.append(errs, [e], axis=0)

    l_max = vals[:, 1]
    l_max_err = np.sqrt(errs[:, 1] ** 2 + syserr ** 2)
    mask = (l_max_err < 2)

    # flux in line
    I_max = vals[:, 2]
    I_max_err = np.sqrt(errs[:, 2] ** 2)

    # total flux
    F_total = np.sum(fullspec, axis=1)
    F_total_err = np.sqrt(np.sum(fullspec_err**2, axis=1))

    v = wl_to_v(l_max, refwl)
    v_err = wlerr_to_verr(l_max_err, refwl)

    plt.figure()
    plt.errorbar(regpos[mask], v[mask], yerr=v_err[mask], marker='.', ls='')
    plt.ylabel('$V_{los}, km/s$')
    plt.xlabel('position, arcsec')
    plt.title(hdr['OBJECT'])
    plt.show()

    result_pd = pd.DataFrame()

    # name of errors for <parameter> should be <parameter>_err
    result_pd['position'] = regpos[mask]
    result_pd['velocity'] = v[mask]
    result_pd['velocity_err'] = v_err[mask]
    result_pd['sigma_v'] = wlerr_to_verr(vals[:, 0], refwl)[mask]
    result_pd['sigma_v_err'] = wlerr_to_verr(errs[:, 0], refwl)[mask]
    result_pd['flux'] = I_max[mask]
    result_pd['flux_err'] = I_max_err[mask]
    result_pd['tflux'] = F_total[mask]
    result_pd['tflux_err'] = F_total_err[mask]
    if binned:
        result_pd['n_rows'] = n_pixels[mask]
    result_pd['RA'], result_pd['DEC'] = get_radec(regpos[mask], hdr)

    return result_pd


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('spectrum',
                        help='''fits file with object spectrum and
                                valid wavelength keys in header''')
    parser.add_argument('-x', '--xrange', nargs=2, type=int,
                        help='''x-borders of the region to use''')
    parser.add_argument('-y', '--yrange', nargs=2, type=int,
                        help='''y-borders of the region to use''')
    parser.add_argument('-o', '--output', help='full name of result file')
    parser.add_argument('-l', '--wavelength', type=float, default=6562.81,
                        help='reference wavelength')
    parser.add_argument('-b', '--binning', type=float, nargs='?', const=2,
                        default=0, help='''desired SNR; if not set, no binning will be performed''')
    parser.add_argument('-e', '--errors',
                        help='''fits file with errors (if not in spectrum)''')
    pargs = parser.parse_args(args[1:])

    specname = pargs.spectrum
    data = fits_to_data([specname])
    if pargs.yrange:
        y = slice(pargs.yrange[0], pargs.yrange[1])
    else:
        y = slice(None, None)

    if pargs.xrange:
        x = slice(pargs.xrange[0], pargs.xrange[1])
    else:
        x = slice(None, None)

    if pargs.errors:
        data['errors'] = [fits.getdata(pargs.errors)]

    if pargs.output:
        outname = pargs.output
    else:
        outname = ".".join(specname.split(".")[:-1]) + ".csv"

    res = meas_velocity(data, x, y, pargs.wavelength, pargs.binning)
    res.to_csv(outname)

    return 0


if __name__ == '__main__':
    import sys
    import argparse

    sys.exit(main(sys.argv))
