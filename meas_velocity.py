#! /usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from astropy.visualization import simple_norm
from astropy.io import fits
import pandas as pd
from tqdm import tqdm
import scipy.optimize as opt
from vorbin.voronoi_2d_binning import voronoi_2d_binning as vorbin
from astropy.coordinates import SkyCoord
import astropy.units as u



def myimshow(img, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    norm = simple_norm(img, 'linear', percent=60.0)
    ax.imshow(img, origin='lower', norm=norm)


def mybin(reg, reg_err, regpos, snr=3):
    y = regpos - regpos.min()
    x = np.ones(len(regpos))
    S = np.sum(reg, axis=1)
    N = np.sqrt(np.sum(reg_err**2, axis=1))
    bin_number, _, _, _, y_bar, sn, nPixels, _ = vorbin(x, y, S, N, snr)
    bin_n = list(set(bin_number))
    bin_n.sort()
    binned_reg = np.array([np.sum(reg[bin_number == n], axis=0) for n in bin_n])
    binned_err = np.array([np.sqrt(np.sum(reg_err[bin_number == n]**2, axis=0))
                           for n in bin_n])
    sortmask = np.argsort(y_bar)
    binned_reg = binned_reg[sortmask]
    binned_err = binned_err[sortmask]
    nPixels = nPixels[sortmask]
    y_bar = y_bar[sortmask] + regpos.min()

    return binned_reg, binned_err, y_bar, nPixels


def fits_to_data(names):
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


def gauss_cont(x, s, x0, A, k, b):
    return A * np.exp(-(x - x0)**2 / (2 * s**2)) + k * x + b


def fit_gauss(data, x, err=None):
    try:
        p0 = [3, x[np.argmax(data)], np.max(data), 0, np.median(data)]
        l_bound = [0, np.min(x), 0, -np.inf, -np.inf]
        h_bound = [np.max(x) - np.min(x), np.max(x), np.max(data) * 3,
                   np.inf, np.inf]
        popt, pcov = opt.curve_fit(gauss_cont, x, data, sigma=err,
                                   p0=p0, absolute_sigma=True,
                                   bounds=(l_bound, h_bound))
        return(popt, np.sqrt(np.diag(pcov)))
    except (RuntimeError, ValueError):
        return(np.array([0] * 5), np.array([1000] * 5))


def wl_to_v(wl, ref=6562.81):
    c = 299792.458
    z = (wl - ref) / ref
    return(c * z)


def wlerr_to_verr(wl, ref=6562.81):
    c = 299792.458
    z = wl / ref
    return(c * z)


def coords_from_header_values(crpix, crval, cdelt, naxis):
    pix = np.arange(int(naxis)) + 1
    coords = float(cdelt) * (pix - int(crpix)) + float(crval)
    return coords


def get_radec(pos, hdr):
    slit_center = SkyCoord(hdr['RA'], hdr['DEC'], unit=(u.hourangle, u.deg))
    slit_ra = slit_center.ra + pos * u.arcsec * np.sin(hdr['POSANG'])
    slit_dec = slit_center.dec + pos * u.arcsec * np.cos(hdr['POSANG'])
    slit = SkyCoord(slit_ra, slit_dec, frame='icrs')
    slit_ra = slit.ra.to_string(unit=u.hourangle, sep=':')
    slit_dec = slit.dec.to_string(unit=u.deg, sep=':')
    return slit_ra, slit_dec


def meas_velocity(data, xlim, ylim, refwl=6562.81, binning=False):
    hdr = data['headers'][0]
    if 'CRDER1' in hdr:
        syserr = hdr['CRDER1']
    else:
        syserr = 0

    wl = coords_from_header_values(hdr['CRPIX1'], hdr['CRVAL1'],
                                   hdr['CDELT1'], hdr['NAXIS1'])
    pos = coords_from_header_values(hdr['CRPIX2'], hdr['CRVAL2'],
                                    hdr['CDELT2'], hdr['NAXIS2'])
    reg = data['data'][0][ylim, xlim]
    reg_err = data['errors'][0][ylim, xlim]
    regwl = wl[xlim]
    regpos = pos[ylim]

    fig, ax = plt.subplots(1, 2)
    myimshow(reg, ax[0])
    myimshow(reg / reg_err, ax[1])
    plt.show()

    if binning:
        reg, reg_err, regpos, nPixels = mybin(reg, reg_err, regpos, snr=binning)
        fig, ax = plt.subplots(1, 2)
        myimshow(reg, ax[0])
        myimshow(reg / reg_err, ax[1])
        plt.show()
        binned = True
    else:
        binned = False

    vals = np.empty((0, 5))
    errs = np.empty((0, 5))

    for i, line in enumerate(tqdm(reg)):
        v, e = fit_gauss(line, regwl, err=reg_err[i])
        vals = np.vstack([vals, v])
        errs = np.append(errs, [e], axis=0)

    l_max = vals[:, 1]
    l_max_err = np.sqrt(errs[:, 1]**2 + syserr**2)
    mask = (l_max_err < 2)

    I_max = vals[:, 2]
    I_max_err = np.sqrt(errs[:, 2]**2)

    v = wl_to_v(l_max, refwl)
    v_err = wlerr_to_verr(l_max_err, refwl)

    plt.figure()
    plt.errorbar(regpos[mask], v[mask], yerr=v_err[mask], marker='.', ls='')
    plt.ylabel('$V_{los}, km/s$')
    plt.xlabel('position, arcsec')
    plt.title(hdr['OBJECT'])
    plt.show()

    result_pd = pd.DataFrame()

    result_pd['position'] = regpos[mask]
    result_pd['velocity'] = v[mask]
    result_pd['v_err'] = v_err[mask]
    result_pd['sigma_v'] = wlerr_to_verr(vals[:, 0], refwl)[mask]
    result_pd['sigma_v_err'] = wlerr_to_verr(errs[:, 0], refwl)[mask]
    result_pd['flux'] = I_max[mask]
    result_pd['flux_err'] = I_max_err[mask]
    if binned:
        result_pd['n_rows'] = nPixels[mask]
    result_pd['RA'], result_pd['DEC'] = get_radec(regpos[mask], hdr)

    return result_pd


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('spectrum',
                        help='''fits file with object spectrum and
                                valid wavelength keys in header''')
    parser.add_argument('-x', '--xrange', nargs=2, type=int,
                        help='''region to measure''')
    parser.add_argument('-y', '--yrange', nargs=2, type=int,
                        help='''region to measure''')
    parser.add_argument('-o', '--output', help='full name of result file')
    parser.add_argument('-l', '--wavelength', type=float, default=6562.81,
                        help='reference wavelength')
    parser.add_argument('-b', '--binning', type=float, nargs='?', const=2,
                        default=0)
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
