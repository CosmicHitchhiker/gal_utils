#! /usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
import argparse
from scipy.optimize import minimize


def fits_to_data(names):
    hduls = [fits.open(name) for name in names]
    frames = np.array([x[0].data for x in hduls])
    headers = [x[0].header for x in hduls]

    try:
        errors = np.array([x['errors'].data for x in hduls])
    except KeyError:
        print('No errors found')

    try:
        mask = np.array([x['mask'].data for x in hduls]).astype(bool)
    except KeyError:
        print('No mask found')

    data = {'data': frames, 'headers': headers,
            'errors': errors, 'mask': mask}
    return data

def my_average(array, weights=None, axis=None):
    if weights is None:
        weights = np.ones_like(array)
    data_copy = array.copy()
    weighted_data = np.sum(data_copy * weights, axis=axis)
    weights_sum = np.sum(weights, axis=axis)
    good_pixels = (weights_sum != 0)
    weighted_data[good_pixels] = weighted_data[good_pixels] \
                                 / weights_sum[good_pixels]
    data_copy = weighted_data
    return data_copy


def fits_from_data(data, summ=False):
    data_copy = data.copy()
    if summ:
        data_copy = sum_data(data_copy)

    if len(np.shape(data_copy['data'])) > 2:
        result = fitses_from_data(data_copy)
        return result

    hdr = prerpare_header(data_copy)

    result = [fits.PrimaryHDU(data_copy['data'], header=hdr)]
    if 'errors' in data_copy:
        result.append(fits.ImageHDU(data_copy['errors'], name='errors'))
    if 'mask' in data_copy:
        result.append(fits.ImageHDU(data_copy['mask'].astype(int),
                                    name='mask'))

    result = fits.HDUList(result)
    return result


def fitses_from_data(data):
    data_copy = data.copy()
    res = []
    for i in range(len(data['data'])):
        data_copy['data'] = data['data'][i]
        if 'errors' in data:
            data_copy['errors'] = data['errors'][i]
        if 'mask' in data:
            data_copy['mask'] = data['mask'][i]
        if 'headers' in data:
            data_copy['headers'] = [data['headers'][i]]
        res.append(fits_from_data(data_copy))
    return res


def sum_data(data):
    data_copy = data.copy()
    if 'mask' in data_copy:
        weights = (~data_copy['mask'].astype(bool)).astype(int)
        data_copy['mask'] = np.sum(data_copy['mask'], axis=0)
    else:
        weights = None

    # data_copy['data'] = np.average(data_copy['data'], weights=weights,
    #                                axis=0)
    data_copy['data'] = my_average(data_copy['data'], weights, axis=0)

    if 'errors' in data_copy:
        data_copy['errors'] = np.sqrt(my_average(data_copy['errors']**2,
                                                 weights=weights, axis=0))
    return data_copy


def prerpare_header(data):
    if 'headers' not in data:
        header = fits.header.Header()
    else:
        header = data['headers'][0]
    if 'keys' in data:
        header.update(data['keys'])
    return header


def norm_vector(vec):
    return vec / np.linalg.norm(vec)


def correlation(vec1, vec2):
    return(np.arccos(np.sum(vec1 * vec2)))


def Qfunc(dx, refspec, spector):
    ref = norm_vector(refspec.copy())
    ref_x = np.arange(len(ref))
    x = np.arange(len(spector)) + dx
    
    moved_spec = np.interp(ref_x, x, spector.copy())
    spec = norm_vector(moved_spec)
    return(correlation(ref, spec))


def moveflux(flux, dx):
    x = np.arange(len(flux))
    x_ref = x - dx
    res = np.interp(x_ref, x, flux)
    return res


def moveframe(flux, dx):
    x = np.arange(len(flux))
    x_ref = x - dx
    res = np.array([np.interp(x_ref, x, f) for f in flux.T]).T
    return res


def find_moves(data, cut=None, verbose=False):
    print(cut)
    data_copy = data['data'][:]
    if cut:
        data_copy = data_copy[:,cut]

    fluxes = np.sum(data_copy, axis=2)
    refspec = fluxes[0]

    dxs = [round(minimize(Qfunc, [0], (refspec, flux)).x[0], 3)
           for flux in fluxes]

    if verbose:
        res_fluxes = np.array([moveflux(f, x) for f, x in zip(fluxes, dxs)])
        plt.plot(res_fluxes.T)
        plt.show()

    return dxs


def correlate_spectra(data, cut=None):
    data_copy = data.copy()
    moves = find_moves(data_copy, cut, verbose=True)
    print(moves)
    data_copy['data'] = np.array([moveframe(f, x)
                                  for f, x in zip(data_copy['data'], moves)])
    if 'errors' in data_copy:
        data_copy['errors'] = np.array([moveframe(f, x)
                                  for f, x in zip(data_copy['errors'], moves)])
    if 'mask' in data_copy:
        data_copy['mask'] = np.array([moveframe(f, x)
                                  for f, x in zip(data_copy['mask'], moves)])
        for i, dx in enumerate(moves):
            if dx >= 0:
                data_copy['mask'][i][:int(dx)+1] = False
            else:
                data_copy['mask'][i][int(dx)+1:] = False

    plt.plot(np.sum(data_copy['data'], axis=2).T)
    plt.show()

    data_sum = sum_data(data_copy)
    result = fits_from_data(data_sum)
    return result


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+',
                        help="""fits files with frames to correlate, first file
                         is used as a reference frame""")
    parser.add_argument('-d', '--dir', help="directory with input files")
    parser.add_argument('-o', '--out', help='output file')
    parser.add_argument('-x', '--initial', nargs='+', 
                        help="initial guesses for shifts")
    parser.add_argument('-c', '--cut', nargs=2, type=int,
                        help="region to use")
    pargs = parser.parse_args(args[1:])

    filenames = pargs.filenames 
    if pargs.dir:
        filenames = [pargs.dir + f for f in filenames]

    if pargs.out:
        outname = pargs.out
    else:
        outname = ''.join(filenames[0].split('.')[:-1]) + '_summ.fits'

    if pargs.cut:
        cut = slice(pargs.cut[0], pargs.cut[1])
    else:
        cut = slice(None, None)

    data = fits_to_data(filenames)
    res = correlate_spectra(data, cut)

    res.writeto(outname, overwrite=True)
    return(0)


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))