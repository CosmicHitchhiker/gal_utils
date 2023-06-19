#! /usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
import argparse
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from scipy import ndimage
from scipy import optimize
import astropy.units as u
from astropy.visualization import simple_norm
from astropy.coordinates import FK5
import scipy.signal as sig


def myinterp(x_new, x, f):
    x_sort = np.argsort(x)
    x_new_sort = np.argsort(x_new)
    x_new_unsort = np.argsort(x_new_sort)
    res = np.interp(x_new[x_new_sort], x[x_sort], f[x_sort])
    return(res[x_new_unsort])


def norm_vector(vec):
    return vec / np.linalg.norm(vec)


def correlation(vec1, vec2):
    return(np.arccos(np.sum(vec1 * vec2)))


def prepare_spec(spec):
    spec_filt = sig.medfilt(spec, 7)
    spec_new = spec - spec_filt.min()
    spec_new[spec_new < 0] = 0
    return spec_new


def Qfunc3(xys0, img, slit0, imgscale, wy=1.5, crpix2=256, spec_x=None,
           verbose=False):
    # Slit is vertical on image
    wslit = np.abs(0.5 * wy / imgscale)
    x0, y0, slitscale = xys0
    # slitscale *= np.sign(imgscale)
    # Move zero-point
    slit = prepare_spec(slit0.copy())
    slit = norm_vector(slit)
    if spec_x is None:
        spec_x = np.arange(len(slit))

    # Cut rectagular region with the width of slit and height of image
    img_x = np.arange(len(img[0])).astype(float)
    img_x_des = np.arange(x0 - wslit, x0 + wslit, 0.1)
    img_reg = np.array([np.interp(img_x_des, img_x, x) for x in img])
    # Sum flux in that region
    img_reg = np.sum(img_reg, axis=1)

    # Coordinates of every spectrum point in arcsec
    slit_x = (spec_x - crpix2) * slitscale
    # Coordinates of every point in cutted region of image in arcsec
    img_x = (np.arange(len(img_reg)) - y0) * imgscale

    img_reg = myinterp(slit_x, img_x, img_reg)
    img_reg = prepare_spec(img_reg)
    # print(img_reg)
    img_reg = norm_vector(img_reg)
    if verbose:
        plt.figure()
        plt.plot(slit_x / imgscale, slit, label='spec')
        plt.plot(slit_x / imgscale, img_reg, label='img')
        plt.legend()

        real_y = slit_x / imgscale + y0
        # len_x = np.abs(len(slit)*slitscale/imgscale).astype(int)
        # real_x = np.arange(len_x)*np.sign(slitscale/imgscale) + x0
        real_x = np.ones(len(slit)) * x0

        plt.figure()
        norm = simple_norm(img, 'linear', percent=98.0)
        plt.imshow(img, origin='lower', norm=norm)
        plt.plot(real_x, real_y)
        plt.plot(x0, y0, 'ro')
        plt.show()

    return(correlation(img_reg, slit))


def dxdy_setup(img, slit, imgscale, slitscale, xy_center, wy=1.5, crpix2=256,
               spec_x=None):
    dx = 0
    dy = 0

    need_to_change = 'Yes'
    while need_to_change != '':
        xys = [*xy_center, slitscale]
        xys[0] += dx
        xys[1] += dy
        Q = Qfunc3(xys, img, slit, imgscale, wy, crpix2, spec_x, verbose=True)
        print(Q)

        params = argparse.ArgumentParser(exit_on_error=False)
        params.add_argument('-dx', type=float, default=dx)
        params.add_argument('-dy', type=float, default=dy)
        params.add_argument('-i', type=float, default=imgscale)
        parags = params.parse_args('')
        print(parags)
        need_to_change = input("Change any parameters?(leave blank if No)")
        if need_to_change:
            parags = params.parse_args(need_to_change.split())
            dx = parags.dx
            dy = parags.dy
            imgscale = parags.i

    return dx, dy, imgscale


def parse_tds_slit(slitname):
    if slitname == '1asec':
        return 1.0
    elif slitname == '1.5asec':
        return 1.5
    elif slitname == '10asec':
        return 10
    raise ValueError('Unknown SLIT keyword')


def myrot(x, y, rot, cent=[0, 0]):
    drot = rot * np.pi / 180.
    xc = x - cent[0]
    yc = y - cent[1]
    xnew = xc * np.cos(drot) + yc * np.sin(drot) + cent[0]
    ynew = -xc * np.sin(drot) + yc * np.cos(drot) + cent[1]
    return [xnew, ynew]


def get_img_scale(slit_cent, wcs, angle, center):
    x = np.arange(-50, 50) + slit_cent[0]
    y = np.ones(100) * slit_cent[1]
    line = np.array(myrot(x, y, -angle, center))
    line = [wcs.pixel_to_world(i, j) for i, j in line.T]
    d_array = []
    for i, p in enumerate(line):
        if i != 0:
            s = line[i].separation(line[i - 1]) / u.deg
            s = s * 3600
            d_array.append(s)
    sep = round(np.mean(d_array), 5)
    return sep


def get_img_PA(wcs):
    '''Return angle between direction to the north pole and vertical vector on
    the center of image.
    '''
    nx, ny = wcs.array_shape
    pix1 = wcs.pixel_to_world(0.5 * nx, 0.5 * ny)
    pix2 = wcs.pixel_to_world(0.5 * nx, 0.6 * ny)
    pa = pix1.position_angle(pix2)
    return(pa.deg)


def get_init_slit_pos(hdr, rot=0):
    if 'EPOCH' in hdr:
        equinox = "J" + str(hdr['EPOCH'])
        spec_refcoord = SkyCoord(hdr['RA'], hdr['DEC'],
                                 unit=(u.hourangle, u.deg),
                                 frame=FK5(equinox=equinox))
    else:
        spec_refcoord = SkyCoord(hdr['RA'], hdr['DEC'],
                                 unit=(u.hourangle, u.deg))

    specscale = hdr['CDELT2']

    if 'CRPIX2' in hdr:
        crpix2 = hdr['CRPIX2']
    else:
        crpix2 = hdr['NAXIS2'] // 2

    if 'POSANG' in hdr:
        # TDS
        PA = hdr['POSANG']
    elif 'PARANGLE' in hdr and 'ROTANGLE' in hdr:
        # SCORPIO
        PA = hdr['PARANGLE'] - hdr['ROTANGLE'] + 132.5
    else:
        PA = 0
    PA += rot

    if specscale < 0:
        PA += 180
        specscale *= -1

    # if PA not in [0,360) (including negative PA)
    PA = PA - 360 * (PA // 360)

    if 'SLIT' in hdr:
        wy = parse_tds_slit(hdr['SLIT'])
    else:
        wy = 1.0

    return spec_refcoord, specscale, crpix2, PA, wy


def plot_img_slit(img, slit_cent, crpix2, spec_x, scalefact):
    plt.figure()
    norm = simple_norm(img, 'linear', percent=98.0)
    plt.imshow(img, origin='lower', norm=norm)
    plt.plot(slit_cent[0], slit_cent[1], 'ro', label='Slit refpoint')
    plt.legend()
    plt.show()


def find_slit_position(image, spec, speclim, rot):
    '''Find slit coordinates that fit image flux distribution the best.

    Parameters
    ----------
    image : ImageHDU or PrimaryHDU
            Image HDU that should contain correct WCS parameters in header.
            The photometry band should be close to the spectrum band.
    spec : ImageHDU or PrimaryHDU
            Spectrum HDU
    speclim : slice
            Slice with bounds of y-coordinates os spectrum. In pixels.

    Returns
    -------
    spec_header_new : fits header
            Fits header that contains correct coordinates of slit and
            can replace the original header of spectrum.
    '''
    # spec_refcoord - SkyCoord of reference pixel in spectrum
    # specscale - float (arcsec/pix), coordinate along slit
    # crpix2 - int or float, reference pixel along slit
    # PA - float (deg), positional angle of slit
    # wy - float (arcsec), width of slit
    spec_refcoord, specscale, crpix2, PA, wy = get_init_slit_pos(spec.header,
                                                                 rot)
    wcs = WCS(image.header)
    print(wcs)
    print('Reference coordinates before correction: ', spec_refcoord)
    xy_refslit = [int(t) for t in wcs.world_to_pixel(spec_refcoord)]

    imgPA = get_img_PA(wcs)
    print('Image PA ', imgPA)

    rotangle = PA - imgPA
    if np.abs(rotangle) < 1:
        rotangle = 0

    spec_data = spec.data[speclim]
    spec_x = np.arange(spec.header['NAXIS2'])[speclim]
    spec_data[spec_data < 0] = 0
    F_slit = np.sum(spec_data, axis=1)

    img = image.data
    # center of rotation
    center_image = [len(img[0]) / 2.0, len(img) / 2.0]
    # counterclockwise rotation
    xy_center = myrot(*xy_refslit, rotangle, center_image)
    img = ndimage.rotate(img, rotangle, reshape=False, mode='nearest')
    print("non-rotated center", xy_refslit)

    imgscale = get_img_scale(xy_center, wcs, rotangle, center_image)

    print('Spectrum CRPPIX2 ', crpix2)
    print('Spectrum scale ', specscale)
    print('Image scale ', imgscale)
    print('PA ', PA)
    print('Slit width ', wy)

    dx, dy, imgscale_n = dxdy_setup(img, F_slit, imgscale, specscale, xy_center,
                                    wy, crpix2, spec_x)

    if imgscale_n / imgscale < 0:
        PA += 180
    if PA > 360:
        PA -= 360

    imgscale = imgscale_n

    x = xy_center[0] + dx
    y = xy_center[1] + dy

    print('Looking for optimal parameters. Please, wait...')
    optargs = (img, F_slit, imgscale, wy, crpix2, spec_x)
    if specscale > 0:
        specbounds = (specscale * 0.98, specscale * 1.02)
    elif specscale < 0:
        specbounds = (specscale * 1.02, specscale * 0.98)
    bounds = [(None, None), (None, None), specbounds]
    xys0 = optimize.minimize(Qfunc3, [x, y, specscale], args=optargs,
                             bounds=bounds)

    print(xys0)
    print(Qfunc3(xys0.x, img, F_slit, imgscale, wy, crpix2, spec_x,
                 verbose=True))
    radec = myrot(*xys0.x[:2], -rotangle, center_image)
    radec = wcs.pixel_to_world(*radec)

    spec_header_new = spec.header.copy()
    spec_header_new['CDELT2'] = xys0.x[2]
    spec_header_new['RA'] = radec.ra.to_string(sep=':', unit=u.hourangle)
    spec_header_new['DEC'] = radec.dec.to_string(sep=':')
    spec_header_new['CRPIX2'] = crpix2
    if 'EPOCH' in spec_header_new:
        spec_header_new['EPOCH'] = 2000.0
    spec_header_new['POSANG'] = PA
    return(spec_header_new)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('image',
                        help='''fits file with object image and
                                valid WCS keys in header''')
    parser.add_argument('spectrum',
                        help='''fits file with object spectrum and
                                CDELT2 key in header''')
    parser.add_argument('-n', '--name', help='name for resulting file',
                        default=None)
    parser.add_argument('--ymin', default=None)
    parser.add_argument('--ymax', default=None)
    parser.add_argument('-r', '--rot', type=float,
                        help='angle to add to slit PA', default=0)
    pargs = parser.parse_args(args[1:])

    image = fits.open(pargs.image)[0]
    spectrum = fits.open(pargs.spectrum)[0]

    if pargs.ymin is not None:
        ymin = int(pargs.ymin)
    else:
        ymin = None
    if pargs.ymax is not None:
        ymax = int(pargs.ymax)
    else:
        ymax = None
    ylim = slice(ymin, ymax)

    hdr_new = find_slit_position(image, spectrum, ylim, pargs.rot)
    if pargs.name is None:
        name = (pargs.spectrum).split('.')[0] + '_corrected.fits'
    else:
        name = pargs.name

    res = fits.open(pargs.spectrum)
    res[0].header = hdr_new
    print('Saving result to ', name)
    res.writeto(name, overwrite=True)
    return(0)


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
