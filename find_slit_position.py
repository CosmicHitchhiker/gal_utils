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


def Qfunc3(xys0, img, slit0, imgscale, wy=1.5, crpix2=256, spec_x=None,
           verbose=False):
    # Slit is horizontal on image
    wy = np.abs(0.5 * wy / imgscale)
    x0, y0, slitscale = xys0
    # slitscale *= np.sign(imgscale)
    slit = norm_vector(slit0.copy())
    if spec_x is None:
        spec_x = np.arange(len(slit))

    img_y = np.arange(len(img)).astype(float)
    img_y_des = np.arange(y0 - wy, y0 + wy, 0.1)
    img_reg = np.array([np.interp(img_y_des, img_y, x) for x in img.T]).T

    # print('here')
    # print(img_reg[10])
    # plt.plot(img_reg[10])
    img_reg = np.sum(img_reg, axis=0)
    # plt.plot(img_reg)
    # plt.show()
    # print(img_reg)

    slit_x = (spec_x - crpix2) * slitscale
    img_x = (np.arange(len(img_reg)) - x0) * imgscale
    # print()
    # print(slit_x)
    # print(img_x)

    img_reg = myinterp(slit_x, img_x, img_reg)
    img_reg = img_reg - img_reg.min()
    # print(img_reg)
    img_reg = norm_vector(img_reg)
    if verbose:
        plt.figure()
        plt.plot(slit_x / slitscale, slit, label='spec')
        plt.plot(slit_x / slitscale, img_reg, label='img')
        plt.legend()

        realx = (spec_x - crpix2) * (slitscale / imgscale) + x0
        # len_x = np.abs(len(slit)*slitscale/imgscale).astype(int)
        # real_x = np.arange(len_x)*np.sign(slitscale/imgscale) + x0
        real_y = np.ones(len(slit)) * y0

        plt.figure()
        norm = simple_norm(img, 'linear', percent=98.0)
        plt.imshow(img, origin='lower', norm=norm)
        plt.plot(realx, real_y)
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
        parags = params.parse_args('')
        print(parags)
        need_to_change = input("Change any parameters?(leave blank if No)")
        if need_to_change:
            parags = params.parse_args(need_to_change.split())
            dx = parags.dx
            dy = parags.dy

    return dx, dy


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
    pix1 = wcs.pixel_to_world(0, 0)
    pix2 = wcs.pixel_to_world(0, 1)
    pa = pix1.position_angle(pix2)
    return(pa.deg)


def find_slit_position(image, spec, speclim):
    if 'EPOCH' in spec.header:
        equinox = "J" + str(spec.header['EPOCH'])
        spec_center = SkyCoord(spec.header['RA'], spec.header['DEC'],
                               unit=(u.hourangle, u.deg),
                               frame=FK5(equinox=equinox))
    else:
        spec_center = SkyCoord(spec.header['RA'], spec.header['DEC'],
                               unit=(u.hourangle, u.deg))
    wcs = WCS(image.header)
    print(wcs)
    print(spec_center)
    xy_cent = [int(t) for t in wcs.world_to_pixel(spec_center)]

    imsc_sgn = np.sign(image.header['CD1_1'] * 3600)
    print('Image scale ', imsc_sgn)

    imgPA = get_img_PA(wcs)
    print('Image PA ', imgPA)

    specscale = spec.header['CDELT2']

    if 'CRPIX2' in spec.header:
        crpix2 = spec.header['CRPIX2']
    else:
        crpix2 = spec.header['NAXIS2'] // 2

    if 'POSANG' in spec.header:
        # TDS
        PA = spec.header['POSANG']
    elif 'PARANGLE' in spec.header and 'ROTANGLE' in spec.header:
        # SCORPIO
        PA = spec.header['PARANGLE'] - spec.header['ROTANGLE'] + 132.5
    else:
        PA = 0

    if PA >= 360:
        PA -= 360

    rotangle = PA - imgPA - 90
    if np.abs(rotangle) < 1:
        rotangle = 0

    if 'SLIT' in spec.header:
        wy = parse_tds_slit(spec.header['SLIT'])
    else:
        wy = 1.0

    spec_data = spec.data[speclim]
    spec_x = np.arange(spec.header['NAXIS2'])[speclim]
    spec_data[spec_data < 0] = 0
    F_slit = np.sum(spec_data, axis=1)

    img = image.data
    center_image = [len(img[0]) / 2, len(img) / 2]
    xy_center = myrot(*xy_cent, rotangle, center_image)
    img = ndimage.rotate(img, rotangle, reshape=False, mode='nearest')
    print("non-rotated center", xy_cent)

    imgscale = get_img_scale(xy_center, wcs, rotangle, center_image) * imsc_sgn

    print('Spectrum CRPPIX2 ', crpix2)
    print('Spectrum scale ', specscale)
    print('Image scale ', imgscale)
    print('PA ', PA)
    print('Slit width ', wy)

    dx, dy = dxdy_setup(img, F_slit, imgscale, specscale, xy_center, wy, crpix2,
                        spec_x)

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

    hdr_new = find_slit_position(image, spectrum, ylim)
    if pargs.name is None:
        name = (pargs.spectrum).split('.')[0] + '_corrected.fits'
    else:
        name = pargs.name

    res = fits.open(pargs.spectrum)
    res[0].header = hdr_new
    res.writeto(name, overwrite=True)
    return(0)


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
