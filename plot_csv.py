#! /usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib as mpl
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astropy.io import fits
from scipy import ndimage
from astropy.visualization import simple_norm
mpl.rcParams['text.usetex'] = True
# for \text command
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


def myrot(x, y, rot, cent=[0, 0], verbose=False):
    drot = rot * np.pi / 180.
    xc = x - cent[0]
    yc = y - cent[1]
    xnew = xc * np.cos(drot) + yc * np.sin(drot) + cent[0]
    ynew = -1 * xc * np.sin(drot) + yc * np.cos(drot) + cent[1]

    if verbose:
        print('rotangle in radians: ', drot)
        print(np.sin(drot))
        print(x, y)
        print(xc, yc)
        print(xnew, ynew)
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


def meas_slit_params(meascsv):
    '''
    Parameters
    ----------
    meascsv : pd.DataFrame
    '''
    # position raltive to reference pixel in arcsec
    pos = meascsv['position'].to_numpy()
    min_p, max_p = np.argmin(pos), np.argmax(pos)

    min_c = SkyCoord(*meascsv[['RA', 'DEC']].iloc[min_p],
                     unit=(u.hourangle, u.deg))
    max_c = SkyCoord(*meascsv[['RA', 'DEC']].iloc[max_p],
                     unit=(u.hourangle, u.deg))
    print(max_c.to_string('hmsdms'))

    PA = min_c.position_angle(max_c)

    max_pos = pos[max_p]
    zero = max_c.directional_offset_by(PA - 180 * u.deg, max_pos * u.arcsec)
    return PA.deg, zero


def onclick(event):
    for a in event.canvas.figure.axes:
        a.axvline(event.xdata, c='green')
    event.canvas.draw()


def plot_csv(csvname, error_lim, title, image=None, dx=0, dy=0):
    meascsv = pd.read_csv(csvname, index_col=0)
    mask = meascsv['v_err'] < error_lim

    fig, ax = plt.subplots(nrows=3, sharex=True, gridspec_kw={'hspace': 0})
    ax[0].set_title(title)
    ax[1].errorbar(meascsv['position'][mask], meascsv['velocity'][mask],
                   meascsv['v_err'][mask], marker='.', linestyle='')
    ax[1].set_ylabel(r'$V_{los}, km/s$', fontsize='x-large')
    ax[2].errorbar(meascsv['position'][mask], meascsv['flux'][mask],
                   meascsv['flux_err'][mask], marker='.', linestyle='')
    ax[2].set_ylabel(r'$I, counts$', fontsize='x-large')

    if image is not None:
        xlim = ax[0].get_xlim()

        PA, spec_center = meas_slit_params(meascsv)
        wcs = WCS(image.header)
        xy_cent = [int(t) for t in wcs.world_to_pixel(spec_center)]

        # imsc_sgn = np.sign(image.header['CD1_1'])
        imgPA = get_img_PA(wcs)
        print('Slit PA: ', PA)
        print('Image PA: ', imgPA)
        print('Spectrum reference point sky coordinates: ',
              spec_center.to_string('hmsdms'))
        print('Spectrum reference point image coordinates: ', xy_cent)
        # 90 to make image horizontal
        rotangle = PA - imgPA + 90
        print('rotangle: ', rotangle)

        img = image.data
        Ny, Nx = np.shape(img)
        center_image = [Nx / 2., Ny / 2.]
        xy_center = myrot(*xy_cent, rotangle, center_image, verbose=True)
        print(xy_center)
        img = ndimage.rotate(img, rotangle, reshape=False, mode='nearest')
        norm = simple_norm(img, 'linear', percent=98.0)
        imgscale = get_img_scale(xy_center, wcs, rotangle, center_image)

        # plt.figure()
        # plt.imshow(img, cmap='bone', origin='lower', norm=norm)
        # plt.plot(xy_center[0], xy_center[1], 'ro')
        # plt.show()

        print(imgscale)
        extent = [-(xy_center[0] * imgscale) + dx,
                  (Nx - xy_center[0]) * imgscale + dx,
                  -(xy_center[1] * imgscale) + dy,
                  (Ny - xy_center[1]) * imgscale + dy]

        ax[0].imshow(img, extent=extent, cmap='bone', origin='lower', norm=norm)
        # ax[0].plot(*xy_center, 'o')
        ax[0].set_xlim(xlim)
        ax[0].set_ylim(-15, 15)
        ax[0].axhline(-0.5, c='red')
        ax[0].axhline(0.5, c='red')
        ax[0].set_adjustable('datalim')
        fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()
    return 0


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('filename',
                        help="""csv file with mesured parameters""")
    parser.add_argument('-t', '--title', help="Graph title",
                        default='')
    parser.add_argument('-i', '--image', help='Image fits')
    parser.add_argument('-e', '--error', type=float, default=100.,
                        help='Limit on velocity error (default=100)')
    parser.add_argument('-dx', type=float, default=0)
    parser.add_argument('-dy', type=float, default=0)

    pargs = parser.parse_args(args[1:])
    csvname = pargs.filename
    dirname = csvname.split('/')[-2]
    title = pargs.title or dirname

    if pargs.image:
        image = fits.open(pargs.image)[0]
    else:
        image = None

    plot_csv(csvname, pargs.error, title, image, pargs.dx, pargs.dy)
    # ax[2].errorbar(meascsv['position'][mask], meascsv['sigma_v'][mask],
    #                meascsv['sigma_v_err'][mask], marker='.', linestyle='')
    # ax[2].set_ylabel(r'$\sigma V_{los}, km/s$', fontsize='x-large')
    # ax[2].set_xlabel('$position, "$', fontsize='x-large')
    return(0)


if __name__ == '__main__':
    import sys
    import argparse
    sys.exit(main(sys.argv))
