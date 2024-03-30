#! /usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib as mpl
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astropy.io import fits
import reproject
from copy import deepcopy
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
    # print(max_c.to_string('hmsdms'))

    PA = min_c.position_angle(max_c)
    PA = 154.6*u.deg
    # print('meas_slit_params; PA slit = ', PA.to(u.deg))

    max_pos = pos[max_p]
    zero = max_c.directional_offset_by(PA - 180 * u.deg, max_pos * u.arcsec)
    # print('Slit center: ', zero)
    return PA.to(u.deg).value, zero


def onclick(event):
    if event.key == 'shift':
        for a in event.canvas.figure.axes:
            a.axvline(event.xdata, c='green')
        event.canvas.draw()


def colnames(dframe):
    keys = dframe.keys().tolist()
    good_keys = [x for x in keys if x+'_err' in keys]
    return good_keys


def get_rot_matrix(angle):
    alph = np.radians(angle + 90)
    rot_matrix = np.array([[-np.cos(alph), np.sin(alph)],[np.sin(alph), np.cos(alph)]])
    return rot_matrix


def make_slit_wcs(slit_PA, slitpos, shape=None, center=None, cdelt=None):
    """Make wcs where x-axis corresponds to the slit PA.
    Refpix of the wcs is the slit center.

    Parameters
    ----------
    slit_PA
    slitpos
    shape
    center
    cdelt

    Returns
    -------

    """
    if cdelt is None:
        cdelt = (0.1*u.arcsec).to(u.deg).value
    if shape is None:
        shape = (1500, 1500)
    if center is None:
        center = [shape[0]/2.0, shape[1]/2.0]

    w = WCS(naxis=2)
    w.wcs.cdelt = [cdelt, cdelt]
    w.wcs.cunit = [u.deg, u.deg]
    w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
    w.wcs.pc = get_rot_matrix(slit_PA)
    w.wcs.crpix = center
    # print('Slit center: ', slitpos)
    w.wcs.crval = [slitpos.ra.to(u.deg).value, slitpos.dec.to(u.deg).value]
    # print('Slit center: ', w.wcs.crval)
    w.pixel_shape = shape
    w.wcs.latpole = 0.
    w.wcs.lonpole = 180.
    w.wcs.equinox = 2000.0

    w_header = w.to_header()
    w_header['NAXIS'] = 2
    w_header['NAXIS1'], w_header['NAXIS2'] = w.pixel_shape

    return w, w_header


def get_rotated_image(image, slit_coords):
    """Rotate image so that slit is horizontal

    Parameters
    ----------
    image : astropy.fits.ImageHDU
        image with correct wcs in the header
    slit_coords : pandas.DataFrame
        Data frame with columns 'RA', 'DEC', 'position'
        'position' - position of points relative to the
        slit center (arcsec)
        'RA' and 'DEC' - coordinates of points (hourangle, degrees)
    """
    slit_PA, slitpos = meas_slit_params(slit_coords)
    # print('Slit center: ', slitpos)
    w, w_header = make_slit_wcs(slit_PA, slitpos)
    # print(w)
    rot_image, _ = reproject.reproject_interp(image, w_header)
    # plt.subplot(projection=w)
    # plt.imshow(rot_image, origin='lower')
    # plt.show()
    # print(w)
    return rot_image, w



def plot_csv(data_val, data_err, position, title, image=None, coords=None, dx=0, dy=0):
    nrows = len(data_val.keys())
    if image is not None:
        is_image=1
    else:
        is_image=0


    fig, ax = plt.subplots(nrows=nrows+is_image, squeeze=False, sharex=True, gridspec_kw={'hspace': 0})
    ax[0][0].set_title(title)

    for i, key_name in enumerate(data_val.keys()):
        ax[i+is_image][0].errorbar(position, data_val[key_name],
                                data_err[key_name], marker='.', linestyle='')
        ax[i+is_image][0].set_ylabel(key_name, fontsize='x-large')

    if image is not None:
        # what borders do the plots already have
        xlim = ax[1][0].get_xlim()

        # rotate image so that x-axis is the slit
        rot_img, wcs = get_rotated_image(image, coords)
        rot_img = np.nan_to_num(rot_img, nan=np.median(rot_img))
        # print(wcs.wcs.cdelt)
        # print(wcs.wcs.cunit)

        imgscale = (wcs.wcs.cdelt[0] * u.Unit(wcs.wcs.cunit[0])).to(u.arcsec)
        imgscale = imgscale.value
        xy_center = wcs.wcs.crpix
        Nx, Ny = wcs.pixel_shape
        # print('DEBUG: ', imgscale, xy_center, Ny, Nx)


    #     xlim = ax[0].get_xlim()
    #
    #     PA, spec_center = meas_slit_params(meascsv)
    #     wcs = WCS(image.header)
    #     xy_cent = [int(t) for t in wcs.world_to_pixel(spec_center)]
    #
    #     # imsc_sgn = np.sign(image.header['CD1_1'])
    #     imgPA = get_img_PA(wcs)
    #     print('Slit PA: ', PA)
    #     print('Image PA: ', imgPA)
    #     print('Spectrum reference point sky coordinates: ',
    #           spec_center.to_string('hmsdms'))
    #     print('Spectrum reference point image coordinates: ', xy_cent)
    #     # 90 to make image horizontal
    #     rotangle = PA - imgPA + 90
    #     print('rotangle: ', rotangle)
    #
    #     img = image.data
    #     Ny, Nx = np.shape(img)
    #     center_image = [Nx / 2., Ny / 2.]
    #     xy_center = myrot(*xy_cent, rotangle, center_image, verbose=True)
    #     print(xy_center)
    #     img = ndimage.rotate(img, rotangle, reshape=False, mode='nearest')
    #     norm = simple_norm(rot_img, 'linear', percent=98.0)
    #     imgscale = get_img_scale(xy_center, wcs, rotangle, center_image)
    #
    #     print(imgscale)
        extent = [-(xy_center[0] * imgscale) + dx,
                  (Nx - xy_center[0]) * imgscale + dx,
                  -(xy_center[1] * imgscale) + dy,
                  (Ny - xy_center[1]) * imgscale + dy]
    #
        ax[0][0].imshow(rot_img, extent=extent, cmap='bone', origin='lower')
    #     # ax[0].plot(*xy_center, 'o')
        ax[0][0].set_xlim(xlim)
        ax[0][0].set_ylim(-15, 15)
        ax[0][0].axhline(-0.5, c='red')
        ax[0][0].axhline(0.5, c='red')
        ax[0][0].set_adjustable('datalim')
    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

    return 0


def get_limits_mask(data, limits_list):
    mask = pd.Series(True, index=data.index)

    key_list = limits_list[::2]
    val_list = limits_list[1::2]
    if len(key_list) != len(val_list):
        raise IndexError('CHECK --max ATTRIBUTE IN TERMINAL PROMT')

    for k, v in zip(key_list, val_list):
        if v[-1] == '%':
            k2 = k[:-4]
            rel_err_mask = (data[k]/data[k2] <= (float(v[:-1])/100.))
            mask = mask & rel_err_mask
        else:
            abs_err_mask = (data[k] < float(v))
            mask = mask & abs_err_mask
    return mask


def choose_columns(pargs):
    data = pd.read_csv(pargs.filename, index_col=0)
    # print(len(data))
    data_res = pd.DataFrame()
    data_err = pd.DataFrame()

    if 'position' in data:
        position = data['position']
    else:
        position = np.arange(len(data))

    if pargs.include:
        for i in pargs.include:
            try:
                data_res[i] = data[i]
            except KeyError:
                print('WARNING: No column named ', i)
                data_res[i] = pd.Series(np.nan, index=data.index)

        if pargs.errors:
            for i, e in zip(pargs.include, pargs.errors):
                try:
                    data_err[i] = data[e]
                except KeyError:
                    data_err[i] = data_res[i] * 0
        else:
            for i in pargs.include:
                e = i+'_err'
                try:
                    data_err[i] = data[e]
                except KeyError:
                    data_err[i] = data_res[i] * 0
    else:
        names = colnames(data)
        for i in names:
            if not i in pargs.exclude:
                e = i + '_err'
                data_res[i] = data[i]
                data_err[i] = data[e]

    if pargs.max:
        mask = get_limits_mask(data, pargs.max)
        data_res = data_res[mask]
        data_err = data_err[mask]
        position = position[mask]


    # print(data_res)
    # print(data_err)
    return data_res, data_err, position


def get_slit_coords(pargs):
    data = pd.read_csv(pargs.filename, index_col=0)
    coords = data[['position','RA','DEC']]
    # print(coords)
    return coords


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('filename',
                        help="""csv file with mesured parameters""")
    parser.add_argument('-t', '--title', help="Graph title",
                        default='')
    parser.add_argument('-i', '--image', help="Image fits")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-I', '--include', nargs='+',
                       help="""list of names of columns that should be plotted
                            (default: every <colname> where <colname>_err is presented)""")
    group.add_argument('-E', '--exclude', nargs='+', default=[],
                       help="list of names of columns that should not be plotted")
    parser.add_argument('-e', '--errors', nargs='+',
                        help="""list of names of columns that should be used as errors
                             for columns set in '--include' option. Use '0' to skip column
                             from '--include'. (default: use <colname>_err as error for
                             every <colname> from '--include' if presented)""")
    parser.add_argument('--max', nargs='+',
                        help="""list of pairs <colname> <maxvalue> to exclude points
                        with values in <colname> greater than <maxvalue>. <maxvalue>
                        should be cenvertable to float or has '%%'. In case it has %%,
                        it is considered as a relative error and <colname> must have '_err'
                        at the and and have corresponding <colname2> without '_err').""")
    # parser.add_argument('-dx', type=float, default=0)
    # parser.add_argument('-dy', type=float, default=0)

    pargs = parser.parse_args(args[1:])
    csvname = pargs.filename
    dirname = csvname.split('/')[-2]
    dirpath = '/'.join(csvname.split('/')[:-1])
    title = pargs.title or dirname
    mpl.rcParams["savefig.directory"] = dirpath

    if pargs.image:
        image = fits.open(pargs.image)[0]
    else:
        image = None

    data_val, data_err, position = choose_columns(pargs)
    if image:
        slit_coords = get_slit_coords(pargs)

    plot_csv(data_val, data_err, position, title, image, slit_coords)
    # ax[2].errorbar(meascsv['position'][mask], meascsv['sigma_v'][mask],
    #                meascsv['sigma_v_err'][mask], marker='.', linestyle='')
    # ax[2].set_ylabel(r'$\sigma V_{los}, km/s$', fontsize='x-large')
    # ax[2].set_xlabel('$position, "$', fontsize='x-large')
    return(0)


if __name__ == '__main__':
    import sys
    import argparse
    sys.exit(main(sys.argv))
