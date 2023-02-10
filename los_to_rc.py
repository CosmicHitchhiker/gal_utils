#! /usr/bin/env python3

import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from matplotlib import pyplot as plt
from astropy.visualization import simple_norm
import astropy.units as u
import numpy as np
from astropy.io import fits


def plot_galaxy(image, gal_frame, rel_slit, obj_name=None, masks=None):
    wcs = WCS(image.header)
    plt.figure()
    ax = plt.subplot(projection=wcs)
    overlay = ax.get_coords_overlay(gal_frame)

    # "Стираем" чёрточки по краям картинки
    ax.coords['ra'].set_ticks(color='white')
    ax.coords['dec'].set_ticks(color='white')

    # "Стираем" чёрточки по краям картинки
    overlay['lon'].set_ticks(color='white')
    overlay['lat'].set_ticks(color='white')

    # Подпись условных осей в системе координат галактики
    overlay['lon'].set_axislabel('rel_lon')
    overlay['lat'].set_axislabel('rel_lat')

    overlay.grid(color='white', linestyle='solid', alpha=0.5)

    if masks is None:
        masks = [np.ones(len(rel_slit)).astype(bool)]

    ax.set_title(obj_name)
    norm = simple_norm(image.data, 'linear', percent=99.3)
    ax.imshow(image.data, cmap='bone', norm=norm)
    for mask in masks:
        ax.plot(rel_slit.lon[mask], rel_slit.lat[mask], marker='.',
                linestyle='', transform=ax.get_transform(gal_frame))
    # ax.plot(rel_slit.lon, rel_slit.lat, color='green', marker='.',
    #         linestyle='', transform=ax.get_transform(gal_frame))
    plt.show()


def los_to_rc(data, gal_center, gal_PA, inclination, sys_vel=0, image=None,
              obj_name=None, verr_lim=200):
    H0 = 70 / (1e+6 * u.parsec)
    slit_ra = data['RA']
    slit_dec = data['DEC']
    slit_pos = data['position']
    slit = SkyCoord(slit_ra, slit_dec, frame='icrs', unit=(u.hourangle, u.deg))

    gal_frame = gal_center.skyoffset_frame(rotation=gal_PA)
    rel_slit = slit.transform_to(gal_frame)
    if image is not None:
        plot_galaxy(image, gal_frame, rel_slit, obj_name)

    dist = sys_vel / H0
    # Исправляем за наклон галактики
    rel_slit_corr = SkyCoord(rel_slit.lon / np.cos(inclination), rel_slit.lat,
                             frame=gal_frame)
    # Угловое расстояние точек щели до центра галактики
    # (с поправкой на наклон галактики)
    Separation = rel_slit_corr.separation(gal_center)
    # Физическое расстояние
    R_slit = dist * np.sin(Separation)

    # Угол направления на центр галактики
    gal_frame_center = gal_center.transform_to(gal_frame)
    slit_gal_PA = gal_frame_center.position_angle(rel_slit_corr)

    vel_lon = (data['velocity'].to_numpy() - sys_vel) / np.sin(inclination)
    vel_lon_err = np.abs(data['v_err'].to_numpy() / np.sin(inclination))

    vel_r = vel_lon / np.cos(slit_gal_PA)
    vel_r_err = np.abs(vel_lon_err / np.cos(slit_gal_PA))

    mask = (vel_r_err < verr_lim)

    # lat = np.array(rel_slit_corr.lat.to(u.arcsec)/u.arcsec)
    # minor_ax = np.argmin(np.abs(lat))

    closest_point = np.argmin(np.abs(R_slit))

    first_side = (slit_pos >= slit_pos[closest_point])
    second_side = (slit_pos < slit_pos[closest_point])
    first_side_mask = (first_side & mask)
    second_side_mask = (second_side & mask)

    if image is not None:
        plot_galaxy(image, gal_frame, rel_slit, obj_name,
                    [first_side_mask, second_side_mask])

    plt.figure()
    # plt.ylim(-70, 140)
    plt.title(obj_name)
    plt.ylabel('Radial Velocity, km/s')
    plt.xlabel('R, parsec')
    plt.errorbar(R_slit[first_side_mask] / u.parsec, -vel_r[first_side_mask],
                 yerr=vel_r_err[first_side_mask], linestyle='', marker='.')
    plt.errorbar(R_slit[second_side_mask] / u.parsec, -vel_r[second_side_mask],
                 yerr=vel_r_err[second_side_mask], linestyle='', marker='.')
    plt.show()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('los',
                        help='''csv or similar file with positions and
                        velocities''')
    parser.add_argument('refcenter', nargs=2,
                        help='''coordinates of center of galaxy''')
    parser.add_argument('-v', '--velocity', type=float, default=0.0,
                        help='system velocity')
    parser.add_argument('-p', '--PA', type=float, default=0.0,
                        help='galaxy PA')
    parser.add_argument('-i', '--inclination', type=float, default=0.0,
                        help='inclination of galaxy')
    parser.add_argument('-f', '--frame', default=None,
                        help='frame with image')

    pargs = parser.parse_args(args[1:])

    data = pd.read_csv(pargs.los, index_col=0)
    gal_center = SkyCoord(*pargs.refcenter, unit=(u.hourangle, u.deg))

    if pargs.frame:
        image = fits.open(pargs.frame)[0]
    else:
        image = None

    res = los_to_rc(data, gal_center, pargs.PA * u.deg,
                    pargs.inclination * u.deg, pargs.velocity, image)
    print(res)
    # res.to_csv(pargs.los)

    return 0


if __name__ == '__main__':
    import sys
    import argparse
    sys.exit(main(sys.argv))
