#! /usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from astropy.coordinates import Angle
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.patches import Rectangle
# from astropy.visualization.wcsaxes import SphericalCircle

from matplotlib.transforms import Affine2D
from astropy.visualization.wcsaxes import WCSAxes

from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm


class mySlit(Polygon):
    """
    Create a patch representing a latitude-longitude quadrangle rotated by
    given angle.

    See astropy.visualization.wcsaxes.Quadrangle for details.

    Parameters
    ----------
    anchor : tuple or `~astropy.units.Quantity` ['angle']
        Center of slit.
        This can be either a tuple of two `~astropy.units.Quantity` objects, or
        a single `~astropy.units.Quantity` array with two elements.
    width : `~astropy.units.Quantity` ['angle'], default=1.5*u.arcsec
        The width of the slit.
    height : `~astropy.units.Quantity` ['angle'], default=3*u.arcmin
        The lenght of the slit.
    theta : `~astropy.units.Quantity` ['angle'], default=0*u.deg
        PA of the slit.
    resolution : int, optional
        The number of points that make up each side of the quadrangle -
        increase this to get a smoother quadrangle.
    vertex_unit : `~astropy.units.Unit` ['angle'], default=u.deg
        The units in which the resulting polygon should be defined - this
        should match the unit that the transformation (e.g. the WCS
        transformation) expects as input.

    Notes
    -----
    Additional keyword arguments are passed to `~matplotlib.patches.Polygon`
    """

    def __init__(self, anchor, width=1.5 * u.arcsec, height=3 * u.arcmin,
                 theta=0 * u.deg, resolution=100, vertex_unit=u.deg, **kwargs):

        # Extract longitude/latitude, either from a tuple of two quantities, or
        # a single 2-element Quantity.
        lon_c, lat_c = u.Quantity(anchor).to_value(vertex_unit)
        center = np.array([[lon_c, lat_c]])

        theta = u.Quantity(theta).to_value(u.rad)

        # Convert the quadrangle dimensions to the appropriate units
        width = width.to_value(vertex_unit)
        height = height.to_value(vertex_unit)

        # Corner coordinates
        longitude = lon_c - width * 0.5
        latitude = lat_c - height * 0.5

        # Create progressions in longitude and latitude
        lon_seq = longitude + np.linspace(0, width, resolution + 1)
        lat_seq = latitude + np.linspace(0, height, resolution + 1)

        # Trace the path of the quadrangle
        lon = np.concatenate([lon_seq[:-1],
                              np.repeat(lon_seq[-1], resolution),
                              np.flip(lon_seq[1:]),
                              np.repeat(lon_seq[0], resolution)])
        lat = np.concatenate([np.repeat(lat_seq[0], resolution),
                              lat_seq[:-1],
                              np.repeat(lat_seq[-1], resolution),
                              np.flip(lat_seq[1:])])

        # Create polygon vertices
        vertices = np.array([lon, lat])

        # Rotation matrix
        print(u.Quantity(theta * u.rad).to_value(u.deg))
        rot_matrix = np.array([[np.cos(theta), +np.sin(theta)],
                               [-np.sin(theta), np.cos(theta)]])

        # Rotate Quadrangle
        vertices = vertices - center.T
        vertices = rot_matrix @ vertices
        vertices = vertices + center.T
        vertices = vertices.T

        super().__init__(vertices, **kwargs)


class mySlit2(Polygon):
    """
    Create a patch representing a latitude-longitude quadrangle rotated by
    given angle.

    See astropy.visualization.wcsaxes.Quadrangle for details.

    Parameters
    ----------
    anchor : tuple or `~astropy.units.Quantity` ['angle']
        Center of slit.
        This can be either a tuple of two `~astropy.units.Quantity` objects, or
        a single `~astropy.units.Quantity` array with two elements.
    width : `~astropy.units.Quantity` ['angle'], default=1.5*u.arcsec
        The width of the slit.
    height : `~astropy.units.Quantity` ['angle'], default=3*u.arcmin
        The lenght of the slit.
    theta : `~astropy.units.Quantity` ['angle'], default=0*u.deg
        PA of the slit.
    resolution : int, optional
        The number of points that make up each side of the quadrangle -
        increase this to get a smoother quadrangle.
    vertex_unit : `~astropy.units.Unit` ['angle'], default=u.deg
        The units in which the resulting polygon should be defined - this
        should match the unit that the transformation (e.g. the WCS
        transformation) expects as input.

    Notes
    -----
    Additional keyword arguments are passed to `~matplotlib.patches.Polygon`
    """

    def __init__(self, anchor, width=1.5 * u.arcsec, height=3 * u.arcmin,
                 theta=0 * u.deg, resolution=100, vertex_unit=u.deg, **kwargs):

        # Extract longitude/latitude, either from a tuple of two quantities, or
        # a single 2-element Quantity.
        lon_c, lat_c = u.Quantity(anchor).to_value(vertex_unit)
        center = np.array([[lon_c, lat_c]])

        theta = u.Quantity(theta).to_value(u.rad)

        # Convert the quadrangle dimensions to the appropriate units
        width = width.to_value(vertex_unit)
        height = height.to_value(vertex_unit)

        # Corner coordinates
        longitude = lon_c - width * 0.5
        latitude = lat_c - height * 0.5

        # Create progressions in longitude and latitude
        lon_seq = longitude + np.linspace(0, width, resolution + 1)
        lat_seq = latitude + np.linspace(0, height, resolution + 1)

        # Trace the path of the quadrangle
        lon = np.concatenate([lon_seq[:-1],
                              np.repeat(lon_seq[-1], resolution),
                              np.flip(lon_seq[1:]),
                              np.repeat(lon_seq[0], resolution)])
        lat = np.concatenate([np.repeat(lat_seq[0], resolution),
                              lat_seq[:-1],
                              np.repeat(lat_seq[-1], resolution),
                              np.flip(lat_seq[1:])])

        lon = np.array([lon_seq[0], lon_seq[0], lon_seq[-1], lon_seq[-1]])
        lat = np.array([lat_seq[0], lat_seq[-1], lat_seq[-1], lat_seq[0]])

        # Create polygon vertices
        vertices = np.array([lon, lat]).T

        super().__init__(vertices, **kwargs)

        # print(self.get_xy())
        # print(self.get_path())

        vertices = vertices.T
        # Rotation matrix
        print(u.Quantity(theta * u.rad).to_value(u.deg))
        rot_matrix = np.array([[np.cos(theta), +np.sin(theta)],
                               [-np.sin(theta), np.cos(theta)]])

        # Rotate Quadrangle
        vertices = vertices - center.T
        vertices = rot_matrix @ vertices
        vertices = vertices + center.T
        vertices = vertices.T
        self.set(xy=vertices)


def newWCS(center, angle, fig):
    if u.Quantity(angle).unit != u.dimensionless_unscaled:
        angle = angle.to_value(u.rad)
    else:
        angle = (angle * u.deg).to_value(u.rad)

    if u.Quantity(center).unit != u.dimensionless_unscaled:
        center = u.Quantity(center).to_value(u.deg)

    # Set up an affine transformation
    transform = Affine2D()
    transform.scale(0.01)
    transform.translate(*center)
    transform.rotate(angle)  # radians

    # Set up metadata dictionary
    coord_meta = {}
    coord_meta['name'] = 'lon', 'lat'
    coord_meta['type'] = 'longitude', 'latitude'
    coord_meta['wrap'] = 180, None
    coord_meta['unit'] = u.deg, u.deg
    coord_meta['format_unit'] = None, None

    ax = WCSAxes(fig, [0.1, 0.1, 0.8, 0.8], aspect='equal',
                 transform=transform, coord_meta=coord_meta)
    return(ax)


def my_slit(center, rot=0, dw=1.5, l_slit=3., trans=None, **kwargs):
    if u.Quantity(center).unit != u.dimensionless_unscaled:
        center_deg = u.Quantity(center).to_value(u.deg)
    if u.Quantity(dw).unit == u.dimensionless_unscaled:
        dw = dw * u.arcsec
    if u.Quantity(l_slit).unit == u.dimensionless_unscaled:
        l_slit = l_slit * u.arcmin
    if u.Quantity(rot).unit != u.dimensionless_unscaled:
        rot = rot.to_value(u.deg)

    print(dw)
    dw = u.Quantity(dw).to_value(u.deg)
    l_slit = l_slit.to_value(u.deg)

    angle_deg = (center_deg[0] - dw * 0.5, center_deg[1] - l_slit * 0.5)
    res = Rectangle(angle_deg, dw, l_slit, angle=rot, rotation_point='center',
                    transform=trans, **kwargs)
    return res


def parse_tds_slit(slitname):
    if slitname == '1asec':
        return 1.0
    elif slitname == '1.5asec':
        return 1.5
    elif slitname == '10asec':
        return 10
    raise ValueError('Unknown SLIT keyword')


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('image',
                        help='''fits file with object image and
                                valid WCS keys in header''')
    parser.add_argument('spectrum',
                        help='''fits file with object spectrum and
                                valid position keys in header''')
    parser.add_argument('--ra-corr', type=float, default=0.0,
                        help='''right acession correction for the center
                        of the slit in arcsec''')
    parser.add_argument('--dec-corr', type=float, default=0.0,
                        help='''declination correction for the center
                        of the slit in arcsec''')
    pargs = parser.parse_args(args[1:])

    image = fits.open(pargs.image)[0]
    wcs = WCS(image.header)
    hdr = fits.getheader(pargs.spectrum)

    radec_slit = [Angle(hdr['RA'] + ' hours'), Angle(hdr['DEC'] + ' degrees')]
    if 'SLIT' in hdr:
        slit_width = parse_tds_slit(hdr['SLIT'])
    else:
        slit_width = 1.0

    radec_slit[0] += pargs.ra_corr * u.arcsec
    radec_slit[1] += pargs.dec_corr * u.arcsec

    PA = hdr['POSANG'] * u.deg

    ax = plt.subplot(projection=wcs)
    ax.set_title(hdr['OBJECT'])
    norm = simple_norm(image.data, 'linear', percent=98.0)
    ax.imshow(image.data, cmap='bone', origin='lower', norm=norm)

    slit_center = SkyCoord(*radec_slit, frame='icrs', unit=(u.hourangle, u.deg))
    # PA щели, туда будет направлена ось "широт"
    slit_frame = slit_center.skyoffset_frame(rotation=PA)

    # overlay = ax.get_coords_overlay(slit_frame)

    # s = my_slit(radec_slit, rot=PA, trans=ax.get_transform('icrs'))

    s = mySlit([0 * u.deg, 0 * u.deg], slit_width * u.arcsec, 3.0 * u.arcmin,
               theta=0 * u.deg,
               edgecolor='tab:olive', facecolor='none', lw=0.5,
               transform=ax.get_transform(slit_frame))
    ax.add_patch(s)
    # c = mySlit(radec_slit, 0.1*u.arcsec, 0.1*u.arcmin, theta=(PA-45*u.deg),
    #            edgecolor='red', facecolor='none',
    #            transform=ax.get_transform('icrs'))
    # ax.add_patch(c)
    plt.show()
    return 0


if __name__ == '__main__':
    import sys
    import argparse
    sys.exit(main(sys.argv))
