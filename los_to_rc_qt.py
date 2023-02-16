#! /usr/bin/python3

# Copyright (C) 2022 The Qt Company Ltd.
# SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause

import sys

import numpy as np
# from scipy.stats import norm
import matplotlib
from astropy.visualization import simple_norm
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
import pandas as pd
from itertools import zip_longest, chain

# from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtCore import Slot, Signal
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QDoubleSpinBox,
    QVBoxLayout,
    QFormLayout,
    QGridLayout,
    QLineEdit,
    QFileDialog,
    QStyle,
    QAbstractSpinBox,
    QPushButton,
)
matplotlib.use('QtAgg')


def plot_slit_points(ax, rel_slit, masks=None, gal_frame=None, line=None):
    if masks is None:
        masks = [np.ones(len(rel_slit)).astype(bool)]
    for mask in masks:
        ax.plot(rel_slit.ra[mask], rel_slit.dec[mask], marker='.',
                linestyle='', transform=ax.get_transform(gal_frame))


def los_to_rc(data, slit, gal_frame, inclination, sys_vel,
              obj_name=None, verr_lim=200):
    H0 = 70 / (1e+6 * u.parsec)
    slit_pos = data['position']

    gal_center = SkyCoord(0 * u.deg, 0 * u.deg, frame=gal_frame)
    rel_slit = slit.transform_to(gal_frame)

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

    data['Radial_v'] = -vel_r
    data['Radial_v_err'] = vel_r_err
    data['R'] = R_slit / u.parsec
    data['mask1'] = np.array(first_side_mask, dtype=bool)
    data['mask2'] = np.array(second_side_mask, dtype=bool)

    return data


class galaxyImage():
    def __init__(self, figure, image):
        self.wcs = WCS(image.header)
        self.figure = figure
        self.axes_gal = figure.subplots(
            subplot_kw={'projection': self.wcs})
        self.image = image
        self.norm_im = simple_norm(image.data, 'linear', percent=99.3)
        self.slits = None
        self.masks = None
        self.slit_draws = None
        self.plot_galaxy()

    def plot_galaxy(self, gal_frame=None):
        self.axes_gal.clear()
        self.axes_gal = self.figure.subplots(
            subplot_kw={'projection': self.wcs})
        self.axes_gal.imshow(self.image.data, cmap='bone', norm=self.norm_im)
        # "Стираем" чёрточки по краям картинки
        self.axes_gal.coords['ra'].set_ticks(color='white')
        self.axes_gal.coords['dec'].set_ticks(color='white')

        if gal_frame is not None:
            self.gal_frame = gal_frame
            self.overlay = self.axes_gal.get_coords_overlay(gal_frame)
            # "Стираем" чёрточки по краям картинки
            self.overlay['lon'].set_ticks(color='white')
            self.overlay['lat'].set_ticks(color='white')
            self.overlay['lon'].set_ticklabel(alpha=0)
            self.overlay['lat'].set_ticklabel(alpha=0)
            self.overlay.grid(color='white', linestyle='solid', alpha=0.5)

        if self.slits is not None:
            self.plot_slit(self.slits, self.masks)

        # plot_galaxy(self.axes_gal, self.image, self.gal_frame)

    def plot_slit(self, slits, masks):
        self.slits = slits
        self.masks = masks
        for line in self.axes_gal.lines:
            self.axes_gal.lines.remove(line)

        for slit, mask in zip(slits, masks):
            plot_slit_points(self.axes_gal, slit, mask, 'icrs')


class csvPlot():
    def __init__(self, data, figure):
        self.data = data
        self.slits = []
        for dat in self.data:
            slit_ra = dat['RA']
            slit_dec = dat['DEC']
            self.slits.append(SkyCoord(slit_ra, slit_dec, frame='icrs',
                                       unit=(u.hourangle, u.deg)))
        self.axes_plot = figure.subplots()

    def calc_rc(self, gal_frame, inclination, sys_vel):
        self.axes_plot.clear()
        self.masks = []
        for dat, slit in zip(self.data, self.slits):
            dat = los_to_rc(dat, slit, gal_frame, inclination, sys_vel)
            self.masks.append([dat['mask1'].to_numpy(),
                               dat['mask2'].to_numpy()])
        self.plot_rc()
        return self.slits, self.masks

    def plot_rc(self):
        self.axes_plot.set_ylabel('Radial Velocity, km/s')
        self.axes_plot.set_xlabel('R, parsec')
        for dat, mask in zip(self.data, self.masks):
            verr = dat['Radial_v_err'].to_numpy()
            mask1, mask2 = mask
            self.axes_plot.errorbar(
                dat['R'][mask1],
                dat['Radial_v'][mask1],
                yerr=verr[mask1],
                linestyle='',
                marker='.')
            self.axes_plot.errorbar(
                dat['R'][mask2],
                dat['Radial_v'][mask2],
                yerr=verr[mask2],
                linestyle='',
                marker='.')


class OpenFile(QWidget):
    changed_path = Signal(str)

    def __init__(self, parent=None, text=None, tt=None, mode='n'):
        super().__init__(parent)

        self.fits_box = QLineEdit()
        self.fits_box.setToolTip(tt)
        self._open_folder_action = self.fits_box.addAction(
            qApp.style().standardIcon(QStyle.SP_DirOpenIcon),
            QLineEdit.TrailingPosition)
        self.fits_box.setPlaceholderText(text)
        self.files = None
        self.mode = mode
        self.dir = "/home/astrolander/Documents/Work/DATA"

        vlayout = QVBoxLayout(self)
        vlayout.addWidget(self.fits_box)
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setSpacing(0)

        self._open_folder_action.triggered.connect(self.on_open_folder)
        self.fits_box.editingFinished.connect(self.check_line)

    @Slot()
    def check_line(self):
        self.files = self.fits_box.text().split(',')
        if self.mode != 'n':
            self.files = self.files[0]
            self.dir = "/".join(self.files.split('/')[:-1])
            self.changed_path.emit(self.dir)
        else:
            self.files = [x.strip() for x in self.files]
            self.dir = "/".join(self.files[0].split('/')[:-1])
            self.changed_path.emit(self.dir)

    @Slot()
    def on_open_folder(self):
        regexps = "All (*)"
        if self.mode == 'n':
            files_path = QFileDialog.getOpenFileNames(self, "Fits", self.dir,
                                                      regexps)[0]
            self.dir = "/".join(files_path[0].split('/')[:-1])
            self.changed_path.emit(self.dir)
        elif self.mode == 'o':
            files_path = QFileDialog.getOpenFileName(self, "Fits", self.dir,
                                                     regexps)[0]
            self.dir = "/".join(files_path.split('/')[:-1])
            self.changed_path.emit(self.dir)
        elif self.mode == 'w':
            files_path = QFileDialog.getSaveFileName(self, "Fits", self.dir,
                                                     regexps)[0]
            self.dir = "/".join(files_path.split('/')[:-1])
            self.changed_path.emit(self.dir)
            ext = files_path.split('.')
            if len(ext) < 2:
                files_path = files_path + '.fits'

        if files_path:
            if self.mode == 'n':
                self.files = files_path.copy()
                self.fits_box.setText(', '.join(files_path))
            else:
                self.files = files_path
                self.fits_box.setText(files_path)

    def fill_string(self, string):
        self.fits_box.setText(string)
        self.check_line()


class radecSpinBox(QAbstractSpinBox):
    valueChanged = Signal(Angle)

    def __init__(self, parent=None, radec='dec', value=0):
        super().__init__(parent)

        self.line = QLineEdit()
        self.setLineEdit(self.line)

        if radec == 'dec':
            self.unit = u.deg
            self.step = Angle('0:0:1', unit=u.deg)
        elif radec == 'ra':
            self.unit = u.hourangle
            self.step = Angle('0:0:0.1', unit=u.hourangle)

        self.setAccelerated(True)
        self.angle = Angle(value, unit=self.unit)

        self.editingFinished.connect(self.valueFromText)

        self.line.setText(self.textFromValue(self.angle.value))

    def textFromValue(self, val):
        return self.angle.to_string(unit=self.unit, sep=':')

    @Slot()
    def valueFromText(self):
        text = self.text()
        self.angle = Angle(text, unit=self.unit)
        self.line.setText(self.textFromValue(self.angle.value))
        self.valueChanged.emit(self.angle)
        return self.angle.value

    def stepEnabled(self):
        ret = QAbstractSpinBox.StepNone
        ret |= QAbstractSpinBox.StepUpEnabled
        ret |= QAbstractSpinBox.StepDownEnabled
        return ret

    def stepBy(self, steps):
        self.angle += steps * self.step
        self.line.setText(self.textFromValue(self.angle.value))
        self.valueChanged.emit(self.angle)

    def getAngle(self):
        return self.angle

    def setValue(self, value):
        self.angle = Angle(value, unit=self.unit)
        self.line.setText(self.textFromValue(self.angle.value))
        self.valueChanged.emit(self.angle)


class PlotWidget(QWidget):
    def __init__(self, parent=None, csv=None, frame=None, refcenter=None, PA=0.,
                 inclination=0., velocity=0.):
        super().__init__(parent)

        self.gal_changed = False
        self.csv_changed = False

        # create widgets
        self.plot_fig = FigureCanvas(Figure(figsize=(5, 3)))
        self.gal_fig = FigureCanvas(Figure(figsize=(5, 3)))
        self.toolbar_plot = NavigationToolbar2QT(self.plot_fig, self)
        self.toolbar_gal = NavigationToolbar2QT(self.gal_fig, self)

        self.image_field = OpenFile(text='image', mode='o')
        if frame is not None:
            self.image_field.fill_string(frame)
            self.gal_changed = True

        self.csv_field = OpenFile(text='csv', mode='n')
        if csv is not None:
            csv = ', '.join(csv)
            self.csv_field.fill_string(csv)
            self.csv_changed = True

        self.i_input = QDoubleSpinBox()
        self.i_input.setKeyboardTracking(False)
        self.i_input.setValue(inclination)

        self.PA_input = QDoubleSpinBox()
        self.PA_input.setKeyboardTracking(False)
        self.PA_input.setMaximum(360.0)
        self.PA_input.setValue(PA)

        self.ra_input = radecSpinBox(radec='ra')
        self.dec_input = radecSpinBox(radec='dec')
        self.ra_input.setKeyboardTracking(False)
        self.dec_input.setKeyboardTracking(False)
        if refcenter is not None:
            self.ra_input.setValue(refcenter[0])
            self.dec_input.setValue(refcenter[1])

        self.vel_input = QDoubleSpinBox()
        self.vel_input.setKeyboardTracking(False)
        self.vel_input.setMaximum(500000)
        self.vel_input.setValue(velocity)

        self.redraw_button = QPushButton(text='Redraw')

        # Layout
        left_layout = QFormLayout()
        left_layout.addRow(self.csv_field)
        left_layout.addRow('i', self.i_input)
        left_layout.addRow('RA', self.ra_input)
        left_layout.addRow('system velocity', self.vel_input)
        right_layout = QFormLayout()
        right_layout.addRow(self.image_field)
        right_layout.addRow('PA', self.PA_input)
        right_layout.addRow('DEC', self.dec_input)
        right_layout.addRow(self.redraw_button)

        glayout = QGridLayout()
        glayout.addWidget(self.toolbar_plot, 0, 0)
        glayout.addWidget(self.toolbar_gal, 0, 1)
        glayout.addWidget(self.plot_fig, 1, 0)
        glayout.addWidget(self.gal_fig, 1, 1)
        glayout.addLayout(left_layout, 2, 0)
        glayout.addLayout(right_layout, 2, 1)
        self.setLayout(glayout)

        self.galIm = None

        self.redraw_button.clicked.connect(self.redraw)
        self.csv_field.changed_path.connect(self.csvChanged)
        self.image_field.changed_path.connect(self.galChanged)
        self.PA_input.valueChanged.connect(self.galFrameChanged)
        self.ra_input.valueChanged.connect(self.galFrameChanged)
        self.dec_input.valueChanged.connect(self.galFrameChanged)
        self.vel_input.valueChanged.connect(self.kinematicsChanged)
        self.i_input.valueChanged.connect(self.kinematicsChanged)

    @Slot()
    def galChanged(self):
        self.gal_changed = True

    @Slot()
    def csvChanged(self):
        self.csv_changed = True

    @Slot()
    def galFrameChanged(self):
        self.updateValues()
        self.galIm.plot_galaxy(self.gal_frame)
        self.csvGraph.calc_rc(self.gal_frame, self.inclination, self.sys_vel)
        self.gal_fig.draw()
        self.plot_fig.draw()

    @Slot()
    def kinematicsChanged(self):
        self.updateValues()
        self.csvGraph.calc_rc(self.gal_frame, self.inclination, self.sys_vel)
        self.plot_fig.draw()

    @Slot()
    def redraw(self):
        """ Update the plot with the current input values """
        self.updateValues()

        if self.gal_changed:
            self.gal_fig.figure.clear()
            image = fits.open(self.image_field.files)[0]
            self.galIm = galaxyImage(self.gal_fig.figure, image)
            self.galIm.plot_galaxy(self.gal_frame)
            self.gal_changed = False

        if self.csv_changed:
            self.plot_fig.figure.clear()
            data = [pd.read_csv(x) for x in self.csv_field.files]
            self.csvGraph = csvPlot(data, self.plot_fig.figure)
            slits, masks = self.csvGraph.calc_rc(self.gal_frame,
                                                 self.inclination,
                                                 self.sys_vel)
            if self.galIm is not None:
                self.galIm.plot_galaxy(self.gal_frame)
                self.galIm.plot_slit(slits, masks)
            self.csv_changed = False

        self.gal_fig.draw()
        self.plot_fig.draw()

    def updateValues(self):
        self.inclination = self.i_input.value() * u.deg
        self.PA = self.PA_input.value() * u.deg
        self.gal_center = SkyCoord(self.ra_input.getAngle(),
                                   self.dec_input.getAngle(),
                                   frame='icrs')
        self.sys_vel = self.vel_input.value()
        self.gal_frame = self.gal_center.skyoffset_frame(rotation=self.PA)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--csv', nargs='+', default=None,
                        help='''csv or similar file with positions and
                        velocities''')
    parser.add_argument('-r', '--refcenter', nargs=2, default=None,
                        help='''coordinates of center of galaxy''')
    parser.add_argument('-v', '--velocity', type=float, default=0.0,
                        help='system velocity')
    parser.add_argument('-p', '--PA', type=float, default=0.0,
                        help='galaxy PA')
    parser.add_argument('-i', '--inclination', type=float, default=0.0,
                        help='inclination of galaxy')
    parser.add_argument('-f', '--frame', default=None,
                        help='frame with image')
    pargs = parser.parse_args(sys.argv[1:])

    app = QApplication(sys.argv)
    w = PlotWidget(None, pargs.csv, pargs.frame, pargs.refcenter, pargs.PA,
                   pargs.inclination, pargs.velocity)
    w.show()
    sys.exit(app.exec())
