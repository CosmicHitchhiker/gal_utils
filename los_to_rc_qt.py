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
    QLineEdit,
    QPushButton,
)
matplotlib.use('QtAgg')


def plot_galaxy(ax, image, gal_frame=None):
    overlay = ax.get_coords_overlay(gal_frame)

    # "Стираем" чёрточки по краям картинки
    ax.coords['ra'].set_ticks(color='white')
    ax.coords['dec'].set_ticks(color='white')

    # # "Стираем" чёрточки по краям картинки
    overlay['lon'].set_ticks(color='white')
    overlay['lat'].set_ticks(color='white')

    # # Подпись условных осей в системе координат галактики
    overlay['lon'].set_axislabel('rel_lon')
    overlay['lat'].set_axislabel('rel_lat')

    overlay.grid(color='white', linestyle='solid', alpha=0.5)

    norm_im = simple_norm(image.data, 'linear', percent=99.3)
    ax.imshow(image.data, cmap='bone', norm=norm_im)


def plot_slit_points(ax, rel_slit, masks=None, gal_frame=None):
    if masks is None:
        masks = [np.ones(len(rel_slit)).astype(bool)]

    for mask in masks:
        ax.plot(rel_slit.lon[mask], rel_slit.lat[mask], marker='.',
                linestyle='', transform=ax.get_transform(gal_frame))


def los_to_rc(data, gal_center, gal_PA, inclination, sys_vel, ax=None,
              obj_name=None, verr_lim=200):
    H0 = 70 / (1e+6 * u.parsec)
    slit_ra = data['RA']
    slit_dec = data['DEC']
    slit_pos = data['position']
    slit = SkyCoord(slit_ra, slit_dec, frame='icrs', unit=(u.hourangle, u.deg))

    gal_frame = gal_center.skyoffset_frame(rotation=gal_PA)
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

    # plt.ylim(-70, 140)
    ax.set_ylabel('Radial Velocity, km/s')
    ax.set_xlabel('R, parsec')
    ax.errorbar(R_slit[first_side_mask] / u.parsec, -vel_r[first_side_mask],
                yerr=vel_r_err[first_side_mask], linestyle='', marker='.')
    ax.errorbar(R_slit[second_side_mask] / u.parsec, -vel_r[second_side_mask],
                yerr=vel_r_err[second_side_mask], linestyle='', marker='.')

    return rel_slit, [first_side_mask, second_side_mask]

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
            # print(files_path)

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
    def __init__(self, parent=None, radec='dec', value=0):
        super().__init__(parent)

        self.line = QLineEdit()
        self.setLineEdit(self.line)

        if radec == 'dec':
            self.unit = u.deg
        elif radec == 'ra':
            self.unit = u.hourangle

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
        return self.angle.value

    def stepEnabled(self):
        ret = QAbstractSpinBox.StepNone
        ret |= QAbstractSpinBox.StepUpEnabled
        ret |= QAbstractSpinBox.StepDownEnabled
        return ret

    def stepBy(self, steps):
        self.angle += steps * Angle('0:0:0.1', unit=self.unit)
        self.line.setText(self.textFromValue(self.angle.value))
        print(self.angle)

    def getAngle(self):
        return self.angle


class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        #  create widgets
        self.plot_fig = FigureCanvas(Figure(figsize=(5, 3)))
        self.gal_fig = FigureCanvas(Figure(figsize=(5, 3)))
        # self.axes_plot = self.rc_graph.figure.subplots()
# self.axes_gal = self.gal_image.figure.subplots(subplot_kw={'projection': wcs})
        self.toolbar_plot = NavigationToolbar2QT(self.plot_fig, self)
        self.toolbar_gal = NavigationToolbar2QT(self.gal_fig, self)
        self.image_field = OpenFile(text='image', mode='o')
        self.csv_field = OpenFile(text='csv', mode='n')
        self.i_input = QDoubleSpinBox()
        self.PA_input = QDoubleSpinBox()
        self.PA_input.setMaximum(360.0)
        self.ra_input = radecSpinBox(radec='ra')
        self.dec_input = radecSpinBox()
        self.vel_input = QDoubleSpinBox()
        self.vel_input.setMaximum(50000)
        self.redraw_button = QPushButton(text='Redraw')

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

        #  Create layout
        glayout = QGridLayout()
        glayout.addWidget(self.toolbar_plot, 0, 0)
        glayout.addWidget(self.toolbar_gal, 0, 1)
        glayout.addWidget(self.plot_fig, 1, 0)
        glayout.addWidget(self.gal_fig, 1, 1)
        glayout.addLayout(left_layout, 2, 0)
        glayout.addLayout(right_layout, 2, 1)
        # glayout.addWidget(self.csv_field, 2, 0)
        # glayout.addWidget(self.image_field, 2, 1)
        # glayout.addWidget(self.i_input, 3, 0)
        # glayout.addWidget(self.PA_input, 3, 1)
        # glayout.addWidget(self.ra_input, 4, 0)
        # glayout.addWidget(self.dec_input, 4, 1)
        # glayout.addWidget(self.vel_input, 5, 0)
        # glayout.addWidget(self.redraw_button, 5, 1)

        self.setLayout(glayout)

        # connect inputs with on_change method
        # self.mu_input.valueChanged.connect(self.on_change)
        # self.std_input.valueChanged.connect(self.on_change)
        self.redraw_button.clicked.connect(self.redraw)

        # plot_galaxy(self.axes_gal, image_file)

    @Slot()
    def redraw(self):
        """ Update the plot with the current input values """
        self.updateValues()

        self.gal_fig.figure.clear()
        self.plot_fig.figure.clear()

        if self.data is not None:
            self.axes_plot = self.plot_fig.figure.subplots()
            rel_slit, masks = los_to_rc(self.data, self.gal_center, self.PA,
                                        self.inclination, self.sys_vel,
                                        ax=self.axes_plot)

        if self.image is not None:
            wcs = WCS(self.image.header)
            self.axes_gal = self.gal_fig.figure.subplots(
                subplot_kw={'projection': wcs})
            plot_galaxy(self.axes_gal, self.image, gal_frame=self.gal_frame)
            if self.data is not None:
                plot_slit_points(self.axes_gal, rel_slit, masks, self.gal_frame)

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
        print(self.gal_frame)

        self.image_name = self.image_field.files
        if self.image_name is not None:
            self.image = fits.open(self.image_name)[0]
            self.wcs = WCS(self.image.header)
            print(self.wcs)
        else:
            self.image = None
            self.wcs = None

        self.csv_name = self.csv_field.files
        if self.csv_name is not None:
            self.data = pd.read_csv(self.csv_name[0])
        else:
            self.data = None


if __name__ == "__main__":

    app = QApplication(sys.argv)
    w = PlotWidget()
    w.show()
    sys.exit(app.exec())
