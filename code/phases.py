import numpy as np
import matplotlib.pyplot as plt
import os
from astropy import constants as const
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.plotlib import plot_radtrans_opacities
from petitRADTRANS import physical_constants as cst, planet
from petitRADTRANS.config import petitradtrans_config_parser
from petitRADTRANS.physics import temperature_profile_function_guillot_global
import scipy.signal as scisig
from petitRADTRANS.spectral_model import SpectralModel
from petitRADTRANS.math import resolving_space
from petitRADTRANS.planet import Planet

# Planet and Star Parameters in cgs!

Rpl = cst.r_jup_mean
Rs = cst.r_sun
Mp = 0.85 * cst.m_jup
Ms = 1.1 * cst.m_sun
a = 0.045 * cst.au  # au to cm
period = 1.5 * (24 * 60**2)  # days to seconds
eccen = 0.1
observer_angle = 90  # degrees
Kp = 2 * np.pi / period * a * np.sin(observer_angle) / (np.sqrt(1 - eccen**2))
b = 0.01

tdur = period / np.pi * np.arcsin(Rs / a * np.sqrt((1 - Rpl / Rs) ** 2 - b**2))
tfull = period / np.pi * np.arcsin(Rs / a * np.sqrt((1 + Rpl / Rs) ** 2 - b**2))

veq = 2 * np.pi * Rpl * 1e-2 / period  # m/s


def broadening_kernel(x, eccen):
    range = np.array([i for i in x if abs(i) <= veq * eccen])
    kernel = np.sqrt(1 - (range / (veq * eccen)) ** 2)
    padding = abs(x.shape[0] - range.shape[0]) // 2
    kernel = np.pad(kernel, padding, "constant")
    return kernel


resolution = 80000
dv = const.c.value / resolution
points_number = 51
n_exposure = 110
kernel_res = n_exposure // 4
range_vel = 0.07

orbital_phase = np.linspace(
    -0.5 + tfull / period, 0.5 - tfull / period, n_exposure
)  # time/period

x = np.linspace(-range_vel, range_vel, points_number) * (points_number // 2) * dv

broadening_kernels_right = np.zeros(shape=(kernel_res, points_number))

# NORMALISATION ISSUE HERE !! will fix later but not normalised at max brightness
for i, j in zip(np.linspace(0.01, 1, kernel_res), range(kernel_res)):
    broadening_kernels_right[j] = broadening_kernel(x, i) / np.sum(
        broadening_kernel(x, 1)
    )
    broadening_kernels_right[j][0 : (points_number // 2)] = np.zeros(points_number // 2)

broadening_kernels_left = np.array([np.flip(i) for i in broadening_kernels_right])

kerneldiff_left = np.array(
    [
        broadening_kernels_left[-1] - broadening_kernels_left[i]
        for i in range(0, kernel_res)
    ]
)

kerneldiff_right = np.array(
    [
        broadening_kernels_right[-1] - broadening_kernels_right[i]
        for i in range(0, kernel_res)
    ]
)
kerneldiff_left = np.flip(kerneldiff_left)

kerneldiff_right = np.flip(kerneldiff_right)


broadening_kernels_left_normal = broadening_kernels_left
for i in range(kernel_res):
    broadening_kernels_left_normal[i][points_number // 2] = 0

broadening_kernels_right_normal = np.array(
    [np.flip(i) for i in broadening_kernels_left_normal]
)

diff_kernels = np.concatenate(
    (
        kerneldiff_right,
        broadening_kernels_right,
        kerneldiff_left,
        broadening_kernels_left,
    ),
    axis=0,
)

broadening_kernels_op = np.tile(
    broadening_kernels_right[-1] + broadening_kernels_left[-2], (n_exposure, 1)
)

# for i in range(n_exposure // 4 - 1, n_exposure//2):
#    broadening_kernels_op[i] = broadening_kernels_right[-1]
#
# for i in range(n_exposure // 2, n_exposure):
#    broadening_kernels_op[i] = np.zeros_like(broadening_kernels_op[i])


for i in range(1, n_exposure - 1):
    if i <= kerneldiff_left.shape[0]:
        broadening_kernels_op[i] -= diff_kernels[i - 1]
    elif i <= kerneldiff_left.shape[0] + broadening_kernels_right.shape[0]:
        broadening_kernels_op[i] = (
            broadening_kernels_op[kerneldiff_left.shape[0]] - diff_kernels[(i - 1)]
        )
    elif (
        i
        <= kerneldiff_left.shape[0]
        + broadening_kernels_right.shape[0]
        + kerneldiff_right.shape[0]
    ):
        broadening_kernels_op[i] = (
            broadening_kernels_op[
                kerneldiff_left.shape[0] + broadening_kernels_right.shape[0]
            ]
            + diff_kernels[(i - 1)]
        )
    else:
        broadening_kernels_op[i] = (
            broadening_kernels_op[
                kerneldiff_left.shape[0]
                + broadening_kernels_right.shape[0]
                + kerneldiff_left.shape[0]
            ]
            + diff_kernels[(i - 1)]
        )

n_columns = 10
n_rows = int(n_exposure / n_columns)
row = 0
column = 0
fig, ax = plt.subplots(n_rows, n_columns, figsize=(10, 5), sharey ='all')
for i in broadening_kernels_op:
    ax[row][column].plot(i)
    column += 1
    column = column % n_columns
    if column == 0:
        row += 1
        row = row % n_rows

n_columns = 10
n_rows = int(n_exposure / n_columns)
row = 0
column = 0
fig, ax = plt.subplots(n_rows, n_columns, figsize=(10, 5), sharey="all")
for i in diff_kernels:
    ax[row][column].plot(i)
    column += 1
    column = column % n_columns
    if column == 0:
        row += 1
        row = row % n_rows

# fig, ax = plt.subplots(figsize=(10, 5))

# ax.plot(x, broadening_kernels_op[0])
# ax.vlines(x=[-veq, veq], ymin=0, ymax=0.033)
# for i in np.linspace(0.01, 1, 10
#    ax.plot(x, np.sqrt(1 - (x / i) ** 2))


wl, flux = np.load("Fe_spectrumwl.npy")
flux -= np.mean(flux)
# fig, ax = plt.subplots()
# ax.plot(wl, flux)
#
#
Kp = 293
vp = Kp * np.sin(orbital_phase * 2 * np.pi)
W = np.outer(1 - vp * 1000 / const.c.value, wl)
doppler_shift = np.interp(W, wl, flux)


convoled_spectrum = np.zeros_like(doppler_shift)

print(doppler_shift.shape)
for i in range(n_exposure):
    convoled_spectrum[i] = scisig.fftconvolve(
        doppler_shift[i], broadening_kernels_op[i], "same"
    )

wl *= 1e4
fig, ax = plt.subplots(2, sharex="all", sharey="all")
ax[0].pcolormesh(wl, orbital_phase, convoled_spectrum)
ax[1].pcolormesh(wl, orbital_phase, doppler_shift)
fig.supxlabel(r"Wavelengths ($\mu$m)")
fig.supylabel(r"Orbital Phase")

plt.show()
