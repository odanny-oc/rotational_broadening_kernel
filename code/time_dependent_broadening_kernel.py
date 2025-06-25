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
eclipse_end = 0.5 - tfull / period
veq = 2 * np.pi * Rpl * 1e-2 / period  # m/s


def broadening_kernel_orbital_phase(x, op):
    if not isinstance(op, np.ndarray):
        op = np.array([op])
    kernel_array = np.zeros(shape=(op.shape[0], x.shape[0]))
    op_index = 0
    ref_range = np.array([i for i in x if abs(i) <= veq])
    ref_kernel = np.sqrt(1 - (ref_range / veq) ** 2)
    ref_padding = abs(x.shape[0] - ref_range.shape[0]) // 2
    ref_kernel = np.pad(ref_kernel, ref_padding, "constant")
    normaliser = np.sum(ref_kernel)
    ref_kernel /= normaliser
    ref_kernel[: x.shape[0] // 2] = np.zeros(x.shape[0] // 2)

    for i, op_i in enumerate(op):
        ref_op = op_i
        vel = veq * np.sin(2 * np.pi * ref_op + np.pi / 2)
        if vel == 0:
            kernel = np.zeros(x.shape[0])
            print("Velcity is zero!")
        else:

            range = np.array([i for i in x if abs(i) <= abs(vel)])
            kernel = np.sqrt(1 - (range / abs(vel)) ** 2) / normaliser
            padding = abs(x.shape[0] - range.shape[0]) // 2
            kernel = np.pad(kernel, padding, "constant")
            if vel < 0:
                kernel[x.shape[0] // 2 :] = np.zeros(x.shape[0] // 2 + 1)
                kernel += ref_kernel
            if vel > 0:
                kernel[: x.shape[0] // 2] = np.zeros(x.shape[0] // 2)
                kernel = ref_kernel - kernel
            if op_i > 0:
                kernel = np.flip(kernel)
        kernel_array[i] = kernel
    if kernel_array.shape[0] == 1:
        return kernel_array[0]
    else:
        return kernel_array


resolution = 80000
dv = const.c.value / resolution
points_number = 51
n_exposure = 150
kernel_res = n_exposure // 2
range_vel = 0.07

# orbital_phase = np.linspace(
#     -0.5 + tfull / period, 0.5 - tfull / period, n_exposure
# )  # time/period

orbital_phase = np.linspace(0.35, 0.5 - tfull / period, n_exposure)  # time/period

x = np.linspace(-range_vel, range_vel, points_number) * (points_number // 2) * dv

time_dependent_broadening_kernels = broadening_kernel_orbital_phase(x, orbital_phase)
n_columns = 10
n_rows = int(n_exposure / n_columns)
row = 0
column = 0
fig, ax = plt.subplots(n_rows, n_columns, figsize=(10, 5), sharey="all")
for i in time_dependent_broadening_kernels:
    ax[row][column].plot(i)
    column += 1
    column = column % n_columns
    if column == 0:
        row += 1
        row = row % n_rows

fig, ax = plt.subplots()
ax.plot(x, broadening_kernel_orbital_phase(x, 0.234))

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

convolved_spectrum = np.zeros_like(doppler_shift)

for i in range(n_exposure):
    convolved_spectrum[i] = scisig.fftconvolve(
        doppler_shift[i], time_dependent_broadening_kernels[i], "same"
    )

wl *= 1e4
fig, ax = plt.subplots(2, sharex="all", sharey="all")
ax[0].pcolormesh(wl, orbital_phase, convolved_spectrum)
ax[1].pcolormesh(wl, orbital_phase, doppler_shift)
fig.supxlabel(r"Wavelengths ($\mu$m)")
fig.supylabel(r"Orbital Phase")

vsys = np.linspace(-Kp, Kp, 1000)
Wl = np.outer(1 - vsys * 1000 / const.c.value, wl)
shifted_templates = np.interp(Wl, wl, flux)


CC = np.dot(convolved_spectrum, shifted_templates.T)

fig, ax = plt.subplots(3)
ax[0].pcolormesh(vsys, orbital_phase, CC)
ax[0].set_xlabel("System Velocity (km/s)")
ax[0].set_ylabel("Orbital Phase")

CC_shifted = np.empty(CC.shape)
for i, vel in enumerate(vp):
    CC_shifted[i] = np.interp(vsys + vel, vsys, CC[i])
cc_sum = np.sum(CC_shifted, axis=0)

ax[1].pcolormesh(vsys, orbital_phase, CC_shifted)
ax[1].set_xlabel("System Velocity (km/s)")
ax[1].set_ylabel("Orbital Phase")

ax[2].plot(vsys, cc_sum)
ax[2].set_xlabel("System Velocity (km/s)")
ax[2].set_ylabel("Cross Correlation Sum")

K = np.linspace(-100, 2 * Kp, 1000)

K_vsys_map = np.empty((K.size, vsys.size))

K_vsys_sum_map = np.empty((K.size))

for j, kp in enumerate(K):
    vp = kp * np.sin(2 * np.pi * orbital_phase)
    CC_array = np.empty(CC.shape)
    for i, vel in enumerate(vp):
        CC_array[i] = np.interp(vsys + vel, vsys, CC[i])
    K_vsys_map[j] = np.sum(CC_array, axis=0)
    np.append(K_vsys_sum_map, cc_sum)

fig, ax = plt.subplots()
ax.pcolormesh(vsys, K, K_vsys_map)
ax.set_ylabel(r"$K_p$")
ax.set_xlabel(r"$v_{\text{sys}}$")
ax.axhline(Kp, ls="--", color="red", lw = 0.5)
ax.axvline(0, ls="--", color="red", lw = 0.5)


plt.show()
