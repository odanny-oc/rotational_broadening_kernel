import numpy as np
import matplotlib.pyplot as plt
import os
from functions import Time_Dependent_Spectrum
from functions import Cross_Correlator
from functions import Kp_vsys_Plotter
from functions import maxIndex
from functions import Kp_vsys_Map_from_Flux
from astropy import constants as const
import scipy.signal as scisig
from petitRADTRANS.spectral_model import SpectralModel
from petitRADTRANS.math import resolving_space
from petitRADTRANS.planet import Planet

home_path = os.environ["HOME"]

local_path = home_path + "/exoplanet_atmospheres/code"

wasp121b_spectrum = np.load(
    os.path.join(local_path, "wasp121b.npz")
)  # relative flux (normalised) and wavelength in microns

wasp121_post_data = np.load(
    local_path + "/crires_posteclipse_WASP121_2021-12-15_processed.npz"
)
wavelength_grid = wasp121_post_data["W"][8] * 1e-4  # resolution ~ 300000


# Planet and Star Parameters
# SI (and km)

Rpl = wasp121b_spectrum["radius_planet"]
Rs = wasp121b_spectrum["radius_star"]
Mpl = wasp121b_spectrum["mass_planet"]
Ms = wasp121b_spectrum["mass_star"]
a = wasp121b_spectrum["semi_major_axis"]
period = wasp121b_spectrum["period"]
eccen = 0
observer_angle = wasp121b_spectrum["observer_angle"]
Kp = 2 * np.pi / period * a * np.sin(observer_angle) / (np.sqrt(1 - eccen**2))
b = wasp121b_spectrum["impact_parameter"]

tdur = period / np.pi * np.arcsin(Rs / a * np.sqrt((1 + Rpl / Rs) ** 2 - b**2))  # Ttot
tfull = (
    period / np.pi * np.arcsin(Rs / a * np.sqrt((1 - Rpl / Rs) ** 2 - b**2))
)  # Tfull
eclipse_end = 0.5 - tfull / period
# veq = 2 * np.pi * Rpl / period  # km/s
veq = 50


def broadening_kernel_orbital_phase(x, op):
    if not isinstance(op, np.ndarray):
        op = np.array([op])
    kernel_array = np.zeros(shape=(op.shape[0], x.shape[0]))
    ref_range = np.array([i for i in x if abs(i) <= veq])
    ref_kernel = np.sqrt(1 - (ref_range / veq) ** 2)
    ref_padding = abs(x.shape[0] - ref_range.shape[0]) // 2
    ref_kernel = np.pad(ref_kernel, ref_padding, "constant")
    normaliser = np.sum(ref_kernel)
    ref_kernel /= normaliser
    full_kernel = ref_kernel.copy()
    ref_kernel[: x.shape[0] // 2] = np.zeros(x.shape[0] // 2)
    period_scaler = period / (4 * (tdur - tfull) / 2)

    for i, op_i in enumerate(op):
        if op_i < 0:
            ref_op = op_i + 1
        else:
            ref_op = op_i

        vel = veq * np.cos(2 * np.pi * ref_op)

        if vel == 0:
            kernel = ref_kernel
            if ref_op < 0.5:
                kernel = np.flip(kernel)
        else:
            if abs(ref_op - 0.5) < tdur / (2 * period) or abs(ref_op + 0.5) < tdur / (
                2 * period
            ):
                if abs(ref_op - 0.5) < tfull / (2 * period) or abs(
                    ref_op + 0.5
                ) < tfull / (2 * period):
                    kernel = np.zeros(x.shape[0])
                else:
                    if ref_op > 0:
                        vel = veq * np.cos(
                            (2 * np.pi * (ref_op - 0.5)) * (period_scaler)
                        )
                    else:
                        vel = veq * np.cos(
                            (2 * np.pi * (ref_op + 0.5)) * (period_scaler)
                        )
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
                    if ref_op > 0.5:
                        kernel = np.flip(kernel)
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
                if ref_op < 0.5:
                    kernel = np.flip(kernel)
        kernel_array[i] = kernel
    if kernel_array.shape[0] == 1:
        return kernel_array[0], full_kernel, normaliser
    else:
        return kernel_array, full_kernel


resolution = 400000
dv = const.c.value * 1e-3 / resolution
points_number = 51
n_exposure = 300
kernel_res = n_exposure // 2
range_vel = 5

orbital_phase_pre_eclipse = np.linspace(0.334, 0.425, n_exposure)  # time/period
orbital_phase_post_eclipse = np.linspace(0.539, 0.626, n_exposure)  # time/period

x = np.linspace(-range_vel, range_vel, points_number) * (points_number // 2) * dv

time_dependent_broadening_kernels_pre_eclipse, ref_kernel = (
    broadening_kernel_orbital_phase(x, orbital_phase_pre_eclipse)
)

time_dependent_broadening_kernels_post_eclipse, ref_kernel = (
    broadening_kernel_orbital_phase(x, orbital_phase_post_eclipse)
)

# ecclipse_phase = np.linspace(0, 1, n_exposure)
# ecclipse_kernels = broadening_kernel_orbital_phase(x, ecclipse_phase)
# n_columns = 10
# n_rows = n_exposure // n_columns
# row = 0
# column = 0
# fig, ax = plt.subplots(n_rows, n_columns, figsize=(10, 5), sharey="all")
# for i in time_dependent_broadening_kernels_pre_eclipse:
#     ax[row][column].plot(x, i)
#     column += 1
#     column = column % n_columns
#     if column == 0:
#         row += 1
#         row = row % n_rows

wl = wasp121b_spectrum["wl"]
flux = wasp121b_spectrum["flux"]
wl = wl[300:1000]
flux = flux[300:1000]


flux -= np.mean(flux)

op_test = 0.25

test_kernel, full_kernel, normaliser = broadening_kernel_orbital_phase(x, op_test)

print(np.sum(full_kernel), normaliser, np.sum(test_kernel))
# Renormalise test_kernel
test_kernel *= normaliser
test_kernel /= np.sum(test_kernel)
print(np.sum(test_kernel))

fig, ax = plt.subplots()
ax.plot(wl, flux)
fig.supxlabel("Wavelength (microns)")
fig.supylabel("Flux")

full_flux = scisig.fftconvolve(flux, full_kernel, "same")

test_flux = scisig.fftconvolve(flux, test_kernel, "same")

ax.plot(wl, full_flux)
ax.plot(wl, test_flux)

shift = 5
wavelenght_shift = wl * (1 + (shift * 1000) / const.c.value)
flux_shift = np.interp(wavelenght_shift, wl, full_flux)

ax.plot(wl, flux_shift)

fig, ax = plt.subplots(2, sharex="all", sharey="all")
ax[0].plot(x, test_kernel)
ax[1].plot(x, full_kernel)

vsys = np.linspace(50, 200, 1501)
vsys_kp = np.linspace(-20, 20, 1501)
op = np.linspace(0.33, 0.42, 500)
K = np.linspace(Kp - 50, Kp + 50, 1001)

test_spectrum = Time_Dependent_Spectrum(wl, test_flux, op=op, kp=Kp)

full_spectrum = Time_Dependent_Spectrum(wl, full_flux, op=op, kp=Kp)

shifted_spectrum = Time_Dependent_Spectrum(wl, flux_shift, op=op, kp=Kp)

CC = Cross_Correlator(wl=wl, flux=flux, vsys=vsys * 1000, spectrum=test_spectrum)

CC_shifted = Cross_Correlator(
    wl=wl, flux=flux, vsys=vsys * 1000, spectrum=shifted_spectrum
)

CC_full = Cross_Correlator(
    wl=wl, flux=flux, vsys=vsys * 1000, spectrum=full_spectrum
)

fig, ax = plt.subplots(3)
ax[0].pcolormesh(vsys, op, CC)
ax[1].pcolormesh(vsys, op, CC_full)
ax[2].pcolormesh(vsys, op, CC_shifted)

Kp_vsys_test, _ = Kp_vsys_Plotter(K, vsys, op, CC, vsys_kp)

Kp_vsys_shifted, _ = Kp_vsys_Plotter(K, vsys, op, CC_shifted, vsys_kp)

Kp_vsys_full, _ = Kp_vsys_Plotter(K, vsys, op, CC_full, vsys_kp)

fig, ax = plt.subplots(3)
fig.suptitle("Kp - vsys Plots")
ax[0].set_title(f"Test Kernel (Orbital Phase of {op_test})")
ax[0].pcolormesh(vsys_kp, K, Kp_vsys_test)
ax[0].axhline(
    K[maxIndex(Kp_vsys_test)[0]],
    lw=0.5,
    ls="--",
    color="red",
    label=f"Kp = {K[maxIndex(Kp_vsys_test)[0]]:.2f}",
)
ax[0].axvline(
    vsys_kp[maxIndex(Kp_vsys_test)[1]],
    lw=0.5,
    ls="--",
    color="red",
    label=f"vsys = {vsys_kp[maxIndex(Kp_vsys_test)[1]]:.2f}",
)

ax[0].legend(loc="upper left")

ax[1].set_title(f"Full Kernel")
ax[1].pcolormesh(vsys_kp, K, Kp_vsys_full)
ax[1].axhline(
    K[maxIndex(Kp_vsys_full)[0]],
    lw=0.5,
    ls="--",
    color="red",
    label=f"Kp = {K[maxIndex(Kp_vsys_full)[0]]:.2f}",
)
ax[1].axvline(
    vsys_kp[maxIndex(Kp_vsys_full)[1]],
    lw=0.5,
    ls="--",
    color="red",
    label=f"vsys = {vsys_kp[maxIndex(Kp_vsys_full)[1]]:.2f}",
)

ax[1].legend(loc="upper left")

ax[2].set_title(f"Spectrum Shifted by {shift}km/s")
ax[2].pcolormesh(vsys_kp, K, Kp_vsys_shifted)

ax[2].axhline(
    K[maxIndex(Kp_vsys_shifted)[0]],
    lw=0.5,
    ls="--",
    color="red",
    label=f"Kp = {K[maxIndex(Kp_vsys_shifted)[0]]:.2f}",
)

ax[2].axvline(
    vsys_kp[maxIndex(Kp_vsys_shifted)[1]],
    lw=0.5,
    ls="--",
    color="red",
    label=f"vsys = {vsys_kp[maxIndex(Kp_vsys_shifted)[1]]:.2f}",
)

ax[2].legend(loc="upper left")


plt.show()
