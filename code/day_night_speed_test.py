import numpy as np
import matplotlib.pyplot as plt
import os
from functions import Kp_vsys_Plotter as Kplotter
from astropy import constants as const
from petitRADTRANS.plotlib import plot_radtrans_opacities
from petitRADTRANS import physical_constants as cst, planet
from petitRADTRANS.config import petitradtrans_config_parser
from petitRADTRANS.physics import temperature_profile_function_guillot_global
import scipy.signal as scisig
from petitRADTRANS.spectral_model import SpectralModel
from petitRADTRANS.math import resolving_space
from petitRADTRANS.planet import Planet
import time

home_path = os.environ["HOME"]

local_path = home_path + "/exoplanet_atmospheres/code"

day_night_atmosphere = np.load(
    os.path.join(local_path, "day_night_atmosphere.npz")
)  # relative flux (normalised) and wavelength in microns


# Planet and Star Parameters
# SI (and km)

Rpl = day_night_atmosphere["radius_planet"]
Rs = day_night_atmosphere["radius_star"]
Mpl = day_night_atmosphere["mass_planet"]
Ms = day_night_atmosphere["mass_star"]
a = day_night_atmosphere["semi_major_axis"]
period = day_night_atmosphere["period"]
eccen = 0
observer_angle = day_night_atmosphere["observer_angle"]
Kp = 2 * np.pi / period * a * np.sin(observer_angle) / (np.sqrt(1 - eccen**2))
b = day_night_atmosphere["impact_parameter"]

tdur = period / np.pi * np.arcsin(Rs / a * np.sqrt((1 + Rpl / Rs) ** 2 - b**2))  # Ttot
tfull = (
    period / np.pi * np.arcsin(Rs / a * np.sqrt((1 - Rpl / Rs) ** 2 - b**2))
)  # Tfull
eclipse_end = 0.5 - tfull / period
veq = 2 * np.pi * Rpl / period  # km/s


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
        return kernel_array[0], full_kernel
    else:
        return kernel_array, full_kernel


resolution = 200000
dv = const.c.value * 1e-3 / resolution
points_number = 51
n_exposure = 300
kernel_res = n_exposure // 2
range_vel = 0.2

orbital_phase_pre_eclipse = np.linspace(0.334, 0.425, n_exposure)  # time/period
orbital_phase_post_eclipse = np.linspace(0.539, 0.626, n_exposure)  # time/period

x = np.linspace(-range_vel, range_vel, points_number) * (points_number // 2) * dv

time_dependent_broadening_kernels_pre_eclipse, ref_kernel = (
    broadening_kernel_orbital_phase(x, orbital_phase_pre_eclipse)
)

anti_kernels_pre_eclipse, ref_kernel = broadening_kernel_orbital_phase(
    x, orbital_phase_pre_eclipse - 0.5
)

time_dependent_broadening_kernels_post_eclipse, ref_kernel = (
    broadening_kernel_orbital_phase(x, orbital_phase_post_eclipse)
)
anti_kernels_post_eclipse, ref_kernel = broadening_kernel_orbital_phase(
    x, orbital_phase_post_eclipse - 0.5
)

wl = day_night_atmosphere["wl_day"]

flux_day = day_night_atmosphere["flux_day"]
flux_night = day_night_atmosphere["flux_night"]

flux_day -= np.mean(flux_day)
flux_night -= np.mean(flux_night)

vp_pre_eclipse = Kp * np.sin(orbital_phase_pre_eclipse * 2 * np.pi)
W_pre = np.outer(1 - vp_pre_eclipse * 1000 / const.c.value, wl)
spectrum_pre_eclipse_day = np.interp(W_pre, wl, flux_day)
spectrum_pre_eclipse_night = np.interp(W_pre, wl, flux_night)

convolved_spectrum_pre_eclipse_day = np.zeros_like(spectrum_pre_eclipse_day)
convolved_spectrum_pre_eclipse_night = np.zeros_like(spectrum_pre_eclipse_night)

for i in range(n_exposure):
    convolved_spectrum_pre_eclipse_day[i] = scisig.fftconvolve(
        spectrum_pre_eclipse_day[i],
        time_dependent_broadening_kernels_pre_eclipse[i],
        "same",
    )

for i in range(n_exposure):
    convolved_spectrum_pre_eclipse_night[i] = scisig.fftconvolve(
        spectrum_pre_eclipse_night[i],
        anti_kernels_pre_eclipse[i],
        "same",
    )

total_convolved_spectrum_pre = (
    convolved_spectrum_pre_eclipse_night + convolved_spectrum_pre_eclipse_day
)

# fig, ax = plt.subplots()
# ax.pcolormesh(wl, orbital_phase_pre_eclipse, total_convolved_spectrum)
# fig.supxlabel(r"Wavelengths ($\mu$m)")
# fig.supylabel(r"Orbital Phase")
# plt.show()

vsys = np.linspace(-200, 200, 1000)
orbital_phase_full = np.linspace(0, 1, 1000)
Wl_post = np.outer(1 - vsys * 1000 / const.c.value, wl)
flux_model = scisig.fftconvolve(flux_day + flux_night, ref_kernel, "same")

shifted_templates_pre = np.interp(Wl_post, wl, flux_model)

CC = np.dot(total_convolved_spectrum_pre, shifted_templates_pre.T)

fig, ax = plt.subplots(3)
ax[0].pcolormesh(vsys, orbital_phase_pre_eclipse, CC)
ax[0].set_xlabel("System Velocity (km/s)")
ax[0].set_ylabel("Orbital Phase")

CC_shifted = np.empty(CC.shape)
for i, vel in enumerate(vp_pre_eclipse):
    CC_shifted[i] = np.interp(vsys + vel, vsys, CC[i])
cc_sum = np.sum(CC_shifted, axis=0)

ax[1].pcolormesh(vsys, orbital_phase_pre_eclipse, CC_shifted)
ax[1].set_xlabel("System Velocity (km/s)")
ax[1].set_ylabel("Orbital Phase")

ax[2].plot(vsys, cc_sum)
ax[2].set_xlabel("System Velocity (km/s)")
ax[2].set_ylabel("Cross Correlation Sum")

K = np.linspace(0, 2 * Kp, 1000)


def Kp_vsys_plotter(K, vsys, op, CC):
    K_vsys_map = np.empty((K.size, vsys.size))
    CC_array = np.empty(CC.shape)
    for i, kp in enumerate(K):
        vp = kp * np.sin(2 * np.pi * op)
        for j, vel in enumerate(vp):
            CC_array[j] = np.interp(vsys + vel, vsys, CC[j])
        K_vsys_map[i] = np.sum(CC_array, axis=0)
    return K_vsys_map

start_time = time.time()
K_vsys_map_pre_eclipse = Kp_vsys_plotter(K, vsys, orbital_phase_pre_eclipse, CC)
print("Kplotter --- %s seconds ---" % (time.time() - start_time))


start_time = time.time()
K_vsys_map_pre_eclipse = Kplotter(K, vsys, orbital_phase_pre_eclipse, CC)
print("Kplotter import --- %s seconds ---" % (time.time() - start_time))

vp_post_eclipse = Kp * np.sin(orbital_phase_post_eclipse * 2 * np.pi)
W_post = np.outer(1 - vp_post_eclipse * 1000 / const.c.value, wl)
spectrum_post_eclipse_day = np.interp(W_post, wl, flux_day)
spectrum_post_eclipse_night = np.interp(W_post, wl, flux_night)

convolved_spectrum_post_eclipse_day = np.zeros_like(spectrum_post_eclipse_day)
convolved_spectrum_post_eclipse_night = np.zeros_like(spectrum_post_eclipse_night)

for i in range(n_exposure):
    convolved_spectrum_post_eclipse_day[i] = scisig.fftconvolve(
        spectrum_post_eclipse_day[i],
        time_dependent_broadening_kernels_post_eclipse[i],
        "same",
    )

for i in range(n_exposure):
    convolved_spectrum_post_eclipse_night[i] = scisig.fftconvolve(
        spectrum_post_eclipse_night[i],
        anti_kernels_post_eclipse[i],
        "same",
    )

total_convolved_spectrum_post = convolved_spectrum_post_eclipse_day + convolved_spectrum_post_eclipse_night
Wl_post = np.outer(1 - vsys * 1000 / const.c.value, wl)
post_model = np.interp(Wl_post, wl, flux_model)

CC_post = np.dot(total_convolved_spectrum_post, post_model.T)
K_vsys_map_post_eclipse = Kp_vsys_plotter(K, vsys, orbital_phase_post_eclipse, CC_post)

combined_Kp_plot = K_vsys_map_post_eclipse + K_vsys_map_pre_eclipse
max_lines_pre = np.where(K_vsys_map_pre_eclipse == np.max(K_vsys_map_pre_eclipse))
max_lines_post = np.where(K_vsys_map_post_eclipse == np.max(K_vsys_map_post_eclipse))
max_lines = np.where(combined_Kp_plot == np.max(combined_Kp_plot))

Kp_from_plot = K[max_lines[0][0]]
Kp_from_plot_pre = K[max_lines_pre[0][0]]
Kp_from_plot_post = K[max_lines_post[0][0]]

fig, ax = plt.subplots()
ax.pcolormesh(vsys, K, combined_Kp_plot)
# ax.imshow(combined_Kp_plot, aspect="auto")
ax.set_ylabel(r"$K_p$")
ax.set_xlabel(r"$v_{\text{sys}}$")
ax.axhline(Kp, ls="--", color="red", lw=0.5)
ax.axvline(0, ls="--", color="red", lw=0.5)
ax.axhline(Kp_from_plot, ls="--", color="blue", lw=0.5)
ax.axvline(vsys[max_lines[1][0]], ls="--", color="blue", lw=0.5)
plt.legend(
    edgecolor="k",
    facecolor="w",
    framealpha=1,
    fancybox=True,
)
plt.show()
