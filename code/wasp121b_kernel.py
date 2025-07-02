import numpy as np
import matplotlib.pyplot as plt
import os
from astropy import constants as const
from petitRADTRANS.plotlib import plot_radtrans_opacities
from petitRADTRANS import physical_constants as cst, planet
from petitRADTRANS.config import petitradtrans_config_parser
from petitRADTRANS.physics import temperature_profile_function_guillot_global
import scipy.signal as scisig
from petitRADTRANS.spectral_model import SpectralModel
from petitRADTRANS.math import resolving_space
from petitRADTRANS.planet import Planet

home_path = os.environ["HOME"]

local_path = home_path + "/exoplanet_atmospheres/code"

wasp121b_spectrum = np.load(
    os.path.join(local_path, "wasp121b.npz")
)  # relative flux (normalised) and wavelength in microns


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
# for index, i in enumerate(time_dependent_broadening_kernels):
#     ax[row][column].plot(x, i, label=f"{index + 1}. {orbital_phase[index]:.4f}")
#     column += 1
#     column = column % n_columns
#     if column == 0:
#         row += 1
#         row = row % n_rows
# fig.legend()
#

wl = wasp121b_spectrum["wl"]
flux = wasp121b_spectrum["flux"]

index_start = np.where(np.isclose(wl, 2.09))[0][0]
index_end = np.where(np.isclose(wl, 2.1))[0][0]
wl = wl[index_start:index_end]
flux = flux[index_start:index_end]
flux -= np.mean(flux)


vp_pre_eclipse = Kp * np.sin(orbital_phase_pre_eclipse * 2 * np.pi)
W_pre = np.outer(1 - vp_pre_eclipse * 1000 / const.c.value, wl)
spectrum_pre_eclipse = np.interp(W_pre, wl, flux)


convolved_spectrum_pre_eclipse = np.zeros_like(spectrum_pre_eclipse)

for i in range(n_exposure):
    convolved_spectrum_pre_eclipse[i] = scisig.fftconvolve(
        spectrum_pre_eclipse[i],
        time_dependent_broadening_kernels_pre_eclipse[i],
        "same",
    )


fig, ax = plt.subplots(2, sharex="all", sharey="all")
ax[0].pcolormesh(wl, orbital_phase_pre_eclipse, convolved_spectrum_pre_eclipse)
ax[1].pcolormesh(wl, orbital_phase_pre_eclipse, spectrum_pre_eclipse)
fig.supxlabel(r"Wavelengths ($\mu$m)")
fig.supylabel(r"Orbital Phase")

vsys = np.linspace(-200, 200, 1000)
orbital_phase_full = np.linspace(0, 1, 1000)
Wl_post = np.outer(1 - vsys * 1000 / const.c.value, wl)
flux_model = scisig.fftconvolve(flux, ref_kernel, "same")

shifted_templates_pre = np.interp(Wl_post, wl, flux_model)

CC = np.dot(convolved_spectrum_pre_eclipse, shifted_templates_pre.T)

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


K_vsys_map_pre_eclipse = Kp_vsys_plotter(K, vsys, orbital_phase_pre_eclipse, CC)

vp_post_eclipse = Kp * np.sin(orbital_phase_post_eclipse * 2 * np.pi)
W_post = np.outer(1 - vp_post_eclipse * 1000 / const.c.value, wl)
spectrum_post_eclipse = np.interp(W_post, wl, flux)

convolved_spectrum_post_eclipse = np.zeros_like(spectrum_post_eclipse)

for i in range(n_exposure):
    convolved_spectrum_post_eclipse[i] = scisig.fftconvolve(
        spectrum_post_eclipse[i],
        time_dependent_broadening_kernels_post_eclipse[i],
        "same",
    )

Wl_post = np.outer(1 - vsys * 1000 / const.c.value, wl)
post_model = np.interp(Wl_post, wl, flux_model)

CC_post = np.dot(convolved_spectrum_post_eclipse, post_model.T)
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
ax.axhline(Kp_from_plot, ls="--", color="orange", lw=0.5)
ax.axvline(vsys[max_lines[1][0]], ls="--", color="orange", lw=0.5)
# ax.axhline(max_lines[1][0], ls="--", color="red", lw=0.5)
# ax.axvline(max_lines[0][0], ls="--", color="red", lw=0.5)
print(Kp - Kp_from_plot)


CC_no_convolve_pre = np.dot(spectrum_pre_eclipse, shifted_templates_pre.T)
CC_no_convolve_post = np.dot(spectrum_post_eclipse, post_model.T)

Kp_vsys_no_convolve_pre = Kp_vsys_plotter(
    K, vsys, orbital_phase_pre_eclipse, CC_no_convolve_pre
)
Kp_vsys_no_convolve_post = Kp_vsys_plotter(
    K, vsys, orbital_phase_post_eclipse, CC_no_convolve_post
)

Kp_from_plot_no_convolve = Kp_vsys_no_convolve_post + Kp_vsys_no_convolve_pre

print(
    Kp - K[np.where(Kp_from_plot_no_convolve == np.max(Kp_from_plot_no_convolve))[0][0]]
)
print(
    Kp, K[np.where(Kp_from_plot_no_convolve == np.max(Kp_from_plot_no_convolve))[0][0]]
)

fig, ax = plt.subplots()
ax.pcolormesh(vsys, K, Kp_from_plot_no_convolve)
ax.set_ylabel(r"$K_p$")
ax.set_xlabel(r"$v_{\text{sys}}$")
ax.axhline(Kp, ls="--", color="red", lw=0.5)
ax.axvline(0, ls="--", color="red", lw=0.5)
plt.show()
