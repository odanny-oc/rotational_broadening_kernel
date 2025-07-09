import numpy as np
import matplotlib.pyplot as plt
import os
from functions import Time_Dependent_Spectrum
from functions import Cross_Correlator
from functions import Kp_vsys_Plotter
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


resolution = 400000
dv = const.c.value * 1e-3 / resolution
points_number = 51
n_exposure = 300
kernel_res = n_exposure // 2
range_vel = 1.0

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
flux -= np.mean(flux)

fitted_flux = np.interp(wavelength_grid, wl, flux)

# fig, ax = plt.subplots(2)
# ax[0].plot(wavelength_grid, fitted_flux)
# ax[1].plot(wl, flux)

vp_pre_eclipse = Kp * np.sin(orbital_phase_pre_eclipse * 2 * np.pi)
W_pre = np.outer(1 - vp_pre_eclipse * 1000 / const.c.value, wavelength_grid)
spectrum_pre_eclipse = np.interp(W_pre, wl, flux)

convolved_spectrum_pre_eclipse = Time_Dependent_Spectrum(
    wl,
    flux,
    orbital_phase_pre_eclipse,
    Kp,
    time_dependent_broadening_kernels_pre_eclipse,
    wavelength_grid,
)

fig, ax = plt.subplots()
full_kernel_flux = scisig.fftconvolve(
    fitted_flux, broadening_kernel_orbital_phase(x, 0)[1], "same"
)
shifted_flux = scisig.fftconvolve(
    fitted_flux, broadening_kernel_orbital_phase(x, 0.25)[0], "same"
)
ax.plot(wavelength_grid, full_kernel_flux)
ax.plot(wavelength_grid, shifted_flux)

fig, ax = plt.subplots(2, sharex="all", sharey="all")
ax[0].pcolormesh(
    wavelength_grid, orbital_phase_pre_eclipse, convolved_spectrum_pre_eclipse
)
ax[1].pcolormesh(wavelength_grid, orbital_phase_pre_eclipse, spectrum_pre_eclipse)
fig.supxlabel(r"Wavelengths ($\mu$m)")
fig.supylabel(r"Orbital Phase")

vsys = np.linspace(-200, 200, 1001)

CC_pre = Cross_Correlator(
    wavelength_grid, fitted_flux, vsys * 1000, convolved_spectrum_pre_eclipse
)

K = np.linspace(0, 2 * Kp, 1001)

K_vsys_map_pre_eclipse, CC_shifted = Kp_vsys_Plotter(
    K, vsys, orbital_phase_pre_eclipse, CC_pre
)

fig, ax = plt.subplots(3)
ax[0].pcolormesh(vsys, orbital_phase_pre_eclipse, CC_pre)
ax[0].set_xlabel("System Velocity (km/s)")
ax[0].set_ylabel("Orbital Phase")


index = 500
CC_shifted = CC_shifted[index]
cc_sum = K_vsys_map_pre_eclipse[index]

ax[1].pcolormesh(vsys, orbital_phase_pre_eclipse, CC_shifted)
ax[1].set_xlabel("System Velocity (km/s)")
ax[1].set_ylabel("Orbital Phase")

ax[2].plot(vsys, cc_sum)
ax[2].set_xlabel("System Velocity (km/s)")
ax[2].set_ylabel("Cross Correlation Sum")


K_vsys_map_pre_eclipse, _ = Kp_vsys_Plotter(K, vsys, orbital_phase_pre_eclipse, CC_pre)

convolved_spectrum_post_eclipse = Time_Dependent_Spectrum(
    wl,
    flux,
    orbital_phase_post_eclipse,
    Kp,
    time_dependent_broadening_kernels_post_eclipse,
    wavelength_grid,
)

CC_post = Cross_Correlator(
    wavelength_grid, fitted_flux, vsys * 1000, convolved_spectrum_post_eclipse
)

K_vsys_map_post_eclipse, _ = Kp_vsys_Plotter(
    K, vsys, orbital_phase_post_eclipse, CC_post
)

combined_Kp_plot = K_vsys_map_post_eclipse + K_vsys_map_pre_eclipse
max_lines_pre = np.where(K_vsys_map_pre_eclipse == np.max(K_vsys_map_pre_eclipse))
max_lines_post = np.where(K_vsys_map_post_eclipse == np.max(K_vsys_map_post_eclipse))
max_lines = np.where(combined_Kp_plot == np.max(combined_Kp_plot))

Kp_from_plot = K[max_lines[0][0]]
Kp_from_plot_pre = K[max_lines_pre[0][0]]
Kp_from_plot_post = K[max_lines_post[0][0]]

fig, ax = plt.subplots(3, sharex="all", sharey="all")
ax[0].pcolormesh(vsys, K, combined_Kp_plot)
# ax[0].imshow(combined_Kp_plot, aspect="auto")
fig.supylabel(r"$K_p$")
fig.supxlabel(r"$v_{\text{sys}}$")
ax[0].axhline(Kp, ls="--", color="red", lw=0.5, label=r"Actual $K_p$ = " + f"{Kp:.2f}")
ax[0].axvline(0, ls="--", color="red", lw=0.5)
ax[0].axhline(
    Kp_from_plot,
    ls="--",
    color="orange",
    lw=0.5,
    label=r"Measured $K_p$ =" + f"{Kp_from_plot:.2f}",
)
ax[0].axvline(vsys[max_lines[1][0]], ls="--", color="orange", lw=0.5)
ax[0].legend()

unconvolved_spec_pre, _ = Kp_vsys_Map_from_Flux(
    wl, flux, orbital_phase_pre_eclipse, vsys, Kp
)

unconvolved_spec_post, _ = Kp_vsys_Map_from_Flux(
    wl, flux, orbital_phase_post_eclipse, vsys, Kp
)

total_unconvolved_spec = unconvolved_spec_pre + unconvolved_spec_post

ax[1].pcolormesh(vsys, K, total_unconvolved_spec)
# ax[1].imshow(combined_Kp_plot, aspect="auto")
ax[1].axhline(Kp, ls="--", color="red", lw=0.5)
ax[1].axvline(0, ls="--", color="red", lw=0.5)
ax[1].axhline(
    K[np.where(total_unconvolved_spec == np.max(total_unconvolved_spec))[0][0]],
    ls="--",
    color="orange",
    lw=0.5,
    label=r"Measured $K_p$ = "
    + f"{
        K[np.where(total_unconvolved_spec == np.max(total_unconvolved_spec))[0][0]]:.2f}",
)
ax[1].legend()


def gaussian(x, sigma):
    return np.exp(-0.5 * (x / sigma) ** 2)


g = gaussian(x, 2)

gaussian_spec_pre, _ = Kp_vsys_Map_from_Flux(
    wl, flux, orbital_phase_pre_eclipse, vsys, Kp, g
)

gaussian_spec_post, _ = Kp_vsys_Map_from_Flux(
    wl, flux, orbital_phase_post_eclipse, vsys, Kp, g
)

total_gaussian_spec = gaussian_spec_pre + gaussian_spec_post

ax[2].pcolormesh(vsys, K, total_gaussian_spec)
# ax[2].imshow(combined_Kp_plot, aspect="auto")
ax[2].axhline(Kp, ls="--", color="red", lw=0.5)
ax[2].axvline(0, ls="--", color="red", lw=0.5)
ax[2].axhline(
    K[np.where(total_gaussian_spec == np.max(total_gaussian_spec))[0][0]],
    ls="--",
    color="orange",
    lw=0.5,
    label=r"Measured $K_p$ = "
    + f"{
        K[np.where(total_gaussian_spec == np.max(total_gaussian_spec))[0][0]]:.2f}",
)
ax[2].legend()

plt.show()
