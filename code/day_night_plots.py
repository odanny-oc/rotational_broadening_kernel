import numpy as np
import matplotlib.pyplot as plt
import os
from functions import Time_Dependent_Spectrum
from functions import Cross_Correlator
from functions import Kp_vsys_Plotter
from functions import Kp_vsys_Map_from_Flux
from functions import maxIndex
from astropy import constants as const
from petitRADTRANS import physical_constants as cst, planet
from petitRADTRANS.config import petitradtrans_config_parser
from petitRADTRANS.physics import temperature_profile_function_guillot_global
import scipy.signal as scisig
from petitRADTRANS.spectral_model import SpectralModel
from petitRADTRANS.math import resolving_space
from petitRADTRANS.planet import Planet
from functools import lru_cache
import time

home_path = os.environ["HOME"]

local_path = home_path + "/exoplanet_atmospheres/code"

day_night_atmosphere = np.load(
    os.path.join(local_path, "day_night_atmosphere.npz")
)  # relative flux (normalised) and wavelength in microns

wasp121_post_data = np.load(
    local_path + "/crires_posteclipse_WASP121_2021-12-15_processed.npz"
)
wavelength_grid = wasp121_post_data["W"][-1] * 1e-4  # resolution ~ 300000


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


def Broadening_Kernel_OP(x, op):
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
    veq_local = veq

    for i, op_i in enumerate(op):
        if op_i < 0:
            ref_op = op_i + 1
        else:
            ref_op = op_i

        vel = veq_local * np.cos(2 * np.pi * ref_op)

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
range_vel = 2

orbital_phase_pre_eclipse = np.linspace(0.334, 0.425, n_exposure)  # time/period
orbital_phase_post_eclipse = np.linspace(0.539, 0.626, n_exposure)  # time/period

x = np.linspace(-range_vel, range_vel, points_number) * (points_number // 2) * dv

time_dependent_broadening_kernels_pre_eclipse, ref_kernel = Broadening_Kernel_OP(
    x, orbital_phase_pre_eclipse
)

anti_kernels_pre_eclipse, ref_kernel = Broadening_Kernel_OP(
    x, orbital_phase_pre_eclipse - 0.5
)

time_dependent_broadening_kernels_post_eclipse, ref_kernel = Broadening_Kernel_OP(
    x, orbital_phase_post_eclipse
)
anti_kernels_post_eclipse, ref_kernel = Broadening_Kernel_OP(
    x, orbital_phase_post_eclipse - 0.5
)

# n_columns = 10
# n_rows = n_exposure // n_columns
# row = 0
# column = 0
# fig, ax = plt.subplots(n_rows, n_columns, figsize=(10, 5), sharey="all")
# for i, j in zip(
#     time_dependent_broadening_kernels_pre_eclipse, anti_kernels_pre_eclipse
# ):
#     ax[row][column].plot(x, i)
#     ax[row][column].plot(x, j)
#     column += 1
#     column = column % n_columns
#     if column == 0:
#         row += 1
#         row = row % n_rows

# Define Wavelength and Flux Grids
wl = day_night_atmosphere["wl_day"]

flux_day = day_night_atmosphere["flux_day"]
flux_night = day_night_atmosphere["flux_night"]

flux_day -= np.mean(flux_day)
flux_night -= np.mean(flux_night)

flux_grid_day = np.interp(wavelength_grid, wl, flux_day)
flux_grid_night = np.interp(wavelength_grid, wl, flux_night)

# Calculate Doppler shift over orbital phase
convolved_spectrum_pre_eclipse_day = Time_Dependent_Spectrum(
    wl,
    flux_day,
    orbital_phase_pre_eclipse,
    Kp,
    time_dependent_broadening_kernels_pre_eclipse,
    wl_grid=wavelength_grid,
)

convolved_spectrum_pre_eclipse_night = Time_Dependent_Spectrum(
    wl,
    flux_night,
    orbital_phase_pre_eclipse,
    Kp,
    anti_kernels_pre_eclipse,
    wl_grid=wavelength_grid,
)

total_convolved_spectrum_pre = (
    convolved_spectrum_pre_eclipse_night + convolved_spectrum_pre_eclipse_day
)

vsys = np.linspace(-200, 200, 1001)
K = np.linspace(0, 2 * Kp, 1001)
flux_grid_tot = flux_grid_day + flux_grid_night
flux_model = scisig.fftconvolve(flux_day + flux_night, ref_kernel, "same")
flux_model_grid = scisig.fftconvolve(flux_grid_tot, ref_kernel, "same")

CC_pre = Cross_Correlator(
    wavelength_grid, flux_model_grid, vsys * 1000, total_convolved_spectrum_pre
)

K_vsys_map_pre_eclipse, CC_shifted = Kp_vsys_Plotter(
    K, vsys, orbital_phase_pre_eclipse, CC_pre
)

fig, ax = plt.subplots(3)
ax[0].pcolormesh(vsys, orbital_phase_pre_eclipse, CC_pre)
ax[0].set_xlabel("System Velocity (km/s)")
ax[0].set_ylabel("Orbital Phase")

index = 500
cc_sum = K_vsys_map_pre_eclipse[index]

ax[1].pcolormesh(vsys, orbital_phase_pre_eclipse, CC_shifted[index])
ax[1].set_xlabel("System Velocity (km/s)")
ax[1].set_ylabel("Orbital Phase")

ax[2].plot(vsys, cc_sum)
ax[2].set_xlabel("System Velocity (km/s)")
ax[2].set_ylabel("Cross Correlation Sum")

convolved_spectrum_post_eclipse_day = Time_Dependent_Spectrum(
    wl,
    flux_day,
    orbital_phase_post_eclipse,
    Kp,
    time_dependent_broadening_kernels_post_eclipse,
    wl_grid=wavelength_grid,
)

convolved_spectrum_post_eclipse_night = Time_Dependent_Spectrum(
    wl,
    flux_night,
    orbital_phase_post_eclipse,
    Kp,
    anti_kernels_post_eclipse,
    wl_grid=wavelength_grid,
)

total_convolved_spectrum_post = (
    convolved_spectrum_post_eclipse_day + convolved_spectrum_post_eclipse_night
)

CC_post = Cross_Correlator(
    wavelength_grid, flux_model_grid, vsys * 1000, total_convolved_spectrum_post
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

unconvolved_Kp_pre, K_array = Kp_vsys_Map_from_Flux(
    wl,
    flux_day + flux_night,
    orbital_phase_pre_eclipse,
    vsys,
    Kp,
    wl_grid=wavelength_grid,
    flux_grid=flux_model_grid,
)

unconvolved_Kp_post, K_array = Kp_vsys_Map_from_Flux(
    wl,
    flux_day + flux_night,
    orbital_phase_post_eclipse,
    vsys,
    Kp,
    wl_grid=wavelength_grid,
    flux_grid=flux_model_grid,
)

total_unconvolve_spec = unconvolved_Kp_pre + unconvolved_Kp_post

max_unconv = np.where(total_unconvolve_spec == np.max(total_unconvolve_spec))

fig, ax = plt.subplots(3, sharex="all", sharey="all")
fig.suptitle(r"$K_p$ - $v_{\text{sys}}$ Plots for Different Kernels")
ax[1].set_title("No Kernel")
ax[0].set_title("Day-Night Kernel")
ax[2].set_title("Gaussian Kernel")
fig.supxlabel(r"$v_{\text{sys}}$ (km/s)")
fig.supylabel(r"$K_p$ (km/s)")

ax[0].pcolormesh(vsys, K, combined_Kp_plot)
ax[0].axhline(
    Kp, ls="--", color="red", lw=0.5, label=r"Actual $K_p$ = " + f"{Kp:.2f}km/s"
)
ax[0].axvline(0, ls="--", color="red", lw=0.5)
ax[0].axhline(
    Kp_from_plot,
    ls="--",
    color="blue",
    lw=0.5,
    label=r"Measured $K_p$ = " + f"{Kp_from_plot:.2f}km/s",
)
ax[0].axvline(vsys[max_lines[1][0]], ls="--", color="blue", lw=0.5)
ax[0].legend(
    edgecolor="k",
    facecolor="w",
    framealpha=1,
    fancybox=True,
)
ax[0].annotate(
    r"$\Delta K_p$ = " + f"{( Kp -  Kp_from_plot):.2f}km/s",
    xy=(0.9, 0.85),
    xycoords="axes fraction",
    size=10,
    bbox=dict(fc="w", ec="k", boxstyle="round", linewidth=2),
)

ax[1].pcolormesh(vsys, K_array, total_unconvolve_spec)
ax[1].axhline(
    Kp, ls="--", color="red", lw=0.5, label=r"Actual $K_p$ = " + f"{Kp:.2f}km/s"
)
ax[1].axvline(0, ls="--", color="red", lw=0.5)
ax[1].axhline(
    K_array[max_unconv[0][0]],
    ls="--",
    color="blue",
    lw=0.5,
    label=r"Measured $K_p$ = " + f"{K_array[max_unconv[0][0]]:.2f}km/s",
)
ax[1].axvline(vsys[max_unconv[1][0]], ls="--", color="blue", lw=0.5)
ax[1].legend(
    edgecolor="k",
    facecolor="w",
    framealpha=1,
    fancybox=True,
)
ax[1].annotate(
    r"$\Delta K_p$ = " + f"{( Kp - K_array[max_unconv[0][0]] ):.2f}km/s",
    xy=(0.9, 0.85),
    xycoords="axes fraction",
    size=10,
    bbox=dict(fc="w", ec="k", boxstyle="round", linewidth=2),
)


def gaussian(x, sigma):
    return np.exp(-0.5 * (x / sigma) ** 2)


g = gaussian(x, 2)

gaussian_convolved_pre, _ = Kp_vsys_Map_from_Flux(
    wl,
    flux_day + flux_night,
    orbital_phase_pre_eclipse,
    vsys,
    Kp,
    g,
    wl_grid=wavelength_grid,
    flux_grid=flux_model_grid,
)

gaussian_convolved_post, _ = Kp_vsys_Map_from_Flux(
    wl,
    flux_day + flux_night,
    orbital_phase_post_eclipse,
    vsys,
    Kp,
    g,
    wl_grid=wavelength_grid,
    flux_grid=flux_model_grid,
)

gaussian_convolved_tot = gaussian_convolved_pre + gaussian_convolved_post

ax[2].pcolormesh(vsys, K, gaussian_convolved_tot)
ax[2].axhline(
    Kp, ls="--", color="red", lw=0.5, label=r"Actual $K_p$ = " + f"{Kp:.2f}km/s"
)
ax[2].axvline(0, ls="--", color="red", lw=0.5)
ax[2].axhline(
    K[np.where(gaussian_convolved_tot == np.max(gaussian_convolved_tot))[0][0]],
    ls="--",
    color="blue",
    lw=0.5,
    label=r"Measured $K_p$ = "
    + f"{K[np.where(gaussian_convolved_tot == np.max(gaussian_convolved_tot))[0][0]]:.2f}km/s",
)
ax[2].axvline(
    vsys[np.where(gaussian_convolved_tot == np.max(gaussian_convolved_tot))[1][0]],
    ls="--",
    color="blue",
    lw=0.5,
)
ax[2].legend(
    edgecolor="k",
    facecolor="w",
    framealpha=1,
    fancybox=True,
)
ax[2].annotate(
    r"$\Delta K_p$ = "
    + f"{( Kp -  K[np.where(gaussian_convolved_tot == np.max(gaussian_convolved_tot))[0][0]]):.2f}km/s",
    xy=(0.9, 0.85),
    xycoords="axes fraction",
    size=10,
    bbox=dict(fc="w", ec="k", boxstyle="round", linewidth=2),
)

flux_model_day = scisig.fftconvolve(flux_grid_day, ref_kernel, "same")
flux_model_night = scisig.fftconvolve(flux_grid_night, ref_kernel, "same")

CC_day_pre = Cross_Correlator(
    wl=wavelength_grid,
    flux=flux_grid_day,
    spectrum=total_convolved_spectrum_pre,
    vsys=vsys * 1000,
)

CC_night_pre = Cross_Correlator(
    wl=wavelength_grid,
    flux=flux_grid_night,
    spectrum=total_convolved_spectrum_pre,
    vsys=vsys * 1000,
)

CC_day_post = Cross_Correlator(
    wl=wavelength_grid,
    flux=flux_grid_day,
    spectrum=total_convolved_spectrum_post,
    vsys=vsys * 1000,
)

CC_night_post = Cross_Correlator(
    wl=wavelength_grid,
    flux=flux_grid_night,
    spectrum=total_convolved_spectrum_post,
    vsys=vsys * 1000,
)

Kp_vsys_day_pre, _ = Kp_vsys_Plotter(
    K=K, vsys=vsys, CC=CC_day_pre, op=orbital_phase_pre_eclipse
)

Kp_vsys_day_post, _ = Kp_vsys_Plotter(
    K=K, vsys=vsys, CC=CC_day_post, op=orbital_phase_post_eclipse
)

Kp_vsys_night_pre, _ = Kp_vsys_Plotter(
    K=K, vsys=vsys, CC=CC_night_pre, op=orbital_phase_pre_eclipse
)

Kp_vsys_night_post, _ = Kp_vsys_Plotter(
    K=K, vsys=vsys, CC=CC_night_post, op=orbital_phase_post_eclipse
)
# ax[0].set_title("Flux Night")
# ax[0].plot(wavelength_grid, flux_grid_night)
# ax[1].set_title("Flux Day")
# ax[1].plot(wavelength_grid, flux_grid_day)

fig, ax = plt.subplots(2)

ax[0].pcolormesh(vsys, K, Kp_vsys_night_pre)
ax[1].pcolormesh(vsys, K, Kp_vsys_night_post)
fig.suptitle(r"$K_p$ - $v_{\text{sys}}$ Night")
fig.supxlabel(r"$v_{\text{sys}}$")
ax[0].set_title("Pre-Eclipse")
ax[1].set_title("Post-Eclipse")

ax[0].axhline(
    K[maxIndex(Kp_vsys_night_pre)[0]],
    ls="--",
    color="red",
    lw=0.5,
    label=r"Measured $K_p$ = " + f"{K[maxIndex(Kp_vsys_night_pre)[0]]:.2f}km/s",
)

ax[0].axvline(
    vsys[maxIndex(Kp_vsys_night_pre)[1]],
    ls="--",
    color="red",
    lw=0.5,
    label=r"Measured $v_{\text{sys}}$ = " + f"{vsys[maxIndex(Kp_vsys_night_pre)[1]]:.2f}km/s",
)

ax[1].axhline(
    K[maxIndex(Kp_vsys_night_post)[0]],
    ls="--",
    color="red",
    lw=0.5,
    label=r"Measured $K_p$ = " + f"{K[maxIndex(Kp_vsys_night_post)[0]]:.2f}km/s",
)

ax[1].axvline(
    vsys[maxIndex(Kp_vsys_night_post)[1]],
    ls="--",
    color="red",
    lw=0.5,
    label=r"Measured $v_{\text{sys}}$ = " + f"{vsys[maxIndex(Kp_vsys_night_post)[1]]:.2f}km/s",
)

ax[0].legend()
ax[1].legend()

fig, ax = plt.subplots(2)
fig.suptitle("Pre-Eclipse")
ax[0].pcolormesh(vsys, orbital_phase_pre_eclipse, CC_day_pre)
ax[1].pcolormesh(vsys, orbital_phase_pre_eclipse, CC_night_pre)
ax[0].set_title("Cross Correlation Day Side")
ax[1].set_title("Cross Correlation Night Side")
fig.supxlabel(r"$v_{\text{sys}}$")
fig.supylabel(r"Orbital Phase")

fig, ax = plt.subplots(2)
fig.suptitle("Post-Eclipse")
ax[0].pcolormesh(vsys, orbital_phase_post_eclipse, CC_day_post)
ax[1].pcolormesh(vsys, orbital_phase_post_eclipse, CC_night_post)
ax[0].set_title("Cross Correlation Day Side")
ax[1].set_title("Cross Correlation Night Side")
fig.supxlabel(r"$v_{\text{sys}}$")
fig.supylabel(r"Orbital Phase")

Kp_vsys_day = Kp_vsys_day_pre + Kp_vsys_day_post
Kp_vsys_night = Kp_vsys_night_pre + Kp_vsys_night_post

fig, ax = plt.subplots(2, sharex="all", sharey="all")
fig.suptitle(r"$K_p$ - $v_{\text{sys}}$ Night")
fig.supxlabel(r"$v_{\text{sys}}$")
fig.supylabel(r"$K_p$")
ax[0].pcolormesh(vsys, K, Kp_vsys_day)
ax[0].axhline(
    Kp, ls="--", color="blue", lw=0.5, label=r"Actual $K_p$ = " + f"{Kp:.2f}km/s"
)
ax[0].axvline(
    0, ls="--", color="blue", lw=0.5, label=r"Actual $v_{\text{sys}}$ = " + "0"
)
ax[0].axhline(
    K[maxIndex(Kp_vsys_day)[0]],
    ls="--",
    color="red",
    lw=0.5,
    label=r"Measured $K_p$ = " + f"{K[maxIndex(Kp_vsys_day)[0]]:.2f}km/s",
)
ax[0].axvline(
    vsys[maxIndex(Kp_vsys_day)[1]],
    ls="--",
    color="red",
    lw=0.5,
    label=r"Measured $v_{\text{sys}}$ = " + f"{vsys[maxIndex(Kp_vsys_day)[1]]:.2f}km/s",
)
ax[0].set_title("Day")
ax[1].set_title("Night")
ax[1].pcolormesh(vsys, K, Kp_vsys_night)

ax[1].axhline(
    Kp,
    ls="--",
    color="blue",
    lw=0.5,
)
ax[1].axvline(
    0,
    ls="--",
    color="blue",
    lw=0.5,
)
ax[1].axhline(
    K[maxIndex(Kp_vsys_night)[0]],
    ls="--",
    color="red",
    lw=0.5,
    label=r"Measured $K_p$ = " + f"{K[maxIndex(Kp_vsys_night)[0]]:.2f}km/s",
)
ax[1].axvline(
    vsys[maxIndex(Kp_vsys_night)[1]],
    ls="--",
    color="red",
    lw=0.5,
    label=r"Measured $v_{\text{sys}}$ = "
    + f"{vsys[maxIndex(Kp_vsys_night)[1]]:.2f}km/s",
)
#print(f"{Kp - K[maxIndex(Kp_vsys_night)[0]]}")
ax[0].legend()
ax[1].legend()
plt.show()
