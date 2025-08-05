import numpy as np
import matplotlib.pyplot as plt
import os
from functions import Time_Dependent_Spectrum
from functions import Cross_Correlator
from functions import Kp_vsys_Plotter
from functions import Kp_vsys_Map_from_Flux
from functions import maxIndex
from functions import vel_array
from astropy import constants as const
import scipy.signal as scisig
import time

home_path = os.environ["HOME"]

local_path = home_path + "/exoplanet_atmospheres/code"

day_night_atmosphere = np.load(
    os.path.join(local_path, "day_night_spectrum.npz")
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
# veq = 350

# x kernel range and op orbital phase
def broadening_kernel_op(x, op):
    # Allows float or int values of op to be passed
    if not isinstance(op, np.ndarray):
        op = np.array([op])

    veq_local = veq

    kernel_array = np.zeros(shape=(op.shape[0], x.shape[0]))
    kernel_range_array = []
    # Sets valid range given veq
    ref_range = np.array([i for i in x if abs(i) <= veq_local])

    ref_kernel = np.sqrt(1 - (ref_range / veq_local) ** 2)
    # Pads evenly the rest of the range with zeros
    ref_padding = abs(x.shape[0] - ref_range.shape[0]) // 2
    ref_kernel = np.pad(ref_kernel, ref_padding, "constant")

    normaliser = np.sum(ref_kernel)
    ref_kernel /= normaliser
    full_kernel = ref_kernel.copy()
    # Takes the half kernel (right side)
    ref_kernel[: x.shape[0] // 2] = np.zeros(x.shape[0] // 2)
    period_scaler = period / (4 * (tdur - tfull) / 2)

    for i, op_i in enumerate(op):
        # Fit all op values between 0 and 1
        if op_i < 0:
            ref_op = op_i + 1
        else:
            ref_op = op_i

        vel = veq_local * np.cos(2 * np.pi * ref_op)

        # Consider 0 case
        if vel == 0:
            kernel = ref_kernel
            kernel_range = ref_range
            # Consider opposite side of orbit
            if ref_op < 0.5:
                kernel = np.flip(kernel)

        else:
            # Check if planet is about to enter secondary eclipse
            if abs(ref_op - 0.5) < tdur / (2 * period) or abs(ref_op + 0.5) < tdur / (
                2 * period
            ):
                # Zero out signal if planet is eclipsed
                if abs(ref_op - 0.5) < tfull / (2 * period) or abs(
                    ref_op + 0.5
                ) < tfull / (2 * period):
                    kernel = np.zeros(x.shape[0])
                    kernel_range = kernel
                else:
                    # Gradually reduce signal otherwise (from left or from right depending on op)
                    if ref_op > 0:
                        vel = veq * np.cos(
                            (2 * np.pi * (ref_op - 0.5)) * (period_scaler)
                        )
                    else:
                        vel = veq * np.cos(
                            (2 * np.pi * (ref_op + 0.5)) * (period_scaler)
                        )
                    # Create kernel
                    kernel_range = np.array([i for i in x if abs(i) <= abs(vel)])
                    kernel = np.sqrt(1 - (kernel_range / abs(vel)) ** 2) / normaliser
                    padding = abs(x.shape[0] - kernel_range.shape[0]) // 2
                    kernel = np.pad(kernel, padding, "constant")
                    if vel < 0:
                        kernel[x.shape[0] // 2 :] = np.zeros(x.shape[0] // 2 + 1)
                        kernel += ref_kernel
                    if vel > 0:
                        kernel[: x.shape[0] // 2] = np.zeros(x.shape[0] // 2)
                        kernel = ref_kernel - kernel
                    if ref_op > 0.5:
                        kernel = np.flip(kernel)
            # Else not about to be eclipsed takes original value of vel
            else:
                kernel_range = np.array([i for i in x if abs(i) <= abs(vel)])
                kernel = np.sqrt(1 - (kernel_range / abs(vel)) ** 2) / normaliser
                padding = abs(x.shape[0] - kernel_range.shape[0]) // 2
                kernel = np.pad(kernel, padding, "constant")
                if vel < 0:
                    kernel[x.shape[0] // 2 :] = np.zeros(x.shape[0] // 2 + 1)
                    kernel += ref_kernel
                if vel > 0:
                    kernel[: x.shape[0] // 2] = np.zeros(x.shape[0] // 2)
                    kernel = ref_kernel - kernel
                if ref_op < 0.5:
                    kernel = np.flip(kernel)
        # Saves kernel to kernel array
        kernel_array[i] = kernel
        kernel_range_array.append(kernel_range)

    if kernel_array.shape[0] == 1:
        return kernel_array[0], kernel_range_array
    else:
        return kernel_array, kernel_range_array


resolution = 400000
n_exposure = 300

# orbital_phase_pre_eclipse = np.linspace(0.33, 0.43, n_exposure)  # time/period
# orbital_phase_post_eclipse = np.linspace(0.55, 0.66, n_exposure)  # time/period

orbital_phase_pre_eclipse = np.linspace(-0.12, -0.04, n_exposure)  # time/period
orbital_phase_post_eclipse = np.linspace(0.04, 0.12, n_exposure)  # time/period

x = vel_array(30, resolution)

time_dependent_broadening_kernels_pre_eclipse, kernel_range_pre = broadening_kernel_op(
    x, orbital_phase_pre_eclipse
)

anti_kernels_pre_eclipse, anti_kernel_range_pre = broadening_kernel_op(
    x, orbital_phase_pre_eclipse - 0.5
)

time_dependent_broadening_kernels_post_eclipse, kernel_range_post = broadening_kernel_op(
    x, orbital_phase_post_eclipse
)
anti_kernels_post_eclipse, anti_kernel_range_post  = broadening_kernel_op(
    x, orbital_phase_post_eclipse - 0.5
)


def cc_weighted_average(kernels, kernel_range, x):
    print(kernels.shape[0])
    predicted_trace = np.empty(kernels.shape[0])
    for i in range(kernels.shape[0]):
        weights =kernels[i][ np.where(x == kernel_range[i][0])[0][0]: np.where(x == kernel_range[i][-1])[0][0] + 1]
        weighted_average = np.average(kernel_range[i], weights=weights)
        predicted_trace[i]= weighted_average
    return predicted_trace


cc_pre_day_fit = cc_weighted_average(time_dependent_broadening_kernels_pre_eclipse, kernel_range_pre, x)

# Define Wavelength and Flux Grids
wl = day_night_atmosphere["wl_day"]

flux_day = day_night_atmosphere["flux_day"]
flux_night = day_night_atmosphere["flux_night"]

flux_day -= np.mean(flux_day)
flux_night -= np.mean(flux_night)

flux_tot = flux_day + flux_night


# Calculate Doppler shift over orbital phase
convolved_spectrum_pre_eclipse_day = Time_Dependent_Spectrum(
    wl,
    flux_day,
    orbital_phase_pre_eclipse,
    Kp,
    time_dependent_broadening_kernels_pre_eclipse,
)

convolved_spectrum_pre_eclipse_night = Time_Dependent_Spectrum(
    wl,
    flux_night,
    orbital_phase_pre_eclipse,
    Kp,
    anti_kernels_pre_eclipse,
)

total_convolved_spectrum_pre = (
    convolved_spectrum_pre_eclipse_night + convolved_spectrum_pre_eclipse_day
)

vsys = np.linspace(-200, 200, 1001)
vsys_kp = np.linspace(-30, 30, 1001)
K = np.linspace(Kp - 85, Kp + 85, 1001)


CC_pre = Cross_Correlator(wl, flux_tot, vsys * 1000, total_convolved_spectrum_pre)

K_vsys_map_pre_eclipse, CC_shifted = Kp_vsys_Plotter(
    K,
    vsys,
    orbital_phase_pre_eclipse,
    CC_pre,
    vsys_kp=vsys_kp,
)

convolved_spectrum_post_eclipse_day = Time_Dependent_Spectrum(
    wl,
    flux_day,
    orbital_phase_post_eclipse,
    Kp,
    time_dependent_broadening_kernels_post_eclipse,
)

convolved_spectrum_post_eclipse_night = Time_Dependent_Spectrum(
    wl,
    flux_night,
    orbital_phase_post_eclipse,
    Kp,
    anti_kernels_post_eclipse,
)

total_convolved_spectrum_post = (
    convolved_spectrum_post_eclipse_day + convolved_spectrum_post_eclipse_night
)

CC_post = Cross_Correlator(wl, flux_tot, vsys * 1000, total_convolved_spectrum_post)

K_vsys_map_post_eclipse, _ = Kp_vsys_Plotter(
    K, vsys, orbital_phase_post_eclipse, CC_post, vsys_kp=vsys_kp
)

combined_Kp_plot = K_vsys_map_post_eclipse + K_vsys_map_pre_eclipse

max_lines_pre = maxIndex(K_vsys_map_pre_eclipse)
max_lines_post = maxIndex(K_vsys_map_post_eclipse)
max_lines = maxIndex(combined_Kp_plot)

Kp_from_plot = K[max_lines[0]]
Kp_from_plot_pre = K[max_lines_pre[0]]
Kp_from_plot_post = K[max_lines_post[0]]

unconvolved_Kp_pre, K_array = Kp_vsys_Map_from_Flux(
    wl, flux_tot, orbital_phase_pre_eclipse, vsys, Kp, vsys_kp=vsys_kp, K=K
)


unconvolved_Kp_post, K_array = Kp_vsys_Map_from_Flux(
    wl, flux_tot, orbital_phase_post_eclipse, vsys, Kp, vsys_kp=vsys_kp, K=K
)

total_unconvolve_spec = unconvolved_Kp_pre + unconvolved_Kp_post

max_unconv = maxIndex(total_unconvolve_spec)

fig, ax = plt.subplots(3, sharex="all", sharey="all")
fig.suptitle(r"$K_p$ - $v_{\text{sys}}$ Plots for Different Kernels")
ax[1].set_title("No Kernel")
ax[0].set_title("Day-Night Kernel")
ax[2].set_title("Gaussian Kernel")
fig.supxlabel(r"$v_{\text{sys}}$ (km/s)")
fig.supylabel(r"$K_p$ (km/s)")

ax[0].pcolormesh(vsys_kp, K, combined_Kp_plot)
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
ax[0].axvline(
    vsys_kp[max_lines[1]],
    ls="--",
    color="blue",
    lw=0.5,
    label=f"Measured vsys = {vsys_kp[max_lines[1]]:.2f}",
)
ax[0].legend(
    edgecolor="k",
    facecolor="w",
    framealpha=1,
    fancybox=True,
    loc="upper left",
)
ax[0].annotate(
    r"$\Delta K_p$ = " + f"{( Kp -  Kp_from_plot):.2f}km/s",
    xy=(0.9, 0.85),
    xycoords="axes fraction",
    size=10,
    bbox=dict(fc="w", ec="k", boxstyle="round", linewidth=2),
)

ax[1].pcolormesh(vsys_kp, K_array, total_unconvolve_spec)
ax[1].axhline(
    Kp, ls="--", color="red", lw=0.5, label=r"Actual $K_p$ = " + f"{Kp:.2f}km/s"
)
ax[1].axvline(0, ls="--", color="red", lw=0.5)
ax[1].axhline(
    K_array[max_unconv[0]],
    ls="--",
    color="blue",
    lw=0.5,
    label=r"Measured $K_p$ = " + f"{K_array[max_unconv[0]]:.2f}km/s",
)
ax[1].axvline(
    vsys_kp[max_unconv[1]],
    ls="--",
    color="blue",
    lw=0.5,
    label=f"Measured vsys = {vsys_kp[max_unconv[1]]:.2f}",
)

ax[1].legend(
    loc="upper left",
    edgecolor="k",
    facecolor="w",
    framealpha=1,
    fancybox=True,
)

ax[1].annotate(
    r"$\Delta K_p$ = " + f"{( Kp - K_array[max_unconv[0]] ):.2f}km/s",
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
    vsys_kp=vsys_kp,
    K=K,
)

gaussian_convolved_post, _ = Kp_vsys_Map_from_Flux(
    wl,
    flux_day + flux_night,
    orbital_phase_post_eclipse,
    vsys,
    Kp,
    g,
    vsys_kp=vsys_kp,
    K=K,
)

gaussian_convolved_tot = gaussian_convolved_pre + gaussian_convolved_post

ax[2].pcolormesh(vsys_kp, K, gaussian_convolved_tot)
ax[2].axhline(
    Kp, ls="--", color="red", lw=0.5, label=r"Actual $K_p$ = " + f"{Kp:.2f}km/s"
)
ax[2].axvline(0, ls="--", color="red", lw=0.5)
ax[2].axhline(
    K[maxIndex(gaussian_convolved_tot)[0]],
    ls="--",
    color="blue",
    lw=0.5,
    label=r"Measured $K_p$ = "
    + f"{K[np.where(gaussian_convolved_tot == np.max(gaussian_convolved_tot))[0][0]]:.2f}km/s",
)
ax[2].axvline(
    vsys_kp[maxIndex(gaussian_convolved_tot)[1]],
    ls="--",
    color="blue",
    lw=0.5,
    label=f"Measured vsys = {vsys_kp[maxIndex(gaussian_convolved_tot)[1]]:.2f}",
)
ax[2].legend(
    loc="upper left",
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

CC_day_pre = Cross_Correlator(
    wl=wl,
    flux=flux_day,
    spectrum=total_convolved_spectrum_pre,
    vsys=vsys * 1000,
)

CC_night_pre = Cross_Correlator(
    wl=wl,
    flux=flux_night,
    spectrum=total_convolved_spectrum_pre,
    vsys=vsys * 1000,
)

CC_day_post = Cross_Correlator(
    wl=wl,
    flux=flux_day,
    spectrum=total_convolved_spectrum_post,
    vsys=vsys * 1000,
)

CC_night_post = Cross_Correlator(
    wl=wl,
    flux=flux_night,
    spectrum=total_convolved_spectrum_post,
    vsys=vsys * 1000,
)

Kp_vsys_day_pre, _ = Kp_vsys_Plotter(
    K=K, vsys=vsys, CC=CC_day_pre, op=orbital_phase_pre_eclipse, vsys_kp=vsys_kp
)

Kp_vsys_day_post, _ = Kp_vsys_Plotter(
    K=K, vsys=vsys, CC=CC_day_post, op=orbital_phase_post_eclipse, vsys_kp=vsys_kp
)

Kp_vsys_night_pre, _ = Kp_vsys_Plotter(
    K=K, vsys=vsys, CC=CC_night_pre, op=orbital_phase_pre_eclipse, vsys_kp=vsys_kp
)

Kp_vsys_night_post, _ = Kp_vsys_Plotter(
    K=K, vsys=vsys, CC=CC_night_post, op=orbital_phase_post_eclipse, vsys_kp=vsys_kp
)

Kp_vsys_night = Kp_vsys_night_pre + Kp_vsys_night_post
Kp_vsys_day = Kp_vsys_day_pre + Kp_vsys_day_post


def kp_fitter(kp, op):
    return kp * np.sin(2 * np.pi * op)


fig, ax = plt.subplots(2)
fig.suptitle("Pre-Eclipse")
ax[0].pcolormesh(vsys, orbital_phase_pre_eclipse, CC_day_pre)

ax[0].plot(
    kp_fitter(K[maxIndex(Kp_vsys_day)[0]], orbital_phase_pre_eclipse),
    orbital_phase_pre_eclipse,
    ls="--",
    color="orange",
    label=f"Kp Day Side = {K[maxIndex(Kp_vsys_day)[0]]:.2f}",
)

ax[0].plot(
    kp_fitter(Kp, orbital_phase_pre_eclipse),
    orbital_phase_pre_eclipse,
    ls="--",
    color="pink",
    label=f"Kp = {Kp:.2f} plus vsys from day = {vsys_kp[maxIndex(Kp_vsys_day)[1]]:.2f}",
)

ax[0].legend(loc="upper left")

ax[1].pcolormesh(vsys, orbital_phase_pre_eclipse, CC_night_pre)


ax[1].plot(
    kp_fitter(K[maxIndex(Kp_vsys_night)[0]], orbital_phase_pre_eclipse),
    orbital_phase_pre_eclipse,
    ls="--",
    color="orange",
    label=f"Kp = {K[maxIndex(Kp_vsys_night)[0]]:.2f}",
)

ax[1].plot(
    kp_fitter(Kp, orbital_phase_pre_eclipse),
    orbital_phase_pre_eclipse,
    ls="--",
    color="pink",
    label=f"Kp = {Kp:.2f}",
)

ax[1].legend(loc="upper left")

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

ax[1].plot(
    kp_fitter(K[maxIndex(Kp_vsys_night)[0]], orbital_phase_post_eclipse),
    orbital_phase_post_eclipse,
    ls="--",
    color="blue",
    label=f"Kp Night side = {K[maxIndex(Kp_vsys_night)[0]]:.2f}",
)

ax[1].plot(
    kp_fitter(Kp, orbital_phase_post_eclipse),
    orbital_phase_post_eclipse,
    ls="--",
    color="pink",
    label=f"Kp = {Kp:.2f}",
)

ax[0].plot(
    kp_fitter(K[maxIndex(Kp_vsys_day)[0]], orbital_phase_post_eclipse),
    orbital_phase_post_eclipse,
    ls="--",
    color="orange",
    label=f"Kp Day Side = {K[maxIndex(Kp_vsys_day)[0]]:.2f}",
)

ax[0].plot(
    kp_fitter(Kp, orbital_phase_post_eclipse),
    orbital_phase_post_eclipse,
    ls="--",
    color="pink",
    label=f"Kp = {Kp:.2f}",
)

ax[0].legend()

ax[1].legend()

fig.supxlabel(r"$v_{\text{sys}}$")
fig.supylabel(r"Orbital Phase")


fig, ax = plt.subplots(2, sharex="all", sharey="all")
fig.suptitle(r"$K_p$ - $v_{\text{sys}}$ Night")
fig.supxlabel(r"$v_{\text{sys}}$")
fig.supylabel(r"$K_p$")
ax[0].pcolormesh(vsys_kp, K, Kp_vsys_day)
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
    vsys_kp[maxIndex(Kp_vsys_day)[1]],
    ls="--",
    color="red",
    lw=0.5,
    label=r"Measured $v_{\text{sys}}$ = "
    + f"{vsys_kp[maxIndex(Kp_vsys_day)[1]]:.2f}km/s",
)
ax[0].set_title("Day")
ax[1].set_title("Night")
ax[1].pcolormesh(vsys_kp, K, Kp_vsys_night)

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
    vsys_kp[maxIndex(Kp_vsys_night)[1]],
    ls="--",
    color="red",
    lw=0.5,
    label=r"Measured $v_{\text{sys}}$ = "
    + f"{vsys_kp[maxIndex(Kp_vsys_night)[1]]:.2f}km/s",
)

ax[0].legend(
    loc="upper left",
)
ax[1].legend(
    loc="upper left",
)

np.savez(
    os.path.join(local_path, "day-night_kp_night"),
    tot = combined_Kp_plot,
    day = Kp_vsys_day,
    night = Kp_vsys_night,
    cc_day_pre = CC_day_pre,
    cc_day_post = CC_day_post,
    cc_night_pre = CC_night_pre,
    cc_night_post = CC_night_post,
    cc_tot_pre = CC_pre, 
    cc_tot_post = CC_post, 
        )

plt.show()
