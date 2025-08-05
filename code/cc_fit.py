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
            # # Check if planet is about to enter secondary eclipse
            # if abs(ref_op - 0.5) < tdur / (2 * period) or abs(ref_op + 0.5) < tdur / (
            #     2 * period
            # ):
            #     # Zero out signal if planet is eclipsed
            #     if abs(ref_op - 0.5) < tfull / (2 * period) or abs(
            #         ref_op + 0.5
            #     ) < tfull / (2 * period):
            #         kernel = np.zeros(x.shape[0])
            #     else:
            #         # Gradually reduce signal otherwise (from left or from right depending on op)
            #         if ref_op > 0:
            #             vel = veq * np.cos(
            #                 (2 * np.pi * (ref_op - 0.5)) * (period_scaler)
            #             )
            #         else:
            #             vel = veq * np.cos(
            #                 (2 * np.pi * (ref_op + 0.5)) * (period_scaler)
            #             )
            #         # Create kernel
            #         range = np.array([i for i in x if abs(i) <= abs(vel)])
            #         kernel = np.sqrt(1 - (range / abs(vel)) ** 2) / normaliser
            #         padding = abs(x.shape[0] - range.shape[0]) // 2
            #         kernel = np.pad(kernel, padding, "constant")
            #         if vel < 0:
            #             kernel[x.shape[0] // 2 :] = np.zeros(x.shape[0] // 2 + 1)
            #             kernel += ref_kernel
            #         if vel > 0:
            #             kernel[: x.shape[0] // 2] = np.zeros(x.shape[0] // 2)
            #             kernel = ref_kernel - kernel
            #         if ref_op > 0.5:
            #             kernel = np.flip(kernel)
            # Else not about to be eclipsed takes original value of vel
            # else:
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

orbital_phase_pre_eclipse = np.linspace(0.33, 0.43, n_exposure)  # time/period
orbital_phase_post_eclipse = np.linspace(0.55, 0.66, n_exposure)  # time/period

# orbital_phase_pre_eclipse = np.linspace(-0.12, -0.04, n_exposure)  # time/period
# orbital_phase_post_eclipse = np.linspace(0.04, 0.12, n_exposure)  # time/period

x = vel_array(30, resolution)

time_dependent_broadening_kernels_pre_eclipse, kernel_range_pre = broadening_kernel_op(
    x, orbital_phase_pre_eclipse
)

anti_kernels_pre_eclipse, anti_kernel_range_pre = broadening_kernel_op(
    x, orbital_phase_pre_eclipse - 0.5
)

time_dependent_broadening_kernels_post_eclipse, kernel_range_post = (
    broadening_kernel_op(x, orbital_phase_post_eclipse)
)
anti_kernels_post_eclipse, anti_kernel_range_post = broadening_kernel_op(
    x, orbital_phase_post_eclipse - 0.5
)


def cc_fitter(kernels, kernel_range, x):
    predicted_trace = np.empty(kernels.shape[0])
    for i in range(kernels.shape[0]):
        weights = kernels[i][
            np.where(x == kernel_range[i][0])[0][0] : np.where(
                x == kernel_range[i][-1]
            )[0][0]
            + 1
        ]
        weighted_average = np.average(kernel_range[i], weights=weights)
        predicted_trace[i] = weighted_average
    return predicted_trace


cc_pre_day_correction = cc_fitter(
    time_dependent_broadening_kernels_pre_eclipse, kernel_range_pre, x
)

cc_post_day_correction = cc_fitter(
    time_dependent_broadening_kernels_post_eclipse, kernel_range_post, x
)

cc_pre_night_correction = cc_fitter(anti_kernels_pre_eclipse, anti_kernel_range_pre, x)

cc_post_night_correction = cc_fitter(
    anti_kernels_post_eclipse, anti_kernel_range_post, x
)

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
CC_pre_day = Cross_Correlator(wl, flux_day, vsys * 1000, total_convolved_spectrum_pre)
CC_pre_night = Cross_Correlator(
    wl, flux_night, vsys * 1000, total_convolved_spectrum_pre
)

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
CC_post_day = Cross_Correlator(wl, flux_day, vsys * 1000, total_convolved_spectrum_post)
CC_post_night = Cross_Correlator(
    wl, flux_night, vsys * 1000, total_convolved_spectrum_post
)

K_vsys_map_post_eclipse, _ = Kp_vsys_Plotter(
    K, vsys, orbital_phase_post_eclipse, CC_post, vsys_kp=vsys_kp
)

K_vsys_map_pre_eclipse_day, _ = Kp_vsys_Plotter(
    K, vsys, orbital_phase_pre_eclipse, CC_pre_day, vsys_kp=vsys_kp
)
K_vsys_map_post_eclipse_day, _ = Kp_vsys_Plotter(
    K, vsys, orbital_phase_post_eclipse, CC_post_day, vsys_kp=vsys_kp
)

K_vsys_map_pre_eclipse_night, _ = Kp_vsys_Plotter(
    K, vsys, orbital_phase_pre_eclipse, CC_pre_night, vsys_kp=vsys_kp
)
K_vsys_map_post_eclipse_night, _ = Kp_vsys_Plotter(
    K, vsys, orbital_phase_post_eclipse, CC_post_night, vsys_kp=vsys_kp
)

Kp_tot_day = K_vsys_map_pre_eclipse_day + K_vsys_map_post_eclipse_day
Kp_tot_night = K_vsys_map_pre_eclipse_night + K_vsys_map_post_eclipse_night

Kp_from_plot_day = K[maxIndex(Kp_tot_day)[0]]
Kp_from_plot_night = K[maxIndex(Kp_tot_night)[0]]

combined_Kp_plot = K_vsys_map_post_eclipse + K_vsys_map_pre_eclipse

max_lines_pre = maxIndex(K_vsys_map_pre_eclipse)
max_lines_post = maxIndex(K_vsys_map_post_eclipse)
max_lines = maxIndex(combined_Kp_plot)

Kp_from_plot = K[max_lines[0]]
Kp_from_plot_pre = K[max_lines_pre[0]]
Kp_from_plot_post = K[max_lines_post[0]]


fig, ax = plt.subplots()
fig.suptitle(r"$K_p$ - $v_{\text{sys}}$ Plots for Different Kernels")
ax.set_title("Day-Night Kernel")
fig.supxlabel(r"$v_{\text{sys}}$ (km/s)")
fig.supylabel(r"$K_p$ (km/s)")

ax.pcolormesh(vsys_kp, K, combined_Kp_plot)
ax.axhline(Kp, ls="--", color="red", lw=0.5, label=r"Actual $K_p$ = " + f"{Kp:.2f}km/s")
ax.axvline(0, ls="--", color="red", lw=0.5)
ax.axhline(
    Kp_from_plot,
    ls="--",
    color="blue",
    lw=0.5,
    label=r"Measured $K_p$ = " + f"{Kp_from_plot:.2f}km/s",
)
ax.axvline(
    vsys_kp[max_lines[1]],
    ls="--",
    color="blue",
    lw=0.5,
    label=f"Measured vsys = {vsys_kp[max_lines[1]]:.2f}",
)
ax.legend(
    edgecolor="k",
    facecolor="w",
    framealpha=1,
    fancybox=True,
    loc="upper left",
)
ax.annotate(
    r"$\Delta K_p$ = " + f"{( Kp -  Kp_from_plot):.2f}km/s",
    xy=(0.9, 0.85),
    xycoords="axes fraction",
    size=10,
    bbox=dict(fc="w", ec="k", boxstyle="round", linewidth=2),
)


def kp_fitter(kp, op):
    return kp * np.sin(2 * np.pi * op)


cc_pre_day_fit = kp_fitter(Kp, orbital_phase_pre_eclipse) + cc_pre_day_correction
cc_post_day_fit = kp_fitter(Kp, orbital_phase_post_eclipse) + cc_post_day_correction

cc_pre_night_fit = kp_fitter(Kp, orbital_phase_pre_eclipse) + cc_pre_night_correction
cc_post_night_fit = kp_fitter(Kp, orbital_phase_post_eclipse) + cc_post_night_correction

K2 = np.linspace(180, 200, 1001)


def kp_finder(K, cc_fit, op):
    kp_array = np.empty(K.shape[0])
    for i, kp in enumerate(K):
        vp = kp * np.sin(2 * np.pi * op)
        kp_array[i] = np.sum(vp, axis=0)

    cc_sum = np.sum(cc_fit, axis=0)
    closest_idx = np.abs(kp_array - cc_sum).argmin()
    return K[closest_idx]


kp_fit_pre_day = kp_finder(K2, cc_pre_day_fit, orbital_phase_pre_eclipse)
kp_fit_post_day = kp_finder(K2, cc_post_day_fit, orbital_phase_post_eclipse)
kp_fit_pre_night = kp_finder(K2, cc_pre_day_fit, orbital_phase_pre_eclipse)
kp_fit_post_night = kp_finder(K2, cc_post_day_fit, orbital_phase_post_eclipse)

fig, ax = plt.subplots(2)
fig.suptitle("Pre-Eclipse")
fig.supxlabel("vsys (km/s)")
fig.supylabel(r"Orbital Phase ($\phi$)")
ax[0].pcolormesh(vsys, orbital_phase_pre_eclipse, CC_pre_day)
ax[0].plot(
    kp_fitter(Kp_from_plot_day, orbital_phase_pre_eclipse),
    orbital_phase_pre_eclipse,
    ls="--",
    color="orange",
    label=f"Kp = {Kp_from_plot_day:.2f}",
)
ax[0].plot(cc_pre_day_fit, orbital_phase_pre_eclipse, ls="--", color="green")
ax[0].plot(
    kp_fitter(kp_fit_pre_day, orbital_phase_pre_eclipse),
    orbital_phase_pre_eclipse,
    ls="--",
    color="red",
    label=f"Kp = {kp_fit_pre_day:.2f}",
)
ax[0].legend(loc="upper left")

ax[1].pcolormesh(vsys, orbital_phase_pre_eclipse, CC_pre_night)
ax[1].plot(
    kp_fitter(Kp_from_plot_night, orbital_phase_pre_eclipse),
    orbital_phase_pre_eclipse,
    ls="--",
    color="orange",
    label=f"Kp = {Kp_from_plot_night:.2f}",
)
ax[1].plot(cc_pre_night_fit, orbital_phase_pre_eclipse, ls="--", color="green")
ax[1].plot(
    kp_fitter(kp_fit_pre_night, orbital_phase_pre_eclipse),
    orbital_phase_pre_eclipse,
    ls="--",
    color="red",
    label=f"Kp = {kp_fit_pre_night:.2f}",
)
ax[1].legend(loc="upper left")

fig, ax = plt.subplots(2)
fig.suptitle("Post-Eclipse")
fig.supxlabel("vsys (km/s)")
fig.supylabel(r"Orbital Phase ($\phi$)")
ax[0].pcolormesh(vsys, orbital_phase_post_eclipse, CC_post_day)
ax[0].plot(
    kp_fitter(Kp_from_plot_day, orbital_phase_post_eclipse),
    orbital_phase_post_eclipse,
    ls="--",
    color="orange",
    label=f"Kp = {Kp_from_plot_day:.2f}",
)
ax[0].plot(cc_post_day_fit, orbital_phase_post_eclipse, ls="--", color="green")
ax[0].plot(
    kp_fitter(kp_fit_post_day, orbital_phase_post_eclipse),
    orbital_phase_post_eclipse,
    ls="--",
    color="red",
    label=f"Kp = {kp_fit_post_day:.2f}",
)
ax[0].legend(loc="upper left")

ax[1].pcolormesh(vsys, orbital_phase_post_eclipse, CC_post_night)
ax[1].plot(
    kp_fitter(Kp_from_plot_night, orbital_phase_post_eclipse),
    orbital_phase_post_eclipse,
    ls="--",
    color="orange",
    label=f"Kp = {Kp_from_plot_night:.2f}",
)
ax[1].plot(cc_post_night_fit, orbital_phase_post_eclipse, ls="--", color="green")
ax[1].plot(
    kp_fitter(kp_fit_post_night, orbital_phase_post_eclipse),
    orbital_phase_post_eclipse,
    ls="--",
    color="red",
    label=f"Kp = {kp_fit_post_night:.2f}",
)
ax[1].legend(loc="upper left")

time_res = 1001
x2 = vel_array(Kp + 10 , resolution)
print(x2.shape)
op_tot = np.linspace(0, 1, time_res)

dv = resolution / const.c.value
vp = Kp * np.sin(2 * np.pi * op_tot)

cc_kernels, _ = broadening_kernel_op(x2, op_tot)
cc_kernels_pre, _ = broadening_kernel_op(x2, orbital_phase_pre_eclipse)
cc_kernels_post, _ = broadening_kernel_op(x2, orbital_phase_post_eclipse)


def kernel_to_cc(kernels, x, op):
    vp = Kp * np.sin(2 * np.pi * op)
    for i,kernel in enumerate(kernels):
        index = np.abs(x - vp[i]).argmin()
        if index < (x2.shape[0])//2:
            index = abs( (x2.shape[0])//2 - index )
            kernels[i] = np.roll(kernel, -index)
        else:
            index = abs( (x2.shape[0])//2 - index )
            kernels[i] = np.roll(kernel, index)
    return kernels


cc_kernels = kernel_to_cc(cc_kernels, x2, op_tot)
cc_kernels_pre = kernel_to_cc(cc_kernels_pre, x2, orbital_phase_pre_eclipse)
cc_kernels_post = kernel_to_cc(cc_kernels_post, x2, orbital_phase_post_eclipse)


vsys_kp = np.linspace(-30,30, x2.shape[0])

cc_pre_fit, _ = Kp_vsys_Plotter(K, x2, orbital_phase_pre_eclipse, cc_kernels_pre, vsys_kp)
cc_post_fit, _ = Kp_vsys_Plotter(K, x2, orbital_phase_post_eclipse, cc_kernels_post, vsys_kp)

cc_fit = cc_pre_fit + cc_post_fit

fig, ax = plt.subplots()
fig.suptitle('Kp - vsys from Kernels (Day-Side Eclipse)')
ax.pcolormesh(vsys_kp, K , cc_fit)
ax.axhline(K[maxIndex(cc_fit)[0]], lw=2, ls='--', color='r', label= f'Kp = {K[maxIndex(cc_fit)[0]]:.2f}')
ax.axvline(vsys_kp[maxIndex(cc_fit)[1]], lw=2, ls='--', color='r', label= f'vsys = {vsys_kp[maxIndex(cc_fit)[1]]:.2f}')
ax.legend(loc = 'upper left')

fig, ax = plt.subplots()
fig.supxlabel('vsys')
fig.supylabel('Orbital Phase')
ax.pcolormesh(x2, orbital_phase_pre_eclipse, cc_kernels_pre)
ax.plot(kp_fitter(K[maxIndex(cc_fit)[0]], orbital_phase_pre_eclipse), orbital_phase_pre_eclipse, ls = '--', color='red')

fig, ax = plt.subplots()
fig.supxlabel('vsys')
fig.supylabel('Orbital Phase')
ax.pcolormesh(x2, orbital_phase_post_eclipse, cc_kernels_post)
ax.plot(kp_fitter(K[maxIndex(cc_fit)[0]], orbital_phase_post_eclipse), orbital_phase_post_eclipse, ls = '--', color='red')

orbital_phase_pre_eclipse = np.linspace(-0.12, -0.04, n_exposure)  # time/period
orbital_phase_post_eclipse = np.linspace(0.04, 0.12, n_exposure)  # time/period

cc_kernels_pre, _ = broadening_kernel_op(x2, orbital_phase_pre_eclipse)
cc_kernels_post, _ = broadening_kernel_op(x2, orbital_phase_post_eclipse)

cc_kernels = kernel_to_cc(cc_kernels, x2, op_tot)
cc_kernels_pre = kernel_to_cc(cc_kernels_pre, x2, orbital_phase_pre_eclipse)
cc_kernels_post = kernel_to_cc(cc_kernels_post, x2, orbital_phase_post_eclipse)

fig, ax = plt.subplots()
ax.pcolormesh(x2, orbital_phase_pre_eclipse, cc_kernels_pre)

fig, ax = plt.subplots()
ax.pcolormesh(x2, orbital_phase_post_eclipse, cc_kernels_post)

vsys_kp = np.linspace(-30,30, x2.shape[0])

cc_pre_fit, _ = Kp_vsys_Plotter(K, x2, orbital_phase_pre_eclipse, cc_kernels_pre, vsys_kp)
cc_post_fit, _ = Kp_vsys_Plotter(K, x2, orbital_phase_post_eclipse, cc_kernels_post, vsys_kp)

cc_fit = cc_pre_fit + cc_post_fit

fig, ax = plt.subplots()
fig.suptitle('Kp - vsys from Kernels (Day-Side Transit)')
ax.pcolormesh(vsys_kp, K , cc_fit)
ax.axhline(K[maxIndex(cc_fit)[0]], lw=1, ls='--', color='r', label= f'Kp = {K[maxIndex(cc_fit)[0]]:.2f}')
ax.axvline(vsys_kp[maxIndex(cc_fit)[1]], lw=1, ls='--', color='r', label= f'vsys = {vsys_kp[maxIndex(cc_fit)[1]]:.2f}')
ax.legend(loc = 'upper left')
plt.show()
