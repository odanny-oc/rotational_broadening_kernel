import numpy as np
import matplotlib.pyplot as plt
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from functions.functions import time_dependent_spectrum
from functions.functions import cross_correlator
from functions.functions import kp_vsys_plotter
from functions.functions import kp_vsys_map_from_flux
from functions.functions import maxindex
from functions.functions import vel_array

from astropy import constants as const
import scipy.signal as scisig
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
local_path = os.path.join(script_dir, "../data")

day_night_atmosphere = np.load(
    os.path.join(local_path, "dual_atmosphere_spectrum.npz")
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
# veq = 35


# x kernel range and op orbital phase
def rotational_broadening_kernel(x, op, veq):
    # Allows float or int values of op to be passed
    if not isinstance(op, np.ndarray):
        op = np.array([op])

    kernel_array = np.zeros(shape=(op.shape[0], x.shape[0]))
    kernel_range_array = []
    # Sets valid range given veq
    ref_range = np.array([i for i in x if abs(i) <= veq])

    ref_kernel = np.sqrt(1 - (ref_range / veq) ** 2)

    # Pads evenly the rest of the range with zeros
    ref_padding = abs(x.shape[0] - ref_range.shape[0]) // 2
    ref_kernel = np.pad(ref_kernel, ref_padding, "constant")

    normaliser = np.sum(ref_kernel)
    ref_kernel /= normaliser

    # Takes the half kernel (right side)
    ref_kernel[: x.shape[0] // 2] = np.zeros(x.shape[0] // 2)

    for i, op_i in enumerate(op):
        # Fit all op values between 0 and 1
        if op_i < 0:
            ref_op = op_i + 1
        else:
            ref_op = op_i

        vel = veq * np.cos(2 * np.pi * ref_op)

        # Consider 0 case
        if vel == 0:
            kernel = ref_kernel
            kernel_range = ref_range
            # Consider opposite side of orbit
            if ref_op < 0.5:
                kernel = np.flip(kernel)

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

# Pre/post eclipse
orbital_phase_pre_eclipse = np.linspace(0.33, 0.43, n_exposure)  # time/period
orbital_phase_post_eclipse = np.linspace(0.55, 0.66, n_exposure)  # time/period

# Pre/post transit
# orbital_phase_pre_eclipse = np.linspace(-0.12, -0.04, n_exposure)  # time/period
# orbital_phase_post_eclipse = np.linspace(0.04, 0.12, n_exposure)  # time/period

x = vel_array(30, resolution)

# Kernel generation
time_dependent_broadening_kernels_pre_eclipse, kernel_range_pre = rotational_broadening_kernel(
    x, orbital_phase_pre_eclipse, veq
)

anti_kernels_pre_eclipse, anti_kernel_range_pre = rotational_broadening_kernel(
    x, orbital_phase_pre_eclipse - 0.5, veq
)

time_dependent_broadening_kernels_post_eclipse, kernel_range_post = (
    rotational_broadening_kernel(x, orbital_phase_post_eclipse, veq)
)
anti_kernels_post_eclipse, anti_kernel_range_post = rotational_broadening_kernel(
    x, orbital_phase_post_eclipse - 0.5, veq
)


def cc_fitter(kernels, kernel_range, x):
    """
    Finds weighted average of kernel (expected percieved doppler shift)
    """
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


# Can add these corrections to vp to see if they are seen in the actual trace (not done here)
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

vsys = np.linspace(-200, 200, 1001)
vsys_kp = np.linspace(-30, 30, 1001)


def kp_fitter(kp, op):
    """
    Plots vp for given kp and orbital phase
    """
    return kp * np.sin(2 * np.pi * op)

# Bulk velocity plus corrections
cc_pre_day_fit = kp_fitter(Kp, orbital_phase_pre_eclipse) + cc_pre_day_correction
cc_post_day_fit = kp_fitter(Kp, orbital_phase_post_eclipse) + cc_post_day_correction

cc_pre_night_fit = kp_fitter(Kp, orbital_phase_pre_eclipse) + cc_pre_night_correction
cc_post_night_fit = kp_fitter(Kp, orbital_phase_post_eclipse) + cc_post_night_correction

K = np.linspace(180, 210, 1001)

time_res = 1001
x2 = vel_array(Kp + 10 , resolution)
op_tot = np.linspace(0, 1, time_res)

dv = resolution / const.c.value
vp = Kp * np.sin(2 * np.pi * op_tot)

# Kernel generation for direct plots
cc_kernels, _ = rotational_broadening_kernel(x2, op_tot, veq)
cc_kernels_pre, _ = rotational_broadening_kernel(x2, orbital_phase_pre_eclipse, veq)
cc_kernels_post, _ = rotational_broadening_kernel(x2, orbital_phase_post_eclipse, veq)

# Velocity resolution
dv = const.c.value * 1e-3 / resolution


def kernel_to_cc(kernels, op):
    """
    Function to move kernel to bulk velocity of planet so that we can do the Kp-vsys plot directly
    """
    vp = Kp * np.sin(2 * np.pi * op)
    for i,kernel in enumerate(kernels):
        index =  vp[i]//dv
        index = int(index)
        kernels[i] = np.roll(kernel, index)
    return kernels


# Shifted Kernels
cc_kernels = kernel_to_cc(cc_kernels, op_tot)
cc_kernels_pre = kernel_to_cc(cc_kernels_pre, orbital_phase_pre_eclipse)
cc_kernels_post = kernel_to_cc(cc_kernels_post, orbital_phase_post_eclipse)

# vsys for Kp-vsys plots
vsys_kp = np.linspace(-30,30, x2.shape[0])

cc_pre_fit, _ = kp_vsys_plotter(K, x2, orbital_phase_pre_eclipse, cc_kernels_pre, vsys_kp)
cc_post_fit, _ = kp_vsys_plotter(K, x2, orbital_phase_post_eclipse, cc_kernels_post, vsys_kp)

cc_fit = cc_pre_fit + cc_post_fit

fig, ax = plt.subplots()
fig.suptitle('Kp - vsys from Kernels (Day-Side Eclipse)')
ax.pcolormesh(vsys_kp, K , cc_fit)
ax.axhline(K[maxindex(cc_fit)[0]], lw=2, ls='--', color='r', label= f'Kp = {K[maxindex(cc_fit)[0]]:.2f}')
ax.axvline(vsys_kp[maxindex(cc_fit)[1]], lw=2, ls='--', color='r', label= f'vsys = {vsys_kp[maxindex(cc_fit)[1]]:.2f}')
ax.legend(loc = 'upper left')

fig, ax = plt.subplots()
fig.suptitle('Trace Pre-Secondary Eclipse')
fig.supxlabel('vsys')
fig.supylabel('Orbital Phase')
ax.pcolormesh(x2, orbital_phase_pre_eclipse, cc_kernels_pre)
ax.plot(kp_fitter(K[maxindex(cc_fit)[0]], orbital_phase_pre_eclipse), orbital_phase_pre_eclipse, ls = '--', color='red')

fig, ax = plt.subplots()
fig.suptitle('Trace Post-Secondary Eclipse')
fig.supxlabel('vsys')
fig.supylabel('Orbital Phase')
ax.pcolormesh(x2, orbital_phase_post_eclipse, cc_kernels_post)
ax.plot(kp_fitter(K[maxindex(cc_fit)[0]], orbital_phase_post_eclipse), orbital_phase_post_eclipse, ls = '--', color='red')

# Kp-vsys plots for transit
orbital_phase_pre_eclipse = np.linspace(-0.12, -0.04, n_exposure)  # time/period
orbital_phase_post_eclipse = np.linspace(0.04, 0.12, n_exposure)  # time/period

cc_kernels_pre, _ = rotational_broadening_kernel(x2, orbital_phase_pre_eclipse, veq)
cc_kernels_post, _ = rotational_broadening_kernel(x2, orbital_phase_post_eclipse, veq)

cc_kernels = kernel_to_cc(cc_kernels, op_tot)
cc_kernels_pre = kernel_to_cc(cc_kernels_pre, orbital_phase_pre_eclipse)
cc_kernels_post = kernel_to_cc(cc_kernels_post, orbital_phase_post_eclipse)

cc_pre_fit, _ = kp_vsys_plotter(K, x2, orbital_phase_pre_eclipse, cc_kernels_pre, vsys_kp)
cc_post_fit, _ = kp_vsys_plotter(K, x2, orbital_phase_post_eclipse, cc_kernels_post, vsys_kp)

cc_fit = cc_pre_fit + cc_post_fit

fig, ax = plt.subplots()
fig.suptitle('Trace Pre-Transit')
fig.supxlabel('vsys')
fig.supylabel('Orbital Phase')
ax.pcolormesh(x2, orbital_phase_pre_eclipse, cc_kernels_pre)
ax.plot(kp_fitter(K[maxindex(cc_fit)[0]], orbital_phase_pre_eclipse), orbital_phase_pre_eclipse, ls = '--', color='red')

fig, ax = plt.subplots()
fig.suptitle('Trace Post-Transit')
fig.supxlabel('vsys')
fig.supylabel('Orbital Phase')
ax.pcolormesh(x2, orbital_phase_post_eclipse, cc_kernels_post)
ax.plot(kp_fitter(K[maxindex(cc_fit)[0]], orbital_phase_post_eclipse), orbital_phase_post_eclipse, ls = '--', color='red')


fig, ax = plt.subplots()
fig.suptitle('Kp - vsys from Kernels (Day-Side Transit)')
ax.pcolormesh(vsys_kp, K , cc_fit)
ax.axhline(K[maxindex(cc_fit)[0]], lw=1, ls='--', color='r', label= f'Kp = {K[maxindex(cc_fit)[0]]:.2f}')
ax.axvline(vsys_kp[maxindex(cc_fit)[1]], lw=1, ls='--', color='r', label= f'vsys = {vsys_kp[maxindex(cc_fit)[1]]:.2f}')
ax.legend(loc = 'upper left')
plt.show()


"""
Can clearly see the assuptiom of a sinusodial shape is wrong and the rotational broadening shifts the flux in unexpected ways
"""
