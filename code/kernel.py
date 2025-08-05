import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os, subprocess
from functions import vel_array

mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.size": 35,
    "figure.figsize": [20, 15],
})

home_path = os.environ["HOME"]

local_path = home_path + "/exoplanet_atmospheres/code"

local_images = home_path + "/exoplanet_atmospheres/images"

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


# x kernel range and op orbital phase
def broadening_kernel_op(x, op):
    # Allows float or int values of op to be passed
    if not isinstance(op, np.ndarray):
        op = np.array([op])

    veq_local = veq

    kernel_array = np.zeros(shape=(op.shape[0], x.shape[0]))
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
            # Consider opposite side of orbit
            if ref_op < 0.5:
                kernel = np.flip(kernel)

        # else:
        #     # Check if planet is about to enter secondary eclipse
        #     if abs(ref_op - 0.5) < tdur / (2 * period) or abs(ref_op + 0.5) < tdur / (
        #         2 * period
        #     ):
        #         # Zero out signal if planet is eclipsed
        #         if abs(ref_op - 0.5) < tfull / (2 * period) or abs(
        #             ref_op + 0.5
        #         ) < tfull / (2 * period):
        #             kernel = np.zeros(x.shape[0])
        #         else:
        #             # Gradually reduce signal otherwise (from left or from right depending on op)
        #             if ref_op > 0:
        #                 vel = veq * np.cos(
        #                     (2 * np.pi * (ref_op - 0.5)) * (period_scaler)
        #                 )
        #             else:
        #                 vel = veq * np.cos(
        #                     (2 * np.pi * (ref_op + 0.5)) * (period_scaler)
        #                 )
        #             # Create kernel
        #             range = np.array([i for i in x if abs(i) <= abs(vel)])
        #             kernel = np.sqrt(1 - (range / abs(vel)) ** 2) / normaliser
        #             padding = abs(x.shape[0] - range.shape[0]) // 2
        #             kernel = np.pad(kernel, padding, "constant")
        #             if vel < 0:
        #                 kernel[x.shape[0] // 2 :] = np.zeros(x.shape[0] // 2 + 1)
        #                 kernel += ref_kernel
        #             if vel > 0:
        #                 kernel[: x.shape[0] // 2] = np.zeros(x.shape[0] // 2)
        #                 kernel = ref_kernel - kernel
        #             if ref_op > 0.5:
        #                 kernel = np.flip(kernel)
            # Else not about to be eclipsed takes original value of vel
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
        # Saves kernel to kernel array
        kernel_array[i] = kernel

    if kernel_array.shape[0] == 1:
        return kernel_array[0], full_kernel
    else:
        return kernel_array, full_kernel

resolution = 400000
x = vel_array(10, resolution)

# plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')


op = np.linspace(-0.56, -0.56 + 1, 11)
kernels, ref_kernel = broadening_kernel_op(x, op)
anti_kernels, _ = broadening_kernel_op(x, op - 0.5)


for i, ker in enumerate(kernels):
    fig, ax = plt.subplots()
    ax.plot(x, ker, lw=5)
    ax.axis('off')
    plt.savefig(os.path.join(local_images, f"kernel_{op[i]:.2f}.png"))


for i, ker in enumerate(kernels):
    fig, ax = plt.subplots()
    ax.plot(x, ker, lw=5)
    ax.plot(x, anti_kernels[i], lw=5)
    ax.axis('off')
    plt.savefig(os.path.join(local_images, f"anti-kernel_{op[i]:.2f}.png"))

fig, ax = plt.subplots()
ax.plot(x, ref_kernel, lw=5)
ax.axis('off')
plt.savefig(os.path.join(local_images, f"ref_kernel.png"))
