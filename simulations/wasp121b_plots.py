import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from functions.functions import time_dependent_spectrum
from functions.functions import cross_correlator
from functions.functions import kp_vsys_plotter
from functions.functions import kp_vsys_map_from_flux
from functions.functions import maxindex
from functions.functions import vel_array
from functions.rotational_broadening_kernel import rotational_broadening_kernel

script_dir = os.path.dirname(os.path.abspath(__file__))
local_path = os.path.join(script_dir, "../data")

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

veq = 2 * np.pi * Rpl / period  # km/s ~ 7km/s

# veq = 35 # effect exaggerated

resolution = 400000
dv = const.c.value * 1e-3 / resolution
n_exposure = 300

orbital_phase_pre_eclipse = np.linspace(0.334, 0.425, n_exposure)  # time/period
orbital_phase_post_eclipse = np.linspace(0.550, 0.640, n_exposure)  # time/period

x = vel_array(30, resolution)

# Kernel generation
time_dependent_broadening_kernels_pre_eclipse = rotational_broadening_kernel(x, orbital_phase_pre_eclipse, veq)

time_dependent_broadening_kernels_post_eclipse =  rotational_broadening_kernel(x, orbital_phase_post_eclipse, veq)

wl = wasp121b_spectrum["wl"]
flux = wasp121b_spectrum["flux"]
flux -= np.mean(flux)

# Spectrum generation
convolved_spectrum_pre_eclipse = time_dependent_spectrum(
    wl,
    flux,
    orbital_phase_pre_eclipse,
    Kp,
    time_dependent_broadening_kernels_pre_eclipse,
)

# Arrays for CC and Kp-vsys plots
vsys = np.linspace(-175, 175, 1001)
K = np.linspace(Kp - 20, Kp + 20, 1001)
vsys_kp = np.linspace(-50, 50, 1001)

CC_pre = cross_correlator(wl, flux, vsys * 1000, convolved_spectrum_pre_eclipse)

K_vsys_map_pre_eclipse, CC_shifted = kp_vsys_plotter(
    K, vsys, orbital_phase_pre_eclipse, CC_pre, vsys_kp=vsys_kp
)

# Demonstation of how Kp-vsys plot works, putting the trace of the cc into the "rest frame"
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


#Spectrum post eclipse
convolved_spectrum_post_eclipse = time_dependent_spectrum(
    wl,
    flux,
    orbital_phase_post_eclipse,
    Kp,
    time_dependent_broadening_kernels_post_eclipse,
)

CC_post = cross_correlator(wl, flux, vsys * 1000, convolved_spectrum_post_eclipse)

K_vsys_map_post_eclipse, _ = kp_vsys_plotter(
    K, vsys, orbital_phase_post_eclipse, CC_post, vsys_kp
)

combined_Kp_plot = K_vsys_map_post_eclipse + K_vsys_map_pre_eclipse

Kp_from_plot = K[maxindex(combined_Kp_plot)[0]]
Kp_from_plot_pre = K[maxindex(K_vsys_map_pre_eclipse)]
Kp_from_plot_post = K[maxindex(K_vsys_map_post_eclipse)]

fig, ax = plt.subplots(3, sharex="all", sharey="all")
fig.suptitle(r"$K_p$ - $v_{\text{sys}}$ Plots for Different Simulations")
ax[0].set_title(r"$V_{\text{eq}}$ Kernel")
ax[1].set_title(r"No Kernel")
ax[2].set_title(r"Gaussian Kernel")
ax[0].pcolormesh(vsys_kp, K, combined_Kp_plot)
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
ax[0].axvline(vsys_kp[maxindex(combined_Kp_plot)[1]], ls="--", color="orange", lw=0.5)
ax[0].legend(loc="upper left")


# Comparison with no kernel and Gaussian kernel (instrumental broadening)
unconvolved_spec_pre, _ = kp_vsys_map_from_flux(
    wl,
    flux,
    orbital_phase_pre_eclipse,
    vsys,
    Kp,
    vsys_kp=vsys_kp,
    K=K,
)

unconvolved_spec_post, _ = kp_vsys_map_from_flux(
    wl,
    flux,
    orbital_phase_post_eclipse,
    vsys,
    Kp,
    vsys_kp=vsys_kp,
    K=K,
)

total_unconvolved_spec = unconvolved_spec_pre + unconvolved_spec_post

ax[1].pcolormesh(vsys_kp, K, total_unconvolved_spec)
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
ax[1].legend(loc="upper left")


def gaussian(x, sigma):
    return np.exp(-0.5 * (x / sigma) ** 2)


g = gaussian(x, 2)

gaussian_spec_pre, _ = kp_vsys_map_from_flux(
    wl,
    flux,
    orbital_phase_pre_eclipse,
    vsys,
    Kp,
    g,
    vsys_kp=vsys_kp,
    K=K,
)

gaussian_spec_post, _ = kp_vsys_map_from_flux(
    wl,
    flux,
    orbital_phase_post_eclipse,
    vsys,
    Kp,
    g,
    vsys_kp=vsys_kp,
    K=K,
)

total_gaussian_spec = gaussian_spec_pre + gaussian_spec_post

ax[2].pcolormesh(vsys_kp, K, total_gaussian_spec)
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
ax[2].legend(loc="upper left")

plt.show()
