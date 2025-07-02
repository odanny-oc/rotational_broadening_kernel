import numpy as np
import matplotlib.pyplot as plt
import os
from astropy import constants as const
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.plotlib import plot_radtrans_opacities
from petitRADTRANS import physical_constants as cst, planet
from petitRADTRANS.config import petitradtrans_config_parser
from petitRADTRANS.physics import temperature_profile_function_guillot_global
import scipy.signal as scisig
from petitRADTRANS.spectral_model import SpectralModel
from petitRADTRANS.math import resolving_space
from petitRADTRANS.planet import Planet

planet = Planet.get("HD 189733 b")
resolution = 80000


def wl_rspace(w_min, w_max, R):
    """

    Simple function to return a wavelength array with constant resolution



    w_min/w_max, min/max wavelengths

    R - resolution



    Simple idea is that taking a draw from logspace has constant resolution. Main bit

    of faff here is to get the number of draws to match the resolution. Easy enough by

    trail and error, though this gets most accurate to 1 pixel.



    Reminder (ln natural log, lg base 10):

    R = w / dw # resolution

    d ln w = dw / w = 1/R

    or d lg w = dw / w * lg e = lg e / R



    ie if d ln w is constant then resolution is constant, and given by lg e / R



    therefore samples within range to get R is R/lg e * [lg(w_max) - lg(w_min)] + 1

    which can be rewritten in multiple ways



    """

    dlogwl = np.log10(np.e) / R

    #     N = np.log10(w_max/w_min)/dlogwl

    #     C = N%1

    #     if np.round(C)==0:

    #       w_max = w_max * 10**(-C*dlogwl)

    #     else:

    #       w_max = w_max * 10**((1-C)*dlogwl)

    # round to nearest int

    N = round(np.log10(w_max / w_min) / dlogwl + 1)

    # use geomspace to get even log sampling

    wl = np.geomspace(w_min, w_max, N)

    # do with logspace instead?

    # w = np.logspace(np.log10(w_min),np.log10(w_max),round(np.log10(w_max/w_min)*R/log10(np.e)+1)

    return wl


# Planet and Star Parameters

Rpl = cst.r_jup_mean
Rs = cst.r_sun
Mp = 0.85 * cst.m_jup
Ms = 1.1 * cst.m_sun
a = 0.045 * cst.au  # au to cm
period = 1.5 * (24 * 60**2)  # days to seconds
eccen = 0.1
observer_angle = 90  # degrees
Kp = 2 * np.pi / period * a * np.sin(observer_angle) / (np.sqrt(1 - eccen**2))
b = 0.01

tdur = period / np.pi * np.arcsin(Rs / a * np.sqrt((1 - Rpl / Rs) ** 2 - b**2))
tfull = period / np.pi * np.arcsin(Rs / a * np.sqrt((1 + Rpl / Rs) ** 2 - b**2))

n_exposures = 150

data_wavelengths = (
    resolving_space(0.6, 0.61, resolution) * 1e-4
)  # (cm) generate wavelengths at a constant resolving power
times = tdur * (np.linspace(-0.5, 0.5, n_exposures))
orbital_phases = times / period
mid_transit_time = 0  # (s)
star_radial_velocity = 0  # (cm.s-1) V_sys
barycentric_velocities = np.linspace(-13.25e6, -23.55e6, times.size)  # (cm.s-1) V_bary
# why need a range? acceleration?
vp = Kp * np.sin(orbital_phases * 2 * np.pi)

# Uncertainties assuming a S/N of 2000
data_uncertainties = 5e-4 * np.ones((1, n_exposures, data_wavelengths.size))

spectral_model = SpectralModel(
    # Radtrans parameters
    pressures=np.logspace(-6, 2, 100),
    line_opacity_mode="lbl",
    # kine_by_line_opacity_sampling= 5,
    line_species=["Fe"],
    # rayleigh_species=['H2', 'He'],
    gas_continuum_contributors=["H2--He"],
    #  wavelength_boundaries=[0.3, 0.8],
    # Model parameters
    # Planet parameters
    planet_radius=1 * cst.r_jup_mean,
    star_radius=Rs,
    transit_duration=tdur,
    reference_gravity=const.G.value * 0.85 * cst.m_jup / (cst.r_jup_mean) ** 2,
    reference_pressure=0.01,
    # Star, system, orbit
    # is_observed=True,  # return the flux observed at system_distance
    # is_around_star=True,  # if True, calculate a PHOENIX stellar spectrum and add it to the emission spectrum
    system_distance=10
    * cst.s_cst.light_year
    * 1e2,  # m to cm, used to scale the spectrum
    is_around_star=True,
    star_effective_temperature=5500,  # used to get the PHOENIX stellar spectrum model
    # Temperature profile parameters
    temperature_profile_mode="guillot",
    temperature=2500,
    intrinsic_temperature=200,
    guillot_temperature_profile_gamma=0.4,
    guillot_temperature_profile_infrared_mean_opacity_solar_metallicity=0.01,
    # Mass fractions
    # use_equilibrium_chemistry=True,
    # metallicity=3,  # times solar
    # co_ratio=0.1,
    # Mass fractions
    imposed_mass_fractions={  # these can also be arrays of the same size as pressures
        "Fe": 1e-3,
    },
    # Velocity parameters
    star_mass=Ms,
    orbit_semi_major_axis=a,
    orbital_period=period,
    orbital_inclination=observer_angle,
    # rest_frame_velocity_shift=-5e8,  # (cm.s-1) V_rest
    system_observer_radial_velocities=vp,
    filling_species={  # automatically fill the atmosphere with H2 and He, such that the sum of MMRs is equal to 1 and H2/He = 37/12
        "H2": 37,
        "He": 12,
    },  # Observation parameters
    rebinned_wavelengths=data_wavelengths,  # (cm) used for the rebinning, and also to set the wavelengths boundaries
    rebin_range_margin_power=4,  # used to set the wavelengths boundaries, adding a margin of ~1 Angstrom (1e-4 * ~1 µm)
    convolve_resolving_power=resolution,  # used for the convolution
    mid_transit_time=mid_transit_time,
    times=times,
    # Preparation parameters
    # tellurics_mask_threshold=0.8,  # mask the fitted transmittances if it is below this value
    # polynomial_fit_degree=2,  # degree of the polynomial fit
    uncertainties=data_uncertainties,
)

noise_matrix = np.random.default_rng(seed=2786).normal(loc=0, scale=data_uncertainties)

wl, transit_radii = spectral_model.calculate_spectrum(mode="emission")
fig, ax = plt.subplots()

wavelengths_rebinned, transit_radii_rebinned = spectral_model.calculate_spectrum(
    mode="emission",
    scale=True,  # scale the spectrum
    shift=True,  # Doppler-shift the spectrum to the planet's radial velocities relative to the observer
    # use_transit_light_loss=True,  # apply the effect of transit ingress and egress
    rebin=True,
)

flux_spectrum = np.concatenate((wl, transit_radii), axis=0)
np.save("Fe_spectrumwl", flux_spectrum)


# Broadening Kernel

veq = 2 * np.pi * Rpl / period  # cm/s
print(veq)


def broadening_kernel_no_vel(x):
    range = np.array([i for i in x if abs(i) <= 1])
    kernel = np.sqrt(1 - (range) ** 2)
    padding = abs(x.shape[0] - range.shape[0]) // 2
    kernel = np.pad(kernel, padding, "constant")
    return kernel


def broadening_kernel(x):
    range = np.array([i for i in x if abs(i) <= veq])
    kernel = np.sqrt(1 - (range / veq) ** 2)
    padding = abs(x.shape[0] - range.shape[0]) // 2
    kernel = np.pad(kernel, padding, "constant")
    return kernel


transit_radii_broadened = np.zeros(shape=(n_exposures, transit_radii_rebinned.shape[2]))
delta_v = const.c.value / resolution * 1e2  # cm
range_vel = 0.2
points_number = 21

x = (
    np.linspace(-range_vel, range_vel, points_number) * (points_number // 2) * delta_v
)  # always odd number of points (delta v)
print(x)
kernel = broadening_kernel(x)
kernel /= np.sum(kernel)

kernel_no_vel = broadening_kernel_no_vel(x)
kernel_no_vel /= np.sum(kernel_no_vel)

fig, ax = plt.subplots(1)
ax.plot(x * 1e-2, kernel, "o", markersize=2)
ax.set_xlabel("Radial Spin Velocity (m/s)")
ax.set_ylabel("Kernel")
ax.set_title(r"$\sqrt{1 - (\frac{x}{v_{eq}}})^2$ Kernel")
ax.annotate(
    f"Maximum Radial Spin = {veq * 1e-2:.2f}m/s",
    xy=(0.76, 0.9),
    xycoords="axes fraction",
    textcoords="offset points",
    size=10,
)

fig, ax = plt.subplots(2, figsize=(10, 6), sharex="all")

ax[0].plot(wavelengths_rebinned[0] * 1e4, transit_radii_rebinned[0][0])

for i in range(n_exposures):
    transit_radii_broadened[i] = scisig.fftconvolve(
        transit_radii_rebinned[0][i] - transit_radii_rebinned[0][i].mean(),
        kernel,
        "same",
    )

# exposure = [3, 75, -1]
exposure = [0]
for i in exposure:
    ax[1].plot(wavelengths_rebinned[0] * 1e4, transit_radii_broadened[i])
fig.supxlabel("Wavelength (microns)")
fig.supylabel(r"Flux (erg cm$^{-2}$ s$^{-1}$ cm$^{-1}$)")

# Plot over Orbital Phase
fig, ax = plt.subplots(figsize=(10, 6))
ax.pcolormesh(wavelengths_rebinned[0] * 1e4, orbital_phases, transit_radii_broadened)
ax.set_xlabel("Wavelength [microns]")
ax.set_ylabel("Orbital phase")

# Interpolation
S = transit_radii_broadened
vsys_range = np.linspace(100 * vp[0], 100 * vp[-1], 5000)
W = np.outer(1 - vsys_range * 1e-2 / const.c.value, wavelengths_rebinned[0])
interpolated_model = np.interp(W, wavelengths_rebinned[0], transit_radii_broadened[75])

CC = np.dot(S, interpolated_model.T)

fig, ax = plt.subplots()
# ax.imshow(CC, aspect ='auto')
ax.pcolormesh(vsys_range, orbital_phases, CC)
ax.set_xlabel(r"Velocity ($v_{\text{sys}}$)")
ax.set_ylabel(r"Orbital Phase")

# neale_wl = wl_rspace(0.6, 0.61, resolution)
# neale_delta_lambda = np.diff(neale_wl, 1)
#
# delta_lambda = np.diff(wavelengths_rebinned[0], 1)
#
# fig, ax = plt.subplots(3)
# ax[0].plot(delta_lambda, wavelengths_rebinned[0][:-1], "o", markersize=0.5)
# ax[1].plot(neale_delta_lambda, neale_wl[:-1], "o", markersize=0.5)
#
# fig, ax = plt.subplots(2)
# ax[0].plot(wavelengths_rebinned[0][:-1] / delta_lambda)
# ax[0].set_ylim(60000, 100000)
# ax[1].plot(neale_wl[:-1] / neale_delta_lambda)
# ax[1].set_ylim(60000, 100000)
print(veq)
plt.show()
