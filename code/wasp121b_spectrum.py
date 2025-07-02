import numpy as np
import matplotlib.pyplot as plt
import os
from astropy import constants as const
from petitRADTRANS import physical_constants as cst, planet
from petitRADTRANS.physics import temperature_profile_function_guillot_global
from petitRADTRANS.spectral_model import SpectralModel
from petitRADTRANS.math import resolving_space
from petitRADTRANS.planet import Planet
import SysRem

resolution = 200000

def black_body(wl, T):
    black_body = (
        (2 * const.h.value * const.c.value**2)
        / (wl**5)
        * (np.exp((const.h.value * const.c.value) / (wl * const.k_B.value * T)) - 1)
        ** (-1)
    )
    return black_body


# Planet and Star Parameters (WASP121b)

Rpl = 1.753 * cst.r_jup
Rs = 1.458 * cst.r_sun
Mpl = 1.157 * cst.m_jup
Ms = 1.358 * cst.m_sun
a = 0.02596 * cst.au  # au to cm
period = 1.27492504 * (24 * 60**2)  # days to seconds
eccen = 0
observer_angle = 90  # degrees
Kp = 2 * np.pi / (period) * a * np.sin(observer_angle) / (np.sqrt(1 - eccen**2))
b = 0.1

ttot = period / np.pi * np.arcsin(Rs / a * np.sqrt((1 + Rpl / Rs) ** 2 - b**2))
tfull = period / np.pi * np.arcsin(Rs / a * np.sqrt((1 - Rpl / Rs) ** 2 - b**2))

data_wavelengths = (
    resolving_space(1.5, 2.5, resolution) * 1e-4
)  # (cm) generate wavelengths at a constant resolving power

spectral_model = SpectralModel(
    # Radtrans parameters
    pressures=np.logspace(-12, 2, 100),
    line_opacity_mode="lbl",
    # line_by_line_opacity_sampling= 5,
    line_species=["Fe", "CO", "H2O"],
    # Planet parameters
    planet_radius=Rpl,
    reference_gravity=const.G.value * Mpl / (Rpl**2),
    reference_pressure=1e-2,
    # Temperature profile parameters
    temperature_profile_mode="guillot",
    temperature=2000,
    intrinsic_temperature=0,
    guillot_temperature_profile_gamma=0.4,
    guillot_temperature_profile_infrared_mean_opacity_solar_metallicity=0.01,
    # Mass fractions
    imposed_mass_fractions={  # these can also be arrays of the same size as pressures
        "Fe": 1e-6,
        "CO": 1e-3,
        "H2O": 1e-2,
    },
    filling_species={  # automatically fill the atmosphere with H2 and He, such that the sum of MMRs is equal to 1 and H2/He = 37/12
        "H2": 37,
        "He": 12,
    },  # Observation parameters
    rebinned_wavelengths=data_wavelengths,  # (cm) used for the rebinning, and also to set the wavelengths boundaries
)

wl, flux = spectral_model.calculate_spectrum(mode="emission")
wl *= 1e4

star_spectrum = (
    black_body(wl * 1e-6, 6430)[0] * 1e7 * 1e-6
)  # Joules to ergs and m^-3 to cm^-3
fig, ax = plt.subplots()
ax.plot(wl[0], star_spectrum)

rel_flux = np.pi * Rpl**2 * flux[0] / (np.pi * Rs**2 * np.pi * star_spectrum)

fig, ax = plt.subplots()
ax.plot(wl[0], flux[0])
ax.set_xlabel(r"Wavelength ($\mu$m)")
ax.set_ylabel(r"Flux (ergs s$^{-1}$ cm$^{-2}$ cm$^{-1}$")

fig, ax = plt.subplots()
ax.plot(wl[0], rel_flux)
ax.set_xlabel(r"Wavelength ($\mu$m)")
ax.set_ylabel(r"Flux ($\Delta$F)")
plt.show()
np.savez(
    os.path.join("/home/danny/exoplanet_atmospheres/code", "wasp121b.npz"),
    wl=wl[0],
    flux= rel_flux,
    radius_planet=Rpl * 1e-5,
    radius_star=Rs * 1e-5,
    mass_planet=Mpl * 1e-3,
    mass_star=Ms * 1e-3,
    impact_parameter=b,
    semi_major_axis=a * 1e-5,
    radial_velocity_semi_amplitude=Kp * 1e-5,
    observer_angle=observer_angle,
    period=period,
    total_transit_time=ttot,
    full_transit_time=tfull,
)
