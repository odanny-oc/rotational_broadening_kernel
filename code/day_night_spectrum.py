import numpy as np
import matplotlib.pyplot as plt
import os
from astropy import constants as const
from petitRADTRANS import physical_constants as cst, planet
from petitRADTRANS.physics import temperature_profile_function_guillot_global
from petitRADTRANS.spectral_model import SpectralModel
from petitRADTRANS.math import resolving_space
from petitRADTRANS.planet import Planet
from petitRADTRANS.config import petitradtrans_config_parser

import SysRem

home_path = os.environ["HOME"]
local_path = home_path + "/exoplanet_atmospheres/code"

resolution = 400000
wasp121_post_data = np.load(
    local_path + "/crires_posteclipse_WASP121_2021-12-15_processed.npz"
)
wavelength_grid = wasp121_post_data["W"][-1] * 1e-4  # resolution ~ 300000

print(wavelength_grid)


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
    resolving_space(2.45, 2.5, resolution) * 1e-4
)  # (cm) generate wavelengths at a constant resolving power

spectral_model_night = SpectralModel(
    # Radtrans parameters
    pressures=np.logspace(-12, 2, 100),
    line_opacity_mode="lbl",
    # line_by_line_opacity_sampling= 5,
    line_species=["CO"],
    # Planet parameters
    planet_radius=Rpl,
    reference_gravity=const.G.value * Mpl / (Rpl**2),
    reference_pressure=1e-2,
    # Temperature profile parameters
    temperature_profile_mode="guillot",
    temperature=1700,
    intrinsic_temperature=0,
    guillot_temperature_profile_gamma=0.4,
    guillot_temperature_profile_infrared_mean_opacity_solar_metallicity=0.01,
    # Mass fractions
    imposed_mass_fractions={"CO": 1e-3, "H2O": 1e-6},
    filling_species={  # automatically fill the atmosphere with H2 and He, such that the sum of MMRs is equal to 1 and H2/He = 37/12
        "H2": 37,
        "He": 12,
    },  # Observation parameters
    rebinned_wavelengths=data_wavelengths,  # (cm) used for the rebinning, and also to set the wavelengths boundaries
)


spectral_model_day = SpectralModel(
    # Radtrans parameters
    pressures=np.logspace(-12, 2, 100),
    line_opacity_mode="lbl",
    # line_by_line_opacity_sampling= 5,
    line_species=["H2O"],
    # Planet parameters
    planet_radius=Rpl,
    reference_gravity=const.G.value * Mpl / (Rpl**2),
    reference_pressure=1e-2,
    # Temperature profile parameters
    temperature_profile_mode="guillot",
    temperature=2400,
    intrinsic_temperature=0,
    guillot_temperature_profile_gamma=0.4,
    guillot_temperature_profile_infrared_mean_opacity_solar_metallicity=0.01,
    # Mass fractions
    imposed_mass_fractions={"H2O": 1e-3, "CO": 1e-6},
    filling_species={  # automatically fill the atmosphere with H2 and He, such that the sum of MMRs is equal to 1 and H2/He = 37/12
        "H2": 37,
        "He": 12,
    },  # Observation parameters
    rebinned_wavelengths=data_wavelengths,  # (cm) used for the rebinning, and also to set the wavelengths boundaries
)

wl_day, flux_day = spectral_model_day.calculate_spectrum(mode="emission")
wl_day *= 1e4

wl_night, flux_night = spectral_model_night.calculate_spectrum(mode="emission")
wl_night *= 1e4

star_spectrum = (
    black_body(wl_day * 1e-6, 6430)[0] * 1e7 * 1e-6
)  # Joules to ergs and m^-3 to cm^-3

rel_flux_day = np.pi * Rpl**2 * flux_day[0] / (np.pi * Rs**2 * np.pi * star_spectrum)
rel_flux_night = (
    np.pi * Rpl**2 * flux_night[0] / (np.pi * Rs**2 * np.pi * star_spectrum)
)

fig, ax = plt.subplots(2)
ax[0].plot(wl_day[0], rel_flux_day)
ax[1].plot(wl_night[0], rel_flux_night)
fig.supxlabel(r"Wavelength ($\mu$m)")
fig.supylabel(r"Flux ($\Delta$F)")
np.savez(
    os.path.join(local_path, "day_night_spectrum.npz"),
    wl_day=wl_day[0],
    flux_day=rel_flux_day,
    wl_night=wl_night[0],
    flux_night=rel_flux_night,
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

from functions import Cross_Correlator
from functions import Time_Dependent_Spectrum

wasp121_post_data = np.load(
    local_path + "/crires_posteclipse_WASP121_2021-12-15_processed.npz"
)
wavelength_grid = wasp121_post_data["W"][-1] * 1e-4  # resolution ~ 300000

wl = wl_day[0]

flux_grid_day = np.interp(wavelength_grid, wl, rel_flux_day)

flux_grid_night = np.interp(wavelength_grid, wl, rel_flux_night)

op = np.linspace(0.334, 0.425, 1001)
op = np.linspace(0.539, 0.626, 1001)

CO_spec = Time_Dependent_Spectrum(
    wl=wl,
    flux=rel_flux_night,
    op=op,
    kp=100,
    wl_grid=wavelength_grid,
)

H2O_spec = Time_Dependent_Spectrum(
    wl=wl,
    flux=rel_flux_day,
    op=op,
    kp=100,
    wl_grid=wavelength_grid,
)

vsys = np.linspace(-200, 200, 1001)

CC_H2O = Cross_Correlator(
    wl=wavelength_grid, flux=flux_grid_night, vsys=vsys * 1000, spectrum=H2O_spec
)

CC_CO = Cross_Correlator(
    wl=wavelength_grid, flux=flux_grid_day, vsys=vsys * 1000, spectrum=CO_spec
)

fig, ax = plt.subplots(2, sharex="all", sharey="all")
vp = 100 * np.sin(2 * np.pi * op)
ax[0].pcolormesh(vsys, op, CC_H2O)
ax[1].pcolormesh(vsys, op, CC_CO)

fig, ax = plt.subplots(2, sharex="all", sharey="all")
ax[0].pcolormesh(wavelength_grid, op, H2O_spec)
ax[1].pcolormesh(wavelength_grid, op, CO_spec)

plt.show()
