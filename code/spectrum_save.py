import numpy as np
import matplotlib.pyplot as plt
import os
from astropy import constants as const
from petitRADTRANS import physical_constants as cst, planet
from petitRADTRANS.physics import temperature_profile_function_guillot_global
from petitRADTRANS.spectral_model import SpectralModel
from petitRADTRANS.math import resolving_space
from petitRADTRANS.planet import Planet

planet = Planet.get("HD 189733 b")
resolution = 80000


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
    planet_radius=Rpl,
    star_radius=Rs,
    star_mass=Ms,
    transit_duration=tdur,
    reference_gravity=const.G.value * Mp / (Rpl**2),
    reference_pressure=0.01,
    # Star, system, orbit
    is_observed=True,  # return the flux observed at system_distance
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

wl, flux = spectral_model.calculate_spectrum(mode="emission")
wl *= 1e4
flux -= np.mean(flux)

fig, ax = plt.subplots()
ax.plot(wl[0], flux[0])
ax.set_xlabel(r"Wavelength ($\mu$m)")
ax.set_ylabel(r"Flux ($\Delta$F)")
plt.show()
np.savez(
    os.path.join("/home/danny/exoplanet_atmospheres/code", "Fe_spectrum.npz"),
    wl=wl[0],
    flux=flux[0],
)
