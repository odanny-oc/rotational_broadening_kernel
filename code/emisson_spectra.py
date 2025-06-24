from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.plotlib import plot_radtrans_opacities
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.config import petitradtrans_config_parser
from petitRADTRANS.physics import temperature_profile_function_guillot_global
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sp

radtrans = Radtrans(
    pressures=np.logspace(-6, 2, 100),
    line_species=[
    "H2O",
    "CO2",
    "Fe"
    ],
    #rayleigh_species=['H2', 'He'],
    gas_continuum_contributors=['H2--He'],
    # cloud_species = ['Fe(s)_crystalline__DHS'],
    wavelength_boundaries=[0.3, 2],
    scattering_in_emission = True
)

temperatures_const = 2500 * np.ones_like(radtrans.pressures)
mass_fractions ={'Fe': 1e-6 * np.ones_like(radtrans.pressures),
          'H2': 0.74 * np.ones_like(radtrans.pressures),
          'He': 0.24 * np.ones_like(radtrans.pressures),
          'H2O': 1e-3 * np.ones_like(radtrans.pressures),
          'CO2': 1e-4 * np.ones_like(radtrans.pressures)}
 
mean_molecular_weight = 2.3 * np.ones_like(radtrans.pressures)

planet_radius = 1.2 * cst.r_jup_mean
planet_mass = 0.8 * cst.m_jup

star_radius =  0.8 * cst.r_sun
star_mass = 2 * cst.m_sun

grav = cst.G * planet_mass / (planet_radius)**2
refpressure = 0.01
infrared_mean_opacity = 0.1
gamma = 0.4

temperature_gradient = temperature_profile_function_guillot_global(
    pressures=radtrans.pressures * 1e-6,
    infrared_mean_opacity=infrared_mean_opacity,
    gamma=gamma,
    gravities=grav,
    intrinsic_temperature= 200,
    equilibrium_temperature=temperatures_const[0]
)

wl, flux, _ = radtrans.calculate_flux(
    temperatures = temperature_gradient,
    mass_fractions = mass_fractions,
    mean_molar_masses = mean_molecular_weight,
    reference_gravity = grav
)

blackbody, flux_bb, _ = radtrans.calculate_flux(
    temperatures = 2800 * np.ones_like(temperature_gradient),
    mass_fractions = mass_fractions,
    mean_molar_masses = mean_molecular_weight,
    reference_gravity = grav
)


fig, ax = plt.subplots()
ax.plot(wl * 1e8, flux)
ax.plot(blackbody * 1e8, flux_bb)
ax.set_xlabel(r'Wavelength ($\AA$')
ax.set_ylabel('Flux')


fig, ax = plt.subplots()
ax.plot(temperature_gradient, radtrans.pressures * 1e-6)
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Pressure (bar)')
ax.set_yscale('log')
plt.show()