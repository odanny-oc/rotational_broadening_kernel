import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.plotlib import plot_radtrans_opacities
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.config import petitradtrans_config_parser
from petitRADTRANS.physics import temperature_profile_function_guillot_global
import scipy.signal as scisig

iron_atmosphere = Radtrans(
        pressures = np.logspace(-6 ,2, 120),
        # line_opacity_mode = "c-k",
        line_species = [
            'Fe'
            ],
        rayleigh_species = ['H2'],
        gas_continuum_contributors = ['H2--H2'],
        wavelength_boundaries = [0.3, 0.8],
)

planet_radius = 1.2 * const.R_jup.value

planet_mass = 0.89 * const.M_jup.value

star_radius = 0.85 * const.R_sun.value

surface_gravity = const.G.value * planet_mass/(planet_radius)**2
refpressure = 0.1

mean_molecular_mass = 2.3 * np.ones_like(iron_atmosphere.pressures)
mass_ratios = {
        'Fe': 1e-6 * np.ones_like(iron_atmosphere.pressures),
        'H2': 0.74 * np.ones_like(iron_atmosphere.pressures)
        }
infrared_mean_opacity = 0.1
gamma = 0.54

temperature_gradient = temperature_profile_function_guillot_global(
    pressures=iron_atmosphere.pressures * 1e-6,
    infrared_mean_opacity=infrared_mean_opacity,
    gamma=gamma,
    gravities= surface_gravity,
    intrinsic_temperature= 200,
    equilibrium_temperature= 2800
)

wavelengths, transit_radii, _ = iron_atmosphere.calculate_transit_radii(
    reference_pressure= refpressure,
    temperatures= temperature_gradient,
    mass_fractions= mass_ratios,
    mean_molar_masses= mean_molecular_mass,
    reference_gravity= surface_gravity,
    planet_radius= planet_radius
)

dflux = -(transit_radii/star_radius)**2

x = np.linspace(-100,100, 1000)
sigma = 0.01
gaussian = np.exp(-0.5 * (x/sigma)**2)
gaussian /= np.sum(gaussian)

dflux_g = scisig.fftconvolve(dflux, gaussian, 'same')

fig, ax = plt.subplots()

ax.plot(wavelengths, dflux_g)
ax.set_xlabel(r"Wavelength (microns)")
ax.set_ylabel(r"$\Delta$Flux $\left(\dfrac{R_p}{R_s}\right)$")

cc = np.correlate(wavelengths, wavelengths, 'same')
fig, ax = plt.subplots()
ax.plot(wavelengths - wavelengths.mean(), cc)

wl = np.linspace(3000, 4000, 3000)

signal = np.interp(wl, wavelengths*1e8, dflux_g)
ccsig = np.correlate(signal,signal, 'same')

fig, ax = plt.subplots()
ax.plot(wl, signal)
ax.set_xlabel(r'Wavelength ($\AA$)')
ax.set_ylabel(r'Flux')

cc = np.correlate(signal, signal, 'same')

fig, ax = plt.subplots()
ax.plot(wl - np.mean(wl), cc)
ax.set_xlabel(r'Shift ($\Delta\lambda \AA$)')
ax.set_ylabel(r'Cross Correlation Function (signal $\ast$ -signal)')

#cconv = scisig.fftconvolve(wavelengths, np.flipud(wavelengths), 'same')
#
#fig, ax = plt.subplots()
#ax.plot(wavelengths - wavelengths.mean(), cconv)
plt.show()








