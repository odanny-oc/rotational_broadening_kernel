import numpy as np
import SysRem
import matplotlib.pyplot as plt
from astropy import constants as const
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.plotlib import plot_radtrans_opacities
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.config import petitradtrans_config_parser
from petitRADTRANS.physics import temperature_profile_function_guillot_global
import scipy.signal as scisig

iron_atmosphere = Radtrans(
    pressures=np.logspace(-6, 2, 120),
    line_opacity_mode="lbl",
    line_species=["Fe"],
    rayleigh_species=["H2"],
    gas_continuum_contributors=["H2--H2"],
    wavelength_boundaries=[0.3, 0.31],
)

planet_radius = 1.2 * const.R_jup.value

planet_mass = 0.89 * const.M_jup.value

star_radius = 0.85 * const.R_sun.value

surface_gravity = const.G.value * planet_mass / (planet_radius) ** 2
refpressure = 0.1

mean_molecular_mass = 2.3 * np.ones_like(iron_atmosphere.pressures)
mass_ratios = {
    "Fe": 1e-6 * np.ones_like(iron_atmosphere.pressures),
    "H2": 0.74 * np.ones_like(iron_atmosphere.pressures),
}
infrared_mean_opacity = 0.1
gamma = 0.54

temperature_gradient = temperature_profile_function_guillot_global(
    pressures=iron_atmosphere.pressures * 1e-6,
    infrared_mean_opacity=infrared_mean_opacity,
    gamma=gamma,
    gravities=surface_gravity,
    intrinsic_temperature=200,
    equilibrium_temperature=2800,
)

wavelengths, transit_radii, _ = iron_atmosphere.calculate_transit_radii(
    reference_pressure=refpressure,
    temperatures=temperature_gradient,
    mass_fractions=mass_ratios,
    mean_molar_masses=mean_molecular_mass,
    reference_gravity=surface_gravity,
    planet_radius=planet_radius,
)

dflux = -((transit_radii / star_radius) ** 2)
dflux -= np.mean(dflux)

x = np.linspace(-100, 100, 51)
sigma = 0.01
gaussian = np.exp(-0.5 * (x / sigma) ** 2)
gaussian /= np.sum(gaussian)

dflux_g = scisig.fftconvolve(dflux, gaussian, "same")

orbital_phase = np.linspace(-0.04, 0.04, 1000)

Kp = 150

vp = Kp * np.sin(2 * np.pi * orbital_phase)
Wl = np.outer(1 - vp / (const.c.value * 1e-3), wavelengths)

shifted_template = np.interp(Wl, wavelengths, dflux_g)

fig, ax = plt.subplots()

ax.plot(wavelengths, dflux_g)
# ax.plot(wavelengths, shifted_template[100])
# ax.plot(wavelengths, shifted_template[15])
ax.set_xlabel(r"Wavelength (microns)")
ax.set_ylabel(r"$\Delta$Flux $\left(\dfrac{R_p}{R_s}\right)$")
fig, ax = plt.subplots()
ax.imshow(shifted_template, aspect="auto")
ax.invert_yaxis()


vsys = np.linspace(-200, 200, 400)
W = np.outer(1 - vsys / (const.c.value * 1e-3), wavelengths)

model_templates = np.interp(W, wavelengths, dflux_g)
CC = np.dot(shifted_template, model_templates.T)
fig, ax = plt.subplots()
ax.pcolormesh(vsys, vp, CC)

K = np.linspace(0, 300, 500)

sum_map = np.empty((K.size, vsys.size))
CC_array = np.empty(CC.shape)

for i, kp in enumerate(K):
    vel = kp * np.sin(2 * np.pi * orbital_phase)
    for j in range(vel.size):
        CC_array[j] = np.interp(vsys + vel[j], vsys, CC[j])
    sum_map[i] = np.sum(CC_array, axis=0)


fig, ax = plt.subplots()
ax.pcolormesh(vsys, K, sum_map)
ax.set_ylabel(r"$K_p$")
ax.set_xlabel(r"$v_{\text{sys}}$")
ax.axhline(Kp, ls="--", lw=0.5)
ax.axvline(0, ls="--", lw=0.5)

max_indices = np.where(sum_map == np.max(sum_map))
print(Kp - K[max_indices[0][0]])
# test_array = np.array([[1, 2, 3], [4, 5, 6]])
# print(np.where(test_array == np.max(test_array)))

plt.show()
