import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as scisig
import os, subprocess
from astropy import constants as const
from petitRADTRANS import physical_constants as cst, planet
from petitRADTRANS.physics import temperature_profile_function_guillot_global
from petitRADTRANS.spectral_model import SpectralModel
from petitRADTRANS.math import resolving_space
from petitRADTRANS.planet import Planet
from petitRADTRANS.config import petitradtrans_config_parser

import SysRem

mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Computer Modern Serif",
    "font.size": 35,
    "figure.figsize": [20, 15],
    "axes.facecolor": (1.0 ,1.0, 1.0, 1.0),
    "savefig.facecolor": (1.0 ,1.0, 1.0, 1.0)
})

home_path = os.environ["HOME"]
local_path = home_path + "/exoplanet_atmospheres/code"
local_images = home_path + "/exoplanet_atmospheres/images"

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

tol = 0.02
wl = wl_day[0]

fig, ax = plt.subplots()
ax.plot(wl_day[0], rel_flux_day)
fig.supxlabel(r"Wavelength ($\mu$m)")
fig.supylabel(r"Flux ($\Delta$F)")
ax.set_xlim(wl[0] + tol, wl[-1] - tol)
plt.savefig(os.path.join(local_images, "H2O_spectrum.png"))
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
from functions import Kp_vsys_Plotter

wasp121_post_data = np.load(
    local_path + "/crires_posteclipse_WASP121_2021-12-15_processed.npz"
)
wavelength_grid = wasp121_post_data["W"][-1] * 1e-4  # resolution ~ 300000


flux_grid_day = np.interp(wavelength_grid, wl, rel_flux_day)

flux_grid_night = np.interp(wavelength_grid, wl, rel_flux_night)

op = np.linspace(0.334, 0.425, 1001)
op = np.linspace(0.539, 0.626, 1001)
op = np.linspace(0.0, 1.0, 1001)

# CO_spec = Time_Dependent_Spectrum(
#     wl=wl,
#     flux=rel_flux_night,
#     op=op,
#     kp=100,
#     wl_grid=wavelength_grid,
# )

H2O_spec = Time_Dependent_Spectrum(
    wl=wl,
    flux=rel_flux_day,
    op=op,
    kp=100,
)

vsys = np.linspace(-150, 150, 1001)

CC_H2O = Cross_Correlator(
    wl=wl, flux=rel_flux_day, vsys=vsys * 1000, spectrum=H2O_spec
)

# CC_CO = Cross_Correlator(
#     wl=wavelength_grid, flux=flux_grid_day, vsys=vsys * 1000, spectrum=CO_spec
# )

tol2 = 0.0236

fig, ax = plt.subplots()
ax.plot(wl, rel_flux_day)
ax.plot(wl, H2O_spec[20])
ax.plot(wl, H2O_spec[50])
fig.supxlabel(r"Wavelength ($\mu$m)")
fig.supylabel(r"Flux ($\Delta$F)")
ax.set_xlim(wl[0] + tol2, wl[-1] - tol2)
plt.savefig(os.path.join(local_images, "doppler_shift.png"))

fig, ax = plt.subplots()
ax.pcolormesh(vsys, op, CC_H2O)
fig.supxlabel("System Velocity (km/s)")
fig.supylabel(r"Orbital Phase ($\phi$)")
plt.savefig(os.path.join(local_images, "cc_h2o.png"))

fig, ax = plt.subplots()
ax.pcolormesh(vsys, op, CC_H2O)
ax.plot(100 * np.sin(2*np.pi*op), op, color='r', ls = '--', lw = 1, label = r'$K_p\sin(2\pi\phi)$ with $K_p = 100$km/s ')
fig.supxlabel("System Velocity (km/s)")
fig.supylabel(r"Orbital Phase ($\phi$)")
ax.legend()
plt.savefig(os.path.join(local_images, "cc_h2o_fit.png"))


fig, ax = plt.subplots()
ax.pcolormesh(wl, op, H2O_spec)
fig.supxlabel(r"Wavelength ($\mu$m)")
fig.supylabel(r"Orbital Phase ($\phi$)")
plt.savefig(os.path.join(local_images, "h2o_spec.png"))

def gaussian(x, sigma):
    return np.exp(-0.5 * (x / sigma) ** 2)


x = np.linspace(-10, 10, 101)

fig, ax = plt.subplots()
g= gaussian(x, 2)
ax.plot(x, g)
plt.savefig(os.path.join(local_images, "gaussian.png"))

rel_flux_day-=np.mean(rel_flux_day)

fig, ax = plt.subplots()
convolved_spectrum = scisig.fftconvolve(rel_flux_day, g, 'same')
ax.plot(wl, convolved_spectrum)
ax.set_xlim(wl[0] + tol, wl[-1] - tol)
fig.supxlabel(r"Wavelength ($\mu$m)")
fig.supylabel(r"Flux ($\Delta$F)")
plt.savefig(os.path.join(local_images, "h2o_spectrum_convolved.png"))

real_op = op[350:450]

H2O_spec = Time_Dependent_Spectrum(
    wl=wl,
    flux=rel_flux_day,
    op=real_op,
    kp=100,
)

CC_H2O = Cross_Correlator(
    wl=wl, flux=rel_flux_day, vsys=vsys * 1000, spectrum=H2O_spec
)

fig, ax = plt.subplots()
ax.pcolormesh(vsys, real_op, CC_H2O)
fig.supxlabel("System Velocity (km/s)")
fig.supylabel(r"Orbital Phase ($\phi$)")
plt.savefig(os.path.join(local_images, "cc_h2o_real.png"))

K = np.linspace(75, 125, 1001)
vsys_kp = np.linspace(-20, 20, 1001)

kp_map, _  = Kp_vsys_Plotter(K, vsys, real_op, CC_H2O, vsys_kp)
kp_map_pre = kp_map

fig, ax = plt.subplots()
ax.pcolormesh(vsys, K, kp_map)
fig.supxlabel("System Velocity (km/s)")
fig.supylabel(r"Radial Velocity Semi-Amplitude $K_p$ (km/s)")
plt.savefig(os.path.join(local_images, "kp_h2o.png"))

real_op = op[550:650]

H2O_spec = Time_Dependent_Spectrum(
    wl=wl,
    flux=rel_flux_day,
    op=real_op,
    kp=100,
)

CC_H2O = Cross_Correlator(
    wl=wl, flux=rel_flux_day, vsys=vsys * 1000, spectrum=H2O_spec
)

fig, ax = plt.subplots()
ax.pcolormesh(vsys, real_op, CC_H2O)
fig.supxlabel("System Velocity (km/s)")
fig.supylabel(r"Orbital Phase ($\phi$)")
plt.savefig(os.path.join(local_images, "cc_h2o_real_post.png"))

K = np.linspace(75, 125, 1001)
vsys_kp = np.linspace(-20, 20, 1001)

kp_map, _  = Kp_vsys_Plotter(K, vsys, real_op, CC_H2O, vsys_kp)

fig, ax = plt.subplots()
ax.pcolormesh(vsys, K, kp_map)
fig.supxlabel("System Velocity (km/s)")
fig.supylabel(r"Radial Velocity Semi-Amplitude $K_p$ (km/s)")
plt.savefig(os.path.join(local_images, "kp_h2o_post.png"))

fig, ax = plt.subplots()
ax.pcolormesh(vsys, K, kp_map + kp_map_pre)
fig.supxlabel("System Velocity (km/s)")
fig.supylabel(r"Radial Velocity Semi-Amplitude $K_p$ (km/s)")
plt.savefig(os.path.join(local_images, "kp_h2o_tot.png"))
