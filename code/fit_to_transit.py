import numpy as np
import matplotlib.pyplot as plt
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


# Planet and Star Parameters

Rpl = cst.r_jup_mean
Rs = cst.r_sun
Mp = 0.85 * cst.m_jup
Ms = 1.1 * cst.m_sun
a = 0.045 * cst.au * 1.495979e13  # au to cm
period = 1.5 * (24 * 60**2)  # days to seconds
eccen = 0.1
observer_angle = 90  # degrees
Kp = 2 * np.pi / period * a * np.sin(observer_angle) / (np.sqrt(1 - eccen**2))
b = 0.01

tdur = period / np.pi * np.arcsin(Rs / a * np.sqrt((1 - Rpl / Rs) ** 2 - b**2))

n_exposures = 150

data_wavelengths = (
    resolving_space(0.61, 0.615, 9e5) * 1e-4
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
    star_effective_temperature=5500,  # used to get the PHOENIX stellar spectrum model
    # Temperature profile parameters
    temperature_profile_mode="isothermal",
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
    convolve_resolving_power=8e4,  # used for the convolution
    mid_transit_time=mid_transit_time,
    times=times,
    # Preparation parameters
    tellurics_mask_threshold=0.8,  # mask the fitted transmittances if it is below this value
    polynomial_fit_degree=2,  # degree of the polynomial fit
    uncertainties=data_uncertainties,
)


wavelengths_rebinned, transit_radii_rebinned = spectral_model.calculate_spectrum(
    mode="transmission",
    scale=True,  # scale the spectrum
    shift=True,  # Doppler-shift the spectrum to the planet's radial velocities relative to the observer
    # use_transit_light_loss=True,  # apply the effect of transit ingress and egress
    convolve=True,  # convolve to the instrument's resolving power
    rebin=True,
)

fig, ax = plt.subplots(2, figsize=(10, 6))

ax[0].pcolormesh(
    wavelengths_rebinned[0] * 1e4, orbital_phases, transit_radii_rebinned[0]
)
ax[0].set_xlabel("Wavelength [microns]")
ax[0].set_ylabel("Orbital phase")

ax[1].plot(wavelengths_rebinned[0], transit_radii_rebinned[0][0])

# 1-x2 Broadening Kerenl
wavelengths, transit_radii = spectral_model.calculate_spectrum(
    mode="transmission",
    scale=True,  # scale the spectrum
    shift=True,  # Doppler-shift the spectrum to the planet's radial velocities relative to the observer
    # use_transit_light_loss=True,  # apply the effect of transit ingress and egress
    # convolve=True,  # convolve to the instrument's resolving power
    rebin=True,
)
wavelengths = wavelengths[0]
transit_radii = transit_radii[0]


def broadening_kernel(vp, x):
    kernel = []
    count = 0
    for i in x:
        if abs(i) <= 1:
            kernel.append(vp * np.sqrt(1 - i**2))
            count += 1
    zeros = (x.size - count) // 2
    it = 0
    while it < zeros:
        kernel.append(0)
        kernel.insert(0, 0)
        it += 1
    return kernel


x = np.linspace(-5, 5, 200)

fig, ax = plt.subplots(2, sharex="all")
ax[0].plot(x, broadening_kernel(1, x) / np.sum(broadening_kernel(1, x)))
ax[0].set_title("Broadening Kernel")

transit_radii_broadened = []

for i in range(0, n_exposures):
    # kernel = broadening_kernel(spectral_model.model_parameters['relative_velocities'][i], x)
    velocity = spectral_model.model_parameters["relative_velocities"][i]
    kernel = broadening_kernel(velocity, x)
    kernel /= np.sum(kernel)
    cc = scisig.fftconvolve(transit_radii[i], kernel, "same")
    cc[:100] = transit_radii[i][:100]
    cc[-100:] = transit_radii[i][-100:]
    transit_radii_broadened.append(cc)

sigma = 0.2
x_gaussian = np.linspace(-50, 50, 101)
gaussian = np.exp(-0.5 * (x_gaussian / sigma) ** 2)
gaussian /= gaussian.sum()

ax[1].plot(x_gaussian, gaussian)
ax[1].set_title("Gaussian Kernel")

crop = 400

transit_radii_gaussian = [
    scisig.fftconvolve(transit_radii[i], gaussian, "same") for i in range(n_exposures)
]
# for i in range(n_exposures):
# transit_radii_gaussian[i][crop:-crop] = transit_radii[i][crop:-crop]
exposure = [3, 10, -1]
# fig, ax = plt.subplots()
# for i in exposure:
#    ax.plot(wavelengths, transit_radii[i], label = f'Shifted spectra {spectral_model.model_parameters['relative_velocities'][i] * 1e-5:.2f}' + r'km s$^{-1}$')
# ax.legend()

fig, ax = plt.subplots(2, sharex="all", sharey="all")
for i in exposure:
    ax[0].plot(
        wavelengths,
        transit_radii_broadened[i],
        label=f"Shifted broadened spectra {spectral_model.model_parameters['relative_velocities'][i] * 1e-5:.2f}"
        + r"km s$^{-1}$",
    )
    ax[1].plot(wavelengths, transit_radii_gaussian[i])
ax[0].legend()

transit_radii_broadened = np.array(transit_radii_broadened)

# fig, ax = plt.subplots()
# ax.pcolormesh(wavelengths, orbital_phases, transit_radii_broadened)
# ax.set_xlabel("Wavelength (microns)")
# ax.set_ylabel("Orbital Phase")
# ax.set_title("Orbital Phase with Wavelength against Relative Flux (Broadened)")


# wavelengths, transit_radii = spectral_model.calculate_spectrum(
#        mode ='transmission',
#        scale = True,  # scale the spectrum
#        shift=True,  # Doppler-shift the spectrum to the planet's radial velocities relative to the observer
#        use_transit_light_loss=True,  # apply the effect of transit ingress and egress
#        #convolve=True,  # convolve to the instrument's resolving power
#        )
# wavelength_index = int(transit_radii.shape[-1]/2)
# flux = []
# for i in range(0,19):
#    flux.append(transit_radii[i][wavelength_index])
# fig, ax = plt.subplots()
# ax.plot(orbital_phases, flux)

# Cross Correlate
# transit_radii_rebinned has shape of (1, 20, ~10000) which are the 20 arrays for each point in the orbital phase containing the relative flux data

# Define what section of the total spectrum you want to analyise
start = wavelengths.shape[0] // 2
end = start + int((wavelengths.shape[0] * 0.9) // 2)
array_length = end - start
S = np.zeros(shape=(n_exposures, array_length))
for i in range(n_exposures):
    S[i] = (
        transit_radii_broadened[i][start:end]
        - transit_radii_broadened[i][start:end].mean()
    )

# Interpolate desired section over larger range of radial velocities using entire data set
vsys = np.linspace(10 * vp[0], 10 * vp[-1], 10000)  # (cm.s-1)
wl = np.linspace(wavelengths[start], wavelengths[end], S.shape[1])
W = np.outer((1 - vsys * 1e-2 / const.c.value), wl)

shifted_values = np.interp(
    W, wavelengths, transit_radii_broadened[50] - transit_radii_broadened[50].mean()
)

CC = np.dot(S, shifted_values.T)

fig, ax = plt.subplots(figsize=(10, 30))
ax.pcolormesh(vsys, orbital_phases, CC, cmap="gist_rainbow")
# ax.imshow(CC, aspect = 'auto')
ax.title.set_text("Cross Correlation Function (np.dot)")
ax.set_xlabel("Systemic Velocity")
ax.set_ylabel("Orbital Phase")

fig, ax = plt.subplots(2, figsize=(10, 30))
ax[0].imshow(shifted_values, aspect="auto")
ax[0].title.set_text("Interpolated Fe Spectrum over twice the Velocity Range")
ax[1].imshow(S, aspect="auto")
plt.ylabel("Velocity (index)")
plt.xlabel("Wavelength (Index)")
ax[1].title.set_text("petitRADTRANS Fe Spectrum")
ax[1].invert_yaxis()
ax[0].invert_yaxis()

test_spectrum = S[140]
test_velocity = spectral_model.model_parameters["relative_velocities"][140]

test_wavelengths = (1 - (test_velocity * 1e-2 / const.c.value)) * wl

correlated_model = np.interp(
    test_wavelengths,
    wavelengths,
    transit_radii_broadened[120] - transit_radii_broadened[120].mean(),
)

cc = np.correlate(correlated_model, test_spectrum, "same")
fig, ax = plt.subplots(3)
ax[0].plot(test_wavelengths, correlated_model)
ax[1].plot(test_wavelengths, test_spectrum)
ax[2].plot(test_wavelengths - test_wavelengths.mean(), cc)


plt.show()
