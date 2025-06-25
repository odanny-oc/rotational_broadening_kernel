import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.plotlib import plot_radtrans_opacities
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.config import petitradtrans_config_parser
from petitRADTRANS.physics import temperature_profile_function_guillot_global
import scipy.signal as scisig

wl,sigma = np.load('/home/danny/exoplanet_atmospheres/notebooks/x_FeI_2500K_m2.npy')
chi_FeI = 1e-6 #define abundance of FeI in volume mixing ratio

Pcloud = 10 #define altitude of clouds, in bar
T = 2500 #atmospheric temperature in K, this is only used to calculate the scale height
  #ie can be adjusted separately to the opacity temperature

Rstar = 1.756*const.R_sun.value #stellar radius in m
Mpl = 0.894*const.M_jup.value #planet mass in kg
Rpl = 0.10852*const.R_sun.value #planet radius in m
mmw = 2.3 *const.m_p.value #mean molecular weight kg
gamma = 0.54

#derive quantities
g = const.G.value * Mpl / Rpl**2 #surface gravity of planet
Hs = const.k_B.value * T / (2.3*const.m_p.value * g) #calculate atmospheric scale height
P0 = 100 * 100000. #reference pressure, bar to Pa at R0
R0 = Rpl #reference radius

#calculate constant of model
K = R0 + Hs * (gamma + np.log(P0 / g) - np.log(mmw) + 0.5*np.log(2.*np.pi*R0/Hs) )

#calcualte the transmission spectrum
r_species = K + Hs * np.log(chi_FeI * sigma) #this is in units of m

#calculate radius of cloud deck, which forms the continuum
r_continuum = Hs * np.log(P0/Pcloud/100000.) + R0

#truncate the transmission spectrum at the altitude of the cloud deck (as atmosphere is completely opaque lower down)
r = np.maximum(r_species, 0)

#finally make a plot of the transmission spectrum
f,a = plt.subplots(1)
# a.plot(wl,r_species,'0.5',alpha=0.5)
# a.plot(wl,r_continuum * np.ones(wl.size),'b-')
a.plot(wl,r,'r-')
a.set_xlabel(r'$\lambda (\AA)$')
a.set_ylabel(r'effective planetary radius (m)')
# a.grid()

iron_atmosphere = Radtrans(
    pressures= np.logspace(-8, 0, 120),
    line_opacity_mode = 'lbl',
    line_species=[
        'Fe'
    ],
    # rayleigh_species= ['H2'],
    gas_continuum_contributors=['H2--H2'],
    wavelength_boundaries= [0.2, 0.8] #microns
)

mass_fractions ={
    'Fe': 1e-6* np.ones_like(iron_atmosphere.pressures),
    'H2': 0.74 * np.ones_like(iron_atmosphere.pressures)
}

temperature_grad = temperature_profile_function_guillot_global(
    pressures = iron_atmosphere.pressures * 1e-6,
    infrared_mean_opacity= 0.01,
    gamma = 0.4,
    gravities= g,
    intrinsic_temperature= 200,
    equilibrium_temperature= 2500
)

temperatures = 1000 * np.ones_like(iron_atmosphere.pressures)

wavelength, transit_radius, opacities = iron_atmosphere.calculate_transit_radii(
    temperatures = temperature_grad,
    reference_gravity= g,
    reference_pressure= 0.1,
    # opaque_cloud_top_pressure= 100,
    mass_fractions= mass_fractions,
    mean_molar_masses= 2.3 * np.ones_like(iron_atmosphere.pressures),
    planet_radius = Rpl,
    return_opacities= True
)

f, a = plt.subplots()
a.plot(wavelength * 1e8, transit_radius)
a.set_xlabel(r'Wavelength ($\AA$)')
a.set_ylabel(r'Transit Radius (m)')

# print(np.logspace(-2,4, 7)[0])
plt.show()
