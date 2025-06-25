import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const

wl,sigma = np.load('/home/danny/exoplanet_atmospheres/notebooks/x_FeI_2500K_m2.npy')
chi_FeI = 1e-6 #define abundance of FeI in volume mixing ratio

Pcloud = 0.1 #define altitude of clouds, in bar
T = 3000 #atmospheric temperature in K, this is only used to calculate the scale height
  #ie can be adjusted separately to the opacity temperature

Rstar = 1.756*const.R_sun.value #stellar radius in m
Mpl = 0.894*const.M_jup.value #planet mass in kg
Rpl = 0.10852*const.R_sun.value #planet radius in m
mmw = 2.3 *const.m_p.value #mean molecular weight kg
gamma = 0.54

#derive quantities
g = const.G.value * Mpl / Rpl**2 #surface gravity of planet
Hs = const.k_B.value * T / (2.3*const.m_p.value * g) #calculate atmospheric scale height
P0 = 10. * 100000. #reference pressure, bar to Pa at R0
R0 = Rpl #reference radius

#calculate constant of model
K = R0 + Hs * (gamma + np.log(P0 / g) - np.log(mmw) + 0.5*np.log(2.*np.pi*R0/Hs) )

#calcualte the transmission spectrum
r_species = K + Hs * np.log(chi_FeI * sigma) #this is in units of m

#calculate radius of cloud deck, which forms the continuum
r_continuum = Hs * np.log(P0/Pcloud/100000.) + R0

#truncate the transmission spectrum at the altitude of the cloud deck (as atmosphere is completely opaque lower down)
r = np.maximum(r_species,r_continuum)

#finally make a plot of the transmission spectrum
# f,a = plt.subplots(1)
# a.plot(wl,r_species,'0.5',alpha=0.5)
# # a.plot(wl,r_continuum * np.ones(wl.size),'b-')
# a.plot(wl,r,'r-')
# a.set_xlabel(r'$\lambda (\AA)$')
# a.set_ylabel(r'effective planetary radius (m)')
# a.grid()

rprs = r / Rstar
rprs_continuum = r_continuum / Rstar
rprs_species = r_species / Rstar

#for high-res, often we want units in negative delta flux
dflux = - (rprs**2 - rprs_continuum**2)

f,a = plt.subplots()
a.plot(wl, dflux*10000)
# a.plot(wl,rprs_species,'0.5',alpha=0.5)
# a.plot(wl,rprs_continuum * np.ones(wl.size),'b-')
# a.plot(wl,rprs,'r-')
a.set_ylabel(r'Flux')
a.set_xlabel(r'$\lambda (\AA)$')

#save the model spectrum
# np.save('model_FeI_2500K',np.array([wl,dflux]))

f,a = plt.subplots()
#finally, we're going to broaden the template a little by convoluting with a Gaussian kernel
#this allows us to match the template to the real data, e.g. rotation/instrumental broadening
sigma = 2 #stdev of gaussian
x = np.linspace(-50,50,1000)
g = np.exp(-0.5*x**2 / sigma**2) #create the guassian function
g /= g.sum() #normalise
dflux_broad = np.convolve(g,dflux,'same')
a.plot(wl,dflux_broad * 10000,'g-')

a.set_xlabel(r'$\lambda (\AA)$')
a.set_ylabel(r'$\Delta F (\times 10^{-4})$')

f,a = plt.subplots()
delta_flux = dflux - dflux_broad
a.plot(wl, delta_flux * 10000, '-r')
a.set_xlabel(r'$\lambda (\AA)$')
a.set_ylabel(r'$\Delta F (\times 10^{-4})$')

f,a = plt.subplots()
a.plot(x, g)

plt.show()