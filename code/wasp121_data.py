import numpy as np
import matplotlib.pyplot as plt
import os
from functions import Time_Dependent_Spectrum
from functions import Cross_Correlator
from functions import Kp_vsys_Plotter
from functions import Kp_vsys_Map_from_Flux
from functions import maxIndex
from functions import vel_array
from astropy import constants as const
import scipy.signal as scisig
import time

home_path = os.environ["HOME"]

local_path = home_path + "/exoplanet_atmospheres/code"

data_pre = np.load(
    os.path.join(local_path, "crires_preeclipse_WASP121_2022-02-14_processed.npz")
)

data_post = np.load(
    os.path.join(local_path, "crires_posteclipse_WASP121_2021-12-15_processed.npz")
)

h2o_spec = np.load(os.path.join(local_path, "h2o_spectrum.npz")
)

print(data_pre.files)

order = 9 
frames = 0

orbital_phase = data_pre['ph'][frames:]
wl = data_pre['W'][order][100:1980]
stellar_spectrum = data_pre['B'][order][frames:]

new_spectrum = np.empty([orbital_phase.shape[0], 1880])
for i in range(orbital_phase.shape[0]):
    new_spectrum[i] = stellar_spectrum[i][100:1980] - np.mean(stellar_spectrum[i][100:1980])

stellar_spectrum = new_spectrum
spec_err = np.zeros_like(stellar_spectrum)
spec_err = np.random.rand(stellar_spectrum.shape[0], stellar_spectrum.shape[1]) * 20
spec_err = abs(spec_err)

print(data_pre['Be'].shape)
spec_err = data_pre['Be'][order][frames:]

new_err = np.empty([orbital_phase.shape[0], 1880])

for i in range(orbital_phase.shape[0]):
    new_err[i] = spec_err[i][100:1980] - np.mean(spec_err[i][100:1980])

spec_err = new_err
print(spec_err.shape)

import SysRem

N_components = 25
Msys = SysRem.FastModelSysRem(stellar_spectrum,spec_err,N_components) #get model of data

#correct data using SysRem model
R = stellar_spectrum / np.abs(Msys) - 1.
Re = spec_err / Msys #do the same for the uncertainties

#we also need to be careful about clearing up nans/infs etc after division
R[np.isnan(R)] = 0.
R[np.isinf(R)] = 0.
Re[Re < 0] = np.inf
Re[np.isnan(Re)] = np.inf #reset errors to infs
Re[np.isclose(Re,0)] = np.inf #reset errors to infs

stellar_spectrum = R/Re


f,a = plt.subplots(3)
a[0].pcolormesh(wl,orbital_phase,Msys)
a[1].pcolormesh(wl,orbital_phase,R)
a[2].pcolormesh(wl,orbital_phase,R/Re)

h2o_wl = h2o_spec['wl']*1e4
h2o_flux = h2o_spec['flux']
h2o_flux-=np.mean(h2o_flux)

fig, ax = plt.subplots()
ax.plot(h2o_wl, h2o_flux)

flux_grid = np.interp(wl, h2o_wl, h2o_flux)

fig, ax = plt.subplots()
ax.pcolormesh(wl, orbital_phase, stellar_spectrum)

Kp = h2o_spec['radial_velocity_semi_amplitude']
vel = np.linspace(0, 500 ,1001)
# vel = Kp * np.sin(2 * np.pi * orbital_phase)
wl_time = np.outer(1 - vel*1000/const.c.value, wl)
model = np.interp(wl_time, wl ,flux_grid)

CC = np.dot(stellar_spectrum, model.T)

fig, ax = plt.subplots(2)
ax[0].plot(wl, stellar_spectrum[0])
ax[1].plot(wl, flux_grid)


fig, ax = plt.subplots()
ax.pcolormesh(vel, orbital_phase, CC)
plt.show()
