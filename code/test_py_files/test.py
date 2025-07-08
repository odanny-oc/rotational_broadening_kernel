import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.config import petitradtrans_config_parser

# mpl.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     "font.family": "serif",
#     "text.usetex": True,
#     "pgf.rcfonts": False,
#     "font.size": 15,
#     "figure.figsize": [20, 15],
#     "axes.facecolor": (1.0 ,1.0, 1.0, 1.0),
#     "savefig.facecolor": (1.0 ,1.0, 1.0, 1.0)
# })

# notes_directory = r"/home/danny/exoplanet_atmospheres/notes/"
input_data_path = r'/home/ocean/petitRADTRANS/input_data'
#
#
petitradtrans_config_parser.set_input_data_path(input_data_path)

radtrans = Radtrans(
    pressures=np.logspace(-6, 2, 100),
    line_species=[
        'H2O',
        'CO-NatAbund',
        'CH4',
        'CO2',
        'Na',
        'K'
    ],
    rayleigh_species=['H2', 'He'],
    gas_continuum_contributors=['H2--H2', 'H2--He'],
    wavelength_boundaries=[0.3, 15]
)

temperatures = 1200 * np.ones_like(radtrans.pressures) # note that radtrans.pressures is in cgs units now, multiply by 1e-6 to get bars

mass_fractions = {
    'H2': 0.74 * np.ones(temperatures.size),
    'He': 0.24 * np.ones(temperatures.size),
    'H2O': 1e-3 * np.ones(temperatures.size),
    'CO-NatAbund': 1e-2 * np.ones(temperatures.size),
    'CO2': 1e-4 * np.ones(temperatures.size),
    'CH4': 1e-5 * np.ones(temperatures.size),
    'Na': 1e-4 * np.ones(temperatures.size),
    'K': 1e-6 * np.ones(temperatures.size)
}

#  2.33 is a typical value for H2-He dominated atmospheres
mean_molar_masses = 2.33 * np.ones(temperatures.size)

planet_radius = 1.0 * cst.r_jup_mean
reference_gravity = 10 ** 3.5
reference_pressure = 0.01

wavelengths, transit_radii, _ = radtrans.calculate_transit_radii(
    temperatures=temperatures,
    mass_fractions=mass_fractions,
    mean_molar_masses=mean_molar_masses,
    reference_gravity=reference_gravity,
    planet_radius=planet_radius,
    reference_pressure=reference_pressure
)


fig, ax = plt.subplots(figsize = (10, 6))

ax.plot(wavelengths * 1e4, transit_radii / cst.r_jup_mean)
ax.set_xscale('log')
ax.set_xlabel('Wavelength [microns]')
ax.set_ylabel(r'Transit radius [$\rm R_{Jup}$]')

plt.show()
# plt.savefig(notes_directory + "transmission1.svg", transparent = True)
