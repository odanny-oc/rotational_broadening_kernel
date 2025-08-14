# Rotational Broadening Kernel
Documents code written and progress of 10 week project to attempt to exploit rotational broadening of exoplanets using a simple time dependent models to improve spatial resolution.

Code implements a time dependent "kernel" to model the phases of the exoplanet as the day side changes with respect to time, thus accounting for the radial velocity of the exoplanet as it spins.

**Note**
To run the spectrum generation directory one must have the petitRADTRANS package installed. https://petitradtrans.readthedocs.io/en/latest/
This isn't neccessary to run the simulations directory as the output from spectrum_generation is in the data directory. This is so one can demonstrate the rotational broadening kernel without generating their own spectra.
