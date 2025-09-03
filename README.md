# Rotational Broadening Kernel
Documents code written and progress of 10 week project to attempt to exploit rotational broadening of exoplanets using a simple time dependent models to improve spatial resolution.

Code implements a time dependent "kernel" to model the phases of the exoplanet as the day side changes with respect to time, thus accounting for the radial velocity of the exoplanet as it spins.

**Note**:

To run the spectrum generation directory one must have the petitRADTRANS package installed. https://petitradtrans.readthedocs.io/en/latest/

This isn't neccessary to run the simulations directory, as the output from spectrum_generation is in the data directory. This is so one can demonstrate the rotational broadening kernel without generating their own spectra.

# How it works
For a given spectra over time (in this case over an exoplanets "orbital phase"), it follows a Keplerian (sinusoidal) motion as the bulk velocity of the planet red and blue shifts the spectrum. Here is a simulated spectrum from the data directory of H2O in the planet WASP-121b.

<img width="640" height="480" alt="spectrum_git" src="https://github.com/user-attachments/assets/40867a9e-b8c1-4759-a403-58bc40c68540" />

This spectrum can then be convolved with a "kernel". Here, the kernel is trying to account for the "phase" (not to be confused with orbital phase) of the planet i.e how much of the planet is illumated by the star. In the emission spectrum of the planet, the flux measured is highly dependent on the temperture. Since the bright sun facing side of the exoplanet would be much hotter, most of the flux comes from the day-side of the exoplanet. 

 ![Exoplanet_phase_curve_ESA21503702](https://github.com/user-attachments/assets/e898da39-d904-4326-9ed2-7f622b08a7ac)

By manipulating the shape of the spectrum in velocity space we can also then account for rotational broadening as discussed above. The kernels are time dependent and evolve as such:

<img width="640" height="480" alt="kernels_git" src="https://github.com/user-attachments/assets/e029c2e9-72b5-42cf-8a06-4836725c6f0e" />

This considers all the flux to be coming from the day-side of the planet.

After convolution with the spectrum we get;

<img width="640" height="480" alt="convolved_spectrum_git" src="https://github.com/user-attachments/assets/13606320-c713-4a50-bc21-8a5d8bd6f34d" />

It's clear that at 0 orbital phase we recieve no signal as the day side is totally facing the star.

A day-night model is also used by generating two kernels and two spectra with different elements (H2O in the day side and CO in the night side).
This allows both sides to analysied seperately.

<img width="640" height="480" alt="anti_kernels_git" src="https://github.com/user-attachments/assets/f2fd0601-85ba-4cf6-bfa6-0b3d45de5d04" />

The results of these models seem to be a breakdown in the Keplerian model of the exoplanet as the change in strength and direction of the rotaional broading breaks the sinusoidal pattern of the planet motion. This can be seen by adding the bulk velocity to the kernels and plotting them directly over time;

<img width="640" height="480" alt="direct_kernel_git" src="https://github.com/user-attachments/assets/f46ea61f-be1f-4015-92dd-5203b68d7033" />

While it still obviously follow an overall sinusoidal shape, the small inconsistencies due to the rotaional broadening cause the traditional methods of analysis, Kp-vsys plots, tp be ineffective. I believe this model demonstrates that with more understanding of this effect, we could retrieve more information about the speed of the planet and increase our spacial resolution, decreasing the uncertainty of measurement of atmospheric species.

