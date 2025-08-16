import numpy as np
import astropy.constants as const
import scipy.signal as scisig


def maxindex(a):
    index = np.where(a == np.max(a))
    return np.array([index[0][0], index[1][0]])


def vel_array(vmax, R):

  """
  Simple function to return a velocity scale from at least -vmax to vmax with a given resolution. In km/s.

  """

  dv = const.c.value * 1e-3 / R # get vel spacing  

  n = np.ceil(vmax / dv) # get number of steps required

  N = 2*int(n)+1 # get total number of steps

  v = np.linspace(-n*dv,n*dv,N) # set velocity array with spacing dv

  return v  


def time_dependent_spectrum(wl, flux, op, kp, kernel=None, wl_grid=None):

    """
    Creates time dependent spectrum over given orbital phase and kp, also performs convolution with kernal and fixes spectrum to wavelength grid. In km/s
    """

    # Define planet bulk velocities
    vp = kp * np.sin(2 * np.pi * op)

    # Doppler shift the wavelength given the bulk velocites
    if not isinstance(wl_grid, np.ndarray):
        Wavelengths = np.outer(1 - vp * 1000 / const.c.value, wl)
    # Use wavelength grid values if given
    else:
        Wavelengths = np.outer(1 - vp * 1000 / const.c.value, wl_grid)

    # Return spectrum if no kernel
    if not isinstance(kernel, np.ndarray):
        spectrum = np.interp(Wavelengths, wl, flux)
        print("Unconvolved spectrum generated")
        return spectrum

    else:
        # If kernel is time dependent convolve each kernel and interpolate
        if len(kernel.shape) > 1:
            convolved_spectrum = np.empty(Wavelengths.shape)
            for i, op in enumerate(op):
                convolved_flux = scisig.fftconvolve(flux, kernel[i], "same")
                convolved_spectrum[i] = np.interp(Wavelengths[i], wl, convolved_flux)
        # If kernel is constant in time convolve and interpolate all at once
        else:
            convolved_flux = scisig.fftconvolve(flux, kernel, 'same')
            convolved_spectrum = np.interp(Wavelengths, wl, convolved_flux)
        print("Convolved spectrum generated")

    return convolved_spectrum


def cross_correlator(wl, flux, vsys, spectrum):
    wl_doppler = np.outer(1 - vsys / const.c.value, wl)
    model = np.interp(wl_doppler, wl, flux)
    CC = np.dot(spectrum, model.T)
    print("Cross correlation complete")
    return CC


def kp_vsys_plotter(K, vsys, op, CC, vsys_kp=None):

    """
    Creates Kp - vsys map for given cross correlation
    """

    K_vsys_map = np.empty((K.size, vsys.size))
    CC_array = np.empty(CC.shape)
    CC_shifted = np.empty((K.size, CC.shape[0], CC.shape[1]))
    # Sum over all Kps
    for i, kp in enumerate(K):
        vp = kp * np.sin(2 * np.pi * op)
        # Sum over all velocities
        for j, vel in enumerate(vp):
            if not isinstance(vsys_kp, np.ndarray):
                CC_array[j] = np.interp(vsys + vel, vsys, CC[j])
            else:
                CC_array[j] = np.interp(vsys_kp + vel, vsys, CC[j])
        K_vsys_map[i] = np.sum(CC_array, axis=0)
        CC_shifted[i] = CC_array
    print("Trace summed over all values of Kp")
    return K_vsys_map, CC_shifted


# Chain together all functions
def kp_vsys_map_from_flux(
    wl, flux, op, vsys, kp, ker=None, K=None, wl_grid=None, flux_grid=None, vsys_kp=None
):
    if not isinstance(K, np.ndarray):
        K = np.linspace(0, 2 * kp, 1001)
    convolved_spectrum = time_dependent_spectrum(wl, flux, op, kp, ker, wl_grid)
    if not isinstance(flux_grid, np.ndarray):
        CC = cross_correlator(wl, flux, vsys * 1000, convolved_spectrum)
    else:
        CC = cross_correlator(wl_grid, flux_grid, vsys * 1000, convolved_spectrum)
    if not isinstance(vsys_kp, np.ndarray):
        Kp_vsys_plot, _ = kp_vsys_plotter(K, vsys, op, CC)
    else: 
        Kp_vsys_plot, _ = kp_vsys_plotter(K, vsys, op, CC, vsys_kp=vsys_kp)
    return Kp_vsys_plot, K

