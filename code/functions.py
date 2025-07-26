import numpy as np
import astropy.constants as const
from numpy.core.multiarray import ndarray
import scipy.signal as scisig
import time

def timer(func):
    def wrapper():
        start_time = time.time()
        func()
        print(f"{func.__name__} -- %s seconds" % (time.time() - start_time))
    return wrapper

def vel_array(vmax, R):

  """
  Simple function to return a velocity scale from at least -vmax to vmax with a given resolution. In km/s.

  """

  dv = const.c.value * 1e-3 / R # get vel spacing  

  n = np.ceil(vmax / dv) # get number of steps required

  N = 2*int(n)+1 # get total number of steps

  v = np.linspace(-n*dv,n*dv,N) # set velocity array with spacing dv

  return v  

def Time_Dependent_Spectrum(wl, flux, op, kp, kernel=None, wl_grid=None):
    vp = kp * np.sin(2 * np.pi * op)

    if not isinstance(wl_grid, np.ndarray):
        Wavelengths = np.outer(1 - vp * 1000 / const.c.value, wl)
    else:
        Wavelengths = np.outer(1 - vp * 1000 / const.c.value, wl_grid)

    if not isinstance(kernel, np.ndarray):
        spectrum = np.interp(Wavelengths, wl, flux)
        return spectrum

    else:
        if len(kernel.shape) > 1:
            convolved_spectrum = np.empty(Wavelengths.shape)
            for i, op in enumerate(op):
                convolved_flux = scisig.fftconvolve(flux, kernel[i], "same")
                convolved_spectrum[i] = np.interp(Wavelengths[i], wl, convolved_flux)
        else:
            convolved_flux = scisig.fftconvolve(flux, kernel, 'same')
            convolved_spectrum = np.interp(Wavelengths, wl, convolved_flux)
    return convolved_spectrum


def Cross_Correlator(wl, flux, vsys, spectrum):
    wl_doppler = np.outer(1 - vsys / const.c.value, wl)
    model = np.interp(wl_doppler, wl, flux)
    CC = np.dot(spectrum, model.T)
    return CC


def Kp_vsys_Plotter(K, vsys, op, CC, vsys_kp=None):
    K_vsys_map = np.empty((K.size, vsys.size))
    CC_array = np.empty(CC.shape)
    CC_shifted = np.empty((K.size, CC.shape[0], CC.shape[1]))
    for i, kp in enumerate(K):
        vp = kp * np.sin(2 * np.pi * op)
        for j, vel in enumerate(vp):
            if not isinstance(vsys_kp, np.ndarray):
                CC_array[j] = np.interp(vsys + vel, vsys, CC[j])
            else:
                CC_array[j] = np.interp(vsys_kp + vel, vsys, CC[j])
        K_vsys_map[i] = np.sum(CC_array, axis=0)
        CC_shifted[i] = CC_array
    return K_vsys_map, CC_shifted


def Kp_vsys_Map_from_Flux(
    wl, flux, op, vsys, kp, ker=None, K=None, wl_grid=None, flux_grid=None, vsys_kp=None
):
    if not isinstance(K, np.ndarray):
        K = np.linspace(0, 2 * kp, 1001)
    convolved_spectrum = Time_Dependent_Spectrum(wl, flux, op, kp, ker, wl_grid)
    if not isinstance(flux_grid, np.ndarray):
        CC = Cross_Correlator(wl, flux, vsys * 1000, convolved_spectrum)
    else:
        CC = Cross_Correlator(wl_grid, flux_grid, vsys * 1000, convolved_spectrum)
    if not isinstance(vsys_kp, np.ndarray):
        Kp_vsys_plot, _ = Kp_vsys_Plotter(K, vsys, op, CC)
    else: 
        Kp_vsys_plot, _ = Kp_vsys_Plotter(K, vsys, op, CC, vsys_kp=vsys_kp)
    return Kp_vsys_plot, K

def maxIndex(a):
    index = np.where(a == np.max(a))
    return np.array([index[0][0], index[1][0]])
