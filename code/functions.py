import numpy as np
import astropy.constants as const
import scipy.signal as scisig

def Time_Dependent_Spectrum(wl, flux, op, kp, kernel=None, wavelength_grid=None):
    vp = kp * np.sin(2 * np.pi * op)

    if not isinstance(wavelength_grid, np.ndarray):
        Wavelengths = np.outer(1 - vp * 1000 / const.c.value, wl)
    else:
        Wavelengths  = np.outer(1 - vp * 1000 / const.c.value, wavelength_grid)

    spectrum = np.interp(Wavelengths, wl, flux)
    if not isinstance(kernel, np.ndarray):
        return spectrum

    if len(kernel.shape) > 1:
        convolved_spectrum = np.empty(spectrum.shape)
        for i, ker in enumerate(kernel):
            convolved_spectrum[i] = scisig.fftconvolve(spectrum[i], ker, "same")
    else:
        convolved_spectrum = np.array(
            [scisig.fftconvolve(i, kernel, 'same') for i in spectrum]
        )
    return convolved_spectrum


def Cross_Correlator(wl, flux, vsys, spectrum):
    wl_doppler = np.outer(1 - vsys / const.c.value, wl)
    model = np.interp(wl_doppler, wl, flux)
    CC = np.dot(spectrum, model.T)
    return CC


def Kp_vsys_Plotter(K, vsys, op, CC):
    K_vsys_map = np.empty((K.size, vsys.size))
    CC_array = np.empty(CC.shape)
    CC_shifted = np.empty((K.size, CC.shape[0], CC.shape[1]))
    for i, kp in enumerate(K):
        vp = kp * np.sin(2 * np.pi * op)
        for j, vel in enumerate(vp):
            CC_array[j] = np.interp(vsys + vel, vsys, CC[j])
        K_vsys_map[i] = np.sum(CC_array, axis=0)
        CC_shifted[i] = CC_array
    return K_vsys_map, CC_shifted


def Kp_vsys_Map_from_Flux(wl, flux, op, vsys, kp, ker=None, K=None, wl_grid=None):
    if K == None:
        K = np.linspace(0, 2 * kp, 1001)
    convolved_spectrum = Time_Dependent_Spectrum(wl, flux, op, kp, ker,wl_grid)
    CC = Cross_Correlator(wl, flux, vsys * 1000, convolved_spectrum)
    Kp_vsys_plot, _ = Kp_vsys_Plotter(K, vsys, op, CC)
    return Kp_vsys_plot, K

