import numpy as np
import astropy.constants as const
import scipy.signal as scisig

# def Broadening_Kernel_OP(x, op):
#     if not isinstance(op, np.ndarray):
#         op = np.array([op])
#     kernel_array = np.zeros(shape=(op.shape[0], x.shape[0]))
#     ref_range = np.array([i for i in x if abs(i) <= veq])
#     ref_kernel = np.sqrt(1 - (ref_range / veq) ** 2)
#     ref_padding = abs(x.shape[0] - ref_range.shape[0]) // 2
#     ref_kernel = np.pad(ref_kernel, ref_padding, "constant")
#     normaliser = np.sum(ref_kernel)
#     ref_kernel /= normaliser
#     full_kernel = ref_kernel.copy()
#     ref_kernel[: x.shape[0] // 2] = np.zeros(x.shape[0] // 2)
#     period_scaler = period / (4 * (tdur - tfull) / 2)
#
#     for i, op_i in enumerate(op):
#         if op_i < 0:
#             ref_op = op_i + 1
#         else:
#             ref_op = op_i
#
#         vel = veq * np.cos(2 * np.pi * ref_op)
#
#         if vel == 0:
#             kernel = ref_kernel
#             if ref_op < 0.5:
#                 kernel = np.flip(kernel)
#         else:
#             if abs(ref_op - 0.5) < tdur / (2 * period) or abs(ref_op + 0.5) < tdur / (
#                 2 * period
#             ):
#                 if abs(ref_op - 0.5) < tfull / (2 * period) or abs(
#                     ref_op + 0.5
#                 ) < tfull / (2 * period):
#                     kernel = np.zeros(x.shape[0])
#                 else:
#                     if ref_op > 0:
#                         vel = veq * np.cos(
#                             (2 * np.pi * (ref_op - 0.5)) * (period_scaler)
#                         )
#                     else:
#                         vel = veq * np.cos(
#                             (2 * np.pi * (ref_op + 0.5)) * (period_scaler)
#                         )
#                     range = np.array([i for i in x if abs(i) <= abs(vel)])
#                     kernel = np.sqrt(1 - (range / abs(vel)) ** 2) / normaliser
#                     padding = abs(x.shape[0] - range.shape[0]) // 2
#                     kernel = np.pad(kernel, padding, "constant")
#                     if vel < 0:
#                         kernel[x.shape[0] // 2 :] = np.zeros(x.shape[0] // 2 + 1)
#                         kernel += ref_kernel
#                     if vel > 0:
#                         kernel[: x.shape[0] // 2] = np.zeros(x.shape[0] // 2)
#                         kernel = ref_kernel - kernel
#                     if ref_op > 0.5:
#                         kernel = np.flip(kernel)
#             else:
#                 range = np.array([i for i in x if abs(i) <= abs(vel)])
#                 kernel = np.sqrt(1 - (range / abs(vel)) ** 2) / normaliser
#                 padding = abs(x.shape[0] - range.shape[0]) // 2
#                 kernel = np.pad(kernel, padding, "constant")
#                 if vel < 0:
#                     kernel[x.shape[0] // 2 :] = np.zeros(x.shape[0] // 2 + 1)
#                     kernel += ref_kernel
#                 if vel > 0:
#                     kernel[: x.shape[0] // 2] = np.zeros(x.shape[0] // 2)
#                     kernel = ref_kernel - kernel
#                 if ref_op < 0.5:
#                     kernel = np.flip(kernel)
#         kernel_array[i] = kernel
#     if kernel_array.shape[0] == 1:
#         return kernel_array[0], full_kernel
#     else:
#         return kernel_array, full_kernel


def Time_Dependent_Spectrum(wl, flux, op, kp, kernel=None):
    vp = kp * np.sin(2 * np.pi * op)
    Wavelengths = np.outer(1 - vp * 1000 / const.c.value, wl)
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


def Kp_vsys_Map_from_Flux(wl, flux, op, vsys, kp, ker=None, K=None):
    if K == None:
        K = np.linspace(0, 2 * kp, 1000)
    convolved_spectrum = Time_Dependent_Spectrum(wl, flux, op, kp, ker)
    CC = Cross_Correlator(wl, flux, vsys * 1000, convolved_spectrum)
    Kp_vsys_plot, _ = Kp_vsys_Plotter(K, vsys, op, CC)
    return Kp_vsys_plot, K

