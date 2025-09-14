import numpy as np
import astropy.constants as const

def kernel_to_cc(kernels, op, Kp, resolution):
    """
    Function to move kernel to bulk velocity of planet so that we can do the Kp-vsys plot directly
    """
    dv = const.c.value * 1e-3 / resolution 
    vp = Kp * np.sin(2 * np.pi * op)
    for i,kernel in enumerate(kernels):
        index =  vp[i]//dv
        index = int(index)
        kernels[i] = np.roll(kernel, index)
    return kernels
