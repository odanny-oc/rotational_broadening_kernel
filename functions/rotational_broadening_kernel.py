import numpy as np


def rotational_broadening_kernel(x, op, veq):
    """
    Produces kernels that model and account for the roational broadening of spectral lines from the emission spectrum of an exoplanet, as well as accounting for the phase of the day side, making the assumption that all of the flux is coming from the day side.
    """
    # Allow for non-array inputs of orbital phase
    if not isinstance(op, np.ndarray):
        op = np.array([op])
    kernel_array = np.zeros(shape=(op.shape[0], x.shape[0]))

    # Finds valid range of x for postive values
    ref_range = np.array([i for i in x if abs(i) <= veq])
    ref_kernel = np.sqrt(1 - (ref_range / veq) ** 2)

    # Pads kernel such that it is the same shape as x
    ref_padding = abs(x.shape[0] - ref_range.shape[0]) // 2
    ref_kernel = np.pad(ref_kernel, ref_padding, "constant")

    normaliser = np.sum(ref_kernel)
    ref_kernel /= normaliser
    # Takes right hand side of the kernel (op = 0.75)
    ref_kernel[: x.shape[0] // 2] = np.zeros(x.shape[0] // 2)

    for i, op_i in enumerate(op):
        # Change to range of (0,1)
        if op_i < 0:
            ref_op = op_i + 1
        else:
            ref_op = op_i

        # Find edge of day side
        vel = veq * np.cos(2 * np.pi * ref_op)

        #Consider zero case
        if vel == 0:
            kernel = ref_kernel
            # Consider opposite side of orbit
            if ref_op < 0.5:
                kernel = np.flip(kernel)
        else:
            # Make kernel with vel instead of veq
            range = np.array([i for i in x if abs(i) <= abs(vel)])
            kernel = np.sqrt(1 - (range / abs(vel)) ** 2) / normaliser
            padding = abs(x.shape[0] - range.shape[0]) // 2
            kernel = np.pad(kernel, padding, "constant")
            if vel < 0:
                kernel[x.shape[0] // 2 :] = np.zeros(x.shape[0] // 2 + 1)
                kernel += ref_kernel

            if vel > 0:
                kernel[: x.shape[0] // 2] = np.zeros(x.shape[0] // 2)
                kernel = ref_kernel - kernel

            if ref_op < 0.5:
                kernel = np.flip(kernel)

        kernel_array[i] = kernel

    # For single values of orbital phase
    if kernel_array.shape[0] == 1:
        return kernel_array[0]
    else:
        return kernel_array
