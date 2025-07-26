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

print(data_pre.files)
print(data_pre['ph'].shape)
print(data_pre['W'].shape)
print(data_pre['B'].shape)
print(data_pre['Be'])
print(data_pre['bjd'])
print(data_pre['bvc'])


