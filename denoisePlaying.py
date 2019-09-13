import numpy as np
import os
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from lmfit.models import GaussianModel
import time
from HPGe_Calibration import calibrate, fullCalibrate
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

import datetime
from skimage.restoration import denoise_wavelet, cycle_spin
from skimage import data, img_as_float
from skimage.util import random_noise

dataa = pd.read_csv("Data/LongTrinititeShielded.csv", names=["Energy (KeV)", "Counts (a.u)"])

denoise_kwargs = dict(wavelet='db1', wavelet_levels=7, sigma=9e-14)

smoothed_data = cycle_spin(dataa["Counts (a.u)"], func=denoise_wavelet, max_shifts=15,
                           func_kw=denoise_kwargs)
# smoothed_data = denoise_wavelet(dataa["Counts (a.u)"], sigma=1.4e-13)
smoothedIntegral = np.max(smoothed_data)
originalIntegral = np.max(dataa["Counts (a.u)"])

ratio = smoothed_data/originalIntegral

plt.plot(smoothed_data, label="Smoothed")
plt.plot(dataa["Counts (a.u)"]*ratio*700, label="Original")
plt.xlim([7100, 7600])
plt.ylim([0, 3e-18])
plt.legend()
plt.show()
