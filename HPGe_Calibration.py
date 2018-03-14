import numpy as np
from scipy.optimize import curve_fit
import os
import pandas as pd

os.chdir(r'C:\Users\Josh\Documents\\git\HPGe2')
xaxis = pd.read_csv('xaxis.csv', header=None, names=['Val'], engine='c')

Na_x = np.array([7.01232586e+02, 1.74965037e+03])  #Measured channel of radiation event
Na_y = np.array([510.9989461, 1274.537])

Co_x = np.array([1.82921107e+03, 1.61055669e+03])
Co_y = np.array([1332.492, 1173.228])

Ba_x = np.array([1.10894684e+02, 2.20327386e+02, 3.06345220e+02, 3.79324341e+02, 4.15673991e+02,
                 4.88662025e+02, 5.26734672e+02])
Ba_y = np.array([80.9979, 160.6121, 223.237, 276.3989, 302.8508, 356.0129, 383.8485])

Others_x = np.array([1891.21476395582, 1537.86095435944]) #bi214, bi214
Others_y = np.array([1377.669, 1120.287])
##Accepted values (y) sourced from IAEA other than 160.6121 and 223.237 which were sourced from nucleide.org

y2 = np.append(Na_y, Co_y)
x2 = np.append(Na_x, Co_x)
# y3 = np.append(Na_y, Co_y)
# x3 = np.append(Na_x, Co_x)
y = np.sort(np.append(y2, Ba_y))
x = np.sort(np.append(x2, Ba_x))

## Polynomials


def g6(v, a, b, c, d, e, f, g):
    return a*v**6 + b*v**5 + c*v**4 + d*v**3 + e*v**2 + f*v + g
def g2(v, m, c):
    return m*v + c
def g3(v, h, i, j):
    return j*v**2 + i*v + h
def g4(v, k, l, m, n):
    return n*v**3 + m*v**2 + l*v + k

sigs = [0.0000013, 0.003, 0.004, 0.003, 0.0011, 0.0016, 0.002, 0.0012, 0.0005, 0.0007, 0.0012]

def calibrate(calib_func):
    popt, pcov = curve_fit(calib_func, x, y, sigma=sigs, absolute_sigma=True, maxfev=100000)
    err = np.diag(pcov)
    low_div = popt - err
    high_div = popt + err
    list = []
    for i in range(0, len(xaxis)):
        list.append([calib_func(xaxis.at[i, 'Val'], *popt),
                            calib_func(xaxis.at[i, 'Val'], *low_div),
                            calib_func(xaxis.at[i, 'Val'], *high_div),
                            calib_func(xaxis.at[i, 'Val'], *popt) - calib_func(xaxis.at[i, 'Val'], *low_div),
                            calib_func(xaxis.at[i, 'Val'], *high_div) - calib_func(xaxis.at[i, 'Val'], *popt)])
    calib_frame = pd.DataFrame(list, columns=['Value', 'Lower limit', 'Upper limit', 'Bottom error', 'Top error'])
    return calib_frame, popt, pcov

def point_errors(calib_func, point):
    popt, pcov = curve_fit(calib_func, x, y, sigma=sigs, absolute_sigma=True)
    err = np.diag(pcov)
    low_div = popt - err
    high_div = popt + err
    return (abs(calib_func(point, *popt) - calib_func(point, *low_div)) + abs(calib_func(point, *high_div) - calib_func(point, *popt)))/2.
