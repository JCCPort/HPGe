import numpy as np
from scipy.optimize import curve_fit
import os
import pandas as pd
import uncertainties
import matplotlib.pyplot as plt


## Polynomials

pd.set_option('display.max_columns', 6)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def g6(v, a, b, c, d, e, f, g):
    return a * v ** 6 + b * v ** 5 + c * v ** 4 + d * v ** 3 + e * v ** 2 + f * v + g


def g2(v, m, c):
    return m * v + c


def g3(v, h, i, j):
    return j * v ** 2 + i * v + h


def g4(v, k, l, m, n):
    return n * v ** 3 + m * v ** 2 + l * v + k


def lingauss(xvar, co_a, co_b, co_c, co_d, co_e):
    return co_a * np.exp(-((xvar - co_b) ** 2) / (2 * (co_c ** 2))) + (co_d * (xvar - co_b)) + co_e


def gauss(xvar, co_a, co_b, co_c):
    return co_a * np.exp(-((xvar - co_b) ** 2) / (2 * (co_c ** 2)))


def calibrate(calib_func, calibData):
    popt, pcov = curve_fit(calib_func, calibData["Val"], calibData["Actual"], sigma=calibData["ValUncertainty"], absolute_sigma=False, maxfev=100000)
    err = np.sqrt(np.diag(pcov))
    low_div = popt - err
    high_div = popt + err
    list = []
    calibData['CalibVal'] = calib_func(calibData['Val'], *popt)
    print(calibData)
    for i in range(0, len(calibData)):
        list.append([calib_func(calibData.at[i, 'Val'], *popt),
                     calib_func(calibData.at[i, 'Val'], *low_div),
                     calib_func(calibData.at[i, 'Val'], *high_div),
                     calib_func(calibData.at[i, 'Val'], *popt) - calib_func(calibData.at[i, 'Val'], *low_div),
                     calib_func(calibData.at[i, 'Val'], *high_div) - calib_func(calibData.at[i, 'Val'], *popt)])
    calib_frame = pd.DataFrame(list, columns=['Value', 'Lower limit', 'Upper limit', 'Bottom error', 'Top error'])
    escp = uncertainties.ufloat(popt[0], err[0])
    mtch = uncertainties.ufloat(popt[1], err[1])
    # plt.plot(calibData["Val"], calibData["Actual"])
    # calibVals = np.linspace(np.min(calibData["Val"]), np.max(calibData["Val"])*2, 1000)
    # plt.plot(calibVals, calib_func(calibVals, *popt))
    # plt.plot(calibData["Val"], calibData["Actual"], 'x')
    # plt.show()
    # plt.plot(calibData["Val"], calibData["Actual"]-calib_func(calibData["Val"], *popt), 'x')
    # plt.title("Calibration residuals")
    # plt.show()
    errors = np.sqrt(np.diag(pcov))
    # print(escp, mtch)
    print(calib_frame)

    def getCalibError(channelVal):
        lowLim = calib_func(channelVal, *popt) - calib_func(channelVal, *low_div)
        highLim = calib_func(channelVal, *high_div) - calib_func(channelVal, *popt)
        calibErr = abs(lowLim-highLim)
        print(err)
        print(calibErr)
        return calibErr

    return calib_frame, popt, pcov, calib_func, errors, getCalibError


def fullCalibrate(calibFunc=g2):
    preData = []
    for filename in os.listdir("Calibrations"):
        preFrame = pd.read_csv("Calibrations/" + filename, index_col=0)
        preFrame['Label'] = None
        for i in range(0, len(preFrame)):
            preFrame['Label'][i] = filename.split('.')[0]
        preData.append(preFrame)
    frame = pd.concat(preData, axis=0)
    frame.reset_index(inplace=True)
    calibStuff = calibrate(calibFunc, frame)
    return calibStuff


def fitUncalibRanges(name):
    data = pd.read_csv("Data/{}.csv".format(name), names=["Energy (KeV)", "Counts (a.u)"])
    # calibInfo = fullCalibrate()
    # data["Energy (KeV)"] = calibInfo[3](data["Energy (KeV)"], *calibInfo[1])
    ranges = pd.read_csv("Ranges/{}.csv".format(name))
    peaks = pd.DataFrame(columns=["Mean", "MeanError", "Chisq", "ReducedChisq", "Amplitude"])
    for i in range(0, len(ranges)):
        xData = data["Energy (KeV)"][ranges["LowerIndex"][i]:ranges["UpperIndex"][i]]
        yData = data["Counts (a.u)"][ranges["LowerIndex"][i]:ranges["UpperIndex"][i]]
        # yData = (yData - np.min(yData)) * (1 / max(yData))
        xNums = np.linspace(np.min(xData), np.max(xData), 1000)
        tipEnergyIndex = np.where(yData == np.max(yData))[0][0]
        initial = [np.max(yData)-np.min(yData),
                   xData.tolist()[tipEnergyIndex],
                   (np.max(xData)-np.min(xData))/2,
                   (yData.iloc[-1]-yData.iloc[0])/(xData.iloc[-1]-xData.iloc[0]),
                   np.min(yData)]
        bounds = [[0, 0.97*initial[1], 0.1*initial[2], initial[3]-20*(np.abs(initial[3]))-0.01, 0.90*initial[4]-1],
                  [1.5*np.max(yData), 1.03*initial[1], 1.3*initial[2], initial[3]+20*(np.abs(initial[3]))+0.01, 1.1*initial[4]+2]]
        pOpt, pCov = curve_fit(lingauss, xData, yData, sigma=np.sqrt(yData+0.0001), absolute_sigma=True, p0=initial, bounds=bounds, maxfev=10000000)
        errors = np.sqrt(np.diag(pCov))
        print(pOpt, "\n", errors)
        # chisq = 0
        # for j in range(0, len(xData)):
        chisq = np.sum(((yData - lingauss(yData, *pOpt))**2)/lingauss(yData, *pOpt))
        reducedChisq = chisq/len(xData)
        # pOpt, pCov = curve_fit(lingauss, xData, yData, p0=initial, maxfev=10000000)
        xData2 = data["Energy (KeV)"][int(ranges["LowerIndex"][i]*0.9):int(ranges["UpperIndex"][i]*1.1)]
        yData2 = data["Counts (a.u)"][int(ranges["LowerIndex"][i]*0.9):int(ranges["UpperIndex"][i]*1.1)]
        plt.axvline(xData.tolist()[tipEnergyIndex], color='r')
        plt.plot(xData, yData, 'x')
        plt.plot(xData2, yData2)
        plt.plot(xNums, lingauss(xNums, *pOpt))
        plt.show()
        # peaks.loc[i] = {"Mean": pOpt[1], "MeanError": pOpt[2]/2.355, "Chisq": chisq, "ReducedChisq": reducedChisq, "Amplitude": pOpt[0]}
        peaks.loc[i] = {"Mean": pOpt[1], "MeanError": errors[1], "Chisq": chisq, "ReducedChisq": reducedChisq, "Amplitude": pOpt[0]}
    peaks.to_csv("Peaks/" + name + ".csv")
    return peaks


def fitRanges(name):
    data = pd.read_csv("Data/{}.csv".format(name), names=["Energy (channel)", "Counts (a.u)"])
    calibInfo = fullCalibrate(g6)
    getCalibError = calibInfo[5]
    data["Energy (KeV)"] = calibInfo[3](data["Energy (channel)"], *calibInfo[1])
    ranges = pd.read_csv("Ranges/{}.csv".format(name))
    peaks = pd.DataFrame(columns=["Mean", "MeanError", "Chisq", "ReducedChisq", "Amplitude"])
    for i in range(0, len(ranges)):
        xData = data["Energy (KeV)"][ranges["LowerIndex"][i]:ranges["UpperIndex"][i]]
        yData = data["Counts (a.u)"][ranges["LowerIndex"][i]:ranges["UpperIndex"][i]]
        xNums = np.linspace(np.min(xData)*0.9, np.max(xData)*1.1, 1000)

        tipEnergyIndex = np.where(yData == np.max(yData))[0][0]
        # yDataTipIndex = np.where(yData == np.max(yData))[0][0]
        # lowSideWhere = find_nearest(yData[:yDataTipIndex-1], ((np.max(yData)*1.2-np.min(yData))/2) + np.min(yData))
        # highSideWhere = find_nearest(yData[yDataTipIndex+1:], ((np.max(yData)*1.2-np.min(yData))/2) + np.min(yData))
        # lowSideFWHM = data["Energy (KeV)"][np.where(data["Counts (a.u)"] == lowSideWhere)[0][0]]
        # highSideFWHM = data["Energy (KeV)"][np.where(data["Counts (a.u)"] == highSideWhere)[0][0]]
        # print(lowSideFWHM, highSideFWHM)
        # initial = [np.max(yData)-np.min(yData),
        #            data["Energy (KeV)"][tipEnergyIndex],
        #            highSideFWHM - lowSideFWHM,
        #            (yData.iloc[-1]-yData.iloc[0])/(xData.iloc[-1]-xData.iloc[0]),
        #            np.min(yData)]
        # bounds = [[0.2*np.max(yData), 0.95*initial[1], 0.5*initial[2], -10000, 0.9*initial[4]],
        #           [1.3*np.max(yData), 1.05*initial[1], 1.5*initial[2], 10000, 1.1*initial[4]+0.001]]
        # plt.axvline(lowSideFWHM)
        # plt.axvline(highSideFWHM)

        print(xData.tolist())
        initial = [np.max(yData)-np.min(yData),
                   xData.tolist()[tipEnergyIndex],
                   (np.max(xData)-np.min(xData))/2,
                   (yData.iloc[-1]-yData.iloc[0])/(xData.iloc[-1]-xData.iloc[0]),
                   np.min(yData)]
        bounds = [[0, 0.97*initial[1], 0.1*initial[2], initial[3]-20*(np.abs(initial[3]))-0.01, 0.90*initial[4]-1],
                  [1.5*np.max(yData), 1.03*initial[1], 1.3*initial[2], initial[3]+20*(np.abs(initial[3]))+0.01, 1.1*initial[4]+2]]
        print(bounds)
        pOpt, pCov = curve_fit(lingauss, xData, yData, sigma=np.sqrt(yData+0.00001), absolute_sigma=True, bounds=bounds, p0=initial, maxfev=10000000)
        print(bounds, "\n", pOpt)
        errors = np.diag(pCov)
        print("Parameters", pOpt, "\n", "Uncertainty", errors)
        chisq = np.sum(((yData - lingauss(yData, *pOpt))**2)/lingauss(yData, *pOpt))
        reducedChisq = chisq/len(xData)
        # print(pOpt)
        xData2 = data["Energy (KeV)"][int(ranges["LowerIndex"][i]*0.9):int(ranges["UpperIndex"][i]*1.1)]
        yData2 = data["Counts (a.u)"][int(ranges["LowerIndex"][i]*0.9):int(ranges["UpperIndex"][i]*1.1)]
        plt.axvline(xData.tolist()[tipEnergyIndex], color='r')
        plt.plot(xData, yData, 'x')
        plt.plot(xData2, yData2)
        plt.plot(xNums, lingauss(xNums, *pOpt))
        plt.show()
        peaks.loc[i] = {"Mean": pOpt[1], "MeanError": np.sqrt((pOpt[2]/2.355)**2 + getCalibError(pOpt[1])**2), "Chisq": chisq, "ReducedChisq": reducedChisq,
        "Amplitude": pOpt[0]}
        # peaks.loc[i] = {"Mean": pOpt[1], "MeanError": np.sqrt((errors[1])**2 + getCalibError(pOpt[1])**2), "Chisq": chisq, "ReducedChisq": reducedChisq,
        #                 "Amplitude": pOpt[0]}
        # print(peaks.loc[i])
    peaks.to_csv("Peaks/" + name + ".csv")
    return peaks


def makeCalibFrame(name):
    frame = pd.DataFrame(columns=["Val", "Actual", "ValUncertainty"])
    k = 0
    singleSetPeaks = fitUncalibRanges(name)
    for i in range(0, len(singleSetPeaks)):
        actual = input("What is the actual energy of the {} peak at {}? (in keV)".format(name, singleSetPeaks.at[i, "Mean"]))
        plt.show(block=True)
        print(actual)
        if actual != "":
            frame.loc[k] = {"Val": singleSetPeaks.at[i, "Mean"], "Actual": actual, "ValUncertainty": singleSetPeaks.at[i, "MeanError"]}
            k += 1
    frame.to_csv("Calibrations/" + name + "Calib.csv")


if __name__ == "__main__":
    print("main")
    # fitUncalibRanges("UnshieldedNa22_2")
    # makeCalibFrame("UnshieldedBa133_2")
    # makeCalibFrame("Na22Shielded")
    # makeCalibFrame("ShieldedTrinitite")
    # fullCalibrate(g4)
    fitRanges("ShieldedTrinititeFIN2")


def point_errors(calib_func, point, calibData):
    popt, pcov = curve_fit(calib_func, calibData["Val"], calibData["Actual"], sigma=calibData["Uncert"], absolute_sigma=True, maxfev=100000)
    err = np.diag(pcov)
    low_div = popt - err
    high_div = popt + err
    return (abs(calib_func(point, *popt) - calib_func(point, *low_div)) + abs(calib_func(point, *high_div) - calib_func(point, *popt))) / 2.
