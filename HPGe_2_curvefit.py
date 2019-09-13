import numpy as np
import os
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from lmfit.models import GaussianModel
import time
from HPGe_Calibration import calibrate, fullCalibrate, g4
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

import datetime

today = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
print(today)

start_time = time.time()


# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.rcParams['text.latex.preamble'] = [
#     r'\usepackage{siunitx}',
#     r'\sisetup{detect-all}',
#     r'\usepackage{lmodern}',
# ]
# plt.rcParams['ps.usedistiller'] = 'xpdf'
# plt.rcParams['ps.distiller.res'] = '1000'


class RangeTool(object):
    """
    Like Cursor but the crosshair snaps to the nearest x,y point
    For simplicity, I'm assuming x is sorted
    """

    def __init__(self, ax, data, figure2, key):
        self.ax = ax
        self.key = key
        self.data = data
        # self.key = key
        self.figure2 = figure2
        self.lx = self.ax.axhline(color='k')  # the horiz line
        self.ly = self.ax.axvline(color='k')  # the vert line
        self.lowers = np.array([])
        self.uppers = np.array([])
        self.IndependentVariable = "Energy (KeV)"
        self.DependentVariable = "Counts (a.u)"
        self.x = data[self.IndependentVariable]
        self.y = data[self.DependentVariable]
        # self.self.ax.set_xlim(np.min(self.x), np.max(self.x))
        width = np.max(self.x) - np.min(self.x)
        height = np.max(self.y) - np.min(self.y)
        # self.self.ax.set_ylim(np.min(self.y) - 0.1 * height, np.max(self.y) + 0.1 * height)
        # text location in axes coords
        self.txt = self.ax.text(0.7, 0.9, '', transform=self.ax.transAxes)
        self.cid1 = self.ax.figure.canvas.mpl_connect('key_press_event', self.keyPress)
        self.cid2 = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        # self.cid3 = self.ax.figure.canvas.mpl_connect('key_press_event', self.rangeremove)
        # self.cid4 = self.ax.figure.canvas.mpl_connect('key_press_event', self.finishplot)
        # self.checkNonePress = self.ax.figure.canvas.mpl_connect('key_press_event', self.nonePressCheck)
        self.Ranges = pd.DataFrame(columns=['Lower Bound', 'LowerIndex', 'Upper Bound', 'UpperIndex', 'Displayed'])
        self.il = 0
        self.iu = 0
        # self.t = 0

    def __call__(self, event):
        print('click', event)
        print(event.xdata, event.ydata)
        if event.inaxes != self.figure2.axes: return

    def keyPress(self, event):
        print(event.key)
        if event.key is None:
            print("None pressed...?")
            return
        if event.key == "shift" or event.key == "control":
            self.rangeselect(event)
        if event.key == "delete":
            self.rangeremove(event)
        if event.key == "enter" or event.key == "escape":
            self.finishplot(event)


    def mouse_move(self, event):

        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata
        indx = min(np.searchsorted(self.x, [x])[0], len(self.x) - 1)
        x = self.x[indx]
        y = self.y[indx]
        # update the line positions
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)
        # print('{},{}'.format(event.xdata, event.ydata))
        self.txt.set_text('x=%1.2f(%1.2f), y=%1.2f' % (x, self.data["Energy (channels)"][indx], y))
        self.ax.figure.canvas.draw_idle()
        # print('x=%1.2f, y=%1.2f' % (x, y))

    def rangeselect(self, event):
        x = event.xdata
        indx = min(np.searchsorted(self.x, [x])[0], len(self.x) - 1)
        x = self.x[indx]
        if event.key == 'shift':
            print("pressed shift")
            print(self.il)
            self.Ranges.at[self.il, 'Lower Bound'] = x
            self.Ranges.at[self.il, 'LowerIndex'] = indx
            self.il += 1
        if event.key == 'control' and self.il > self.iu:
            self.Ranges.at[self.iu, 'Upper Bound'] = x
            self.Ranges.at[self.iu, 'UpperIndex'] = indx
            self.iu += 1
        print(self.il, self.iu, self.Ranges, "\n")
        if self.il == self.iu and self.il*self.iu != 0:
            # try:
            print(self.il - 1)
            if math.isnan(self.Ranges.at[self.il - 1, 'Displayed']):
                self.ax.axvspan(self.Ranges.at[self.il - 1, 'Lower Bound'],
                                self.Ranges.at[self.iu - 1, 'Upper Bound'],
                                alpha=0.1, edgecolor='k', linestyle='--')
                self.Ranges.at[self.il - 1, 'Displayed'] = 1

    def rangeremove(self, event):
        print("yeet")
        if event.key == 'delete':
            if not self.Ranges.empty:
                # self.ax.figure.canvas.mpl_disconnect(self.cid1)
                try:
                    self.Ranges.at[self.il - 1, 'Displayed'] = float('NaN')
                    self.il -= 1
                    self.iu -= 1
                    self.Ranges.drop(self.Ranges.index[-1], inplace=True)
                    Polys = self.ax.get_children()
                    Polys[len(self.Ranges.index)].remove()
                except IndexError:
                    self.Ranges.at[self.il - 1, 'Displayed'] = float('NaN')
                    self.il -= 1
                    self.iu -= 1
                    self.Ranges.drop(self.Ranges.index[0], inplace=True)
                    Polys = self.ax.get_children()
                    Polys[0].remove()
                    if self.Ranges == 'Empty DataFrame':
                        print('Range list is empty')
                # except NotImplementedError:
                #     Polys[len(self.Ranges.index)] = Polys(alpha=0)[len(self.Ranges.index)]
                finally:
                    pass

    def finishplot(self, event):
        if event.key == 'enter':
            self.Ranges.to_csv('Ranges/{}.csv'.format(self.key), index=False, encoding='utf-8',
                               columns=['Lower Bound', 'LowerIndex', 'Upper Bound', 'UpperIndex'])
            plt.close()
        if event.key == 'escape':
            plt.close()
        # print('\n')
        # print('Ranges are \n {}'.format(self.Ranges))


class DataRead:
    def __init__(self, dataName, calib=None):
        self.IndependentVariable = "Energy (KeV)"
        self.DependentVariable = "Counts (a.u)"
        self.files = []
        self.xrange = []
        self.yrange = []
        self.xranges = {}
        self.yranges = {}
        self.datpath = 'Data'
        self.rangepath = 'Ranges'
        print(os.getcwd())
        # os.chdir('{}'.format(self.datpath))
        self.datafilename = dataName
        self.dataset = self.datafilename.split('.')[0]
        self.dataset = pd.read_csv('Data/{}.csv'.format(self.datafilename), header=None, delimiter=',',
                                   names=[self.IndependentVariable, self.DependentVariable],
                                   float_precision='round_trip')
        if calib is not None:
            self.dataset[self.IndependentVariable] = calibrate(calib)['Value']
        self.rangename = pd.read_csv('Ranges/{}.csv'.format(self.datafilename))

    def range(self):
        for i in range(0, len(self.rangename)):
            self.xrange.append((self.dataset[self.IndependentVariable][self.rangename['LowerIndex'][i]
                                                                       :self.rangename['UpperIndex'][i] + 1]).values)
            self.yrange.append((self.dataset[self.DependentVariable][self.rangename['LowerIndex'][i]
                                                                     :self.rangename['UpperIndex'][i] + 1]).values)
        for i in range(0, len(self.xrange)):
            self.xranges[i] = self.xrange[i]
            self.yranges[i] = self.yrange[i]
        return self.xranges, self.yranges, self.xrange, self.yrange

    def singleplot(self):
        os.chdir('{}'.format(self.rangepath))

        self.range()
        for i in range(0, len(self.xranges)):
            def lingauss(xvar, co_a, co_b, co_c, co_d, co_e):
                return co_a * np.exp(-((xvar - co_b) ** 2) / (2 * (co_c ** 2))) + co_d * xvar + co_e

            try:
                initial = [np.max(self.yranges[i]),
                           self.xranges[i][(np.where(self.yranges[i] == np.max(self.yranges[i])))[0]],
                           np.std(self.xranges[i]), -0.1, 100]

                popt, pcov = curve_fit(lingauss, self.xranges[i], self.yranges[i],
                                       initial, sigma=np.sqrt(self.yranges[i]), absolute_sigma=True, maxfev=100000)
                plt.plot(self.xranges[i], lingauss(self.xranges[i], *popt))
                print(popt)
            except TypeError:
                continue

    def multiplot(self):
        os.chdir('{}'.format(self.rangepath))
        self.range()
        for i in range(0, len(self.xrange)):
            def lingauss(xvar, co_a, co_b, co_c, co_d, co_e):
                return co_a * np.exp(-((xvar - co_b) ** 2) / (2 * (co_c ** 2))) + co_d * xvar + co_e

            try:
                initial = [np.max(self.yranges[i]), self.xranges[i][(np.where(self.yranges[i] == np.max(self.yranges[i])))[0]], np.std(self.xranges[i]), -0.1,
                           100]
                popt, pcov = curve_fit(lingauss, self.xranges[i], self.yranges[i], initial, sigma=np.sqrt(self.yranges[i]), absolute_sigma=True, maxfev=100000)
            except TypeError:
                continue
            fig = plt.figure()
            fig.subplots_adjust(hspace=0.3, wspace=0)
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.plot(self.xranges[i], lingauss(self.xranges[i], *popt), antialiased=True)
            ax1.plot(self.xranges[i], self.yranges[i], '.', color='#1c1c1c')
            dely = np.sqrt(self.yranges[i])
            ax1.fill_between(self.xranges[i], lingauss(self.xranges[i], *popt) - dely, lingauss(self.xranges[i], *popt) + dely, color="#ABABAB")
            ax1.grid(color='k', linestyle='--', alpha=0.2)
            plt.title('Peak with 1 sigma error bands')

            ax2 = fig.add_subplot(2, 2, 2)
            ax2.plot(self.xranges[i], self.yranges[i] - lingauss(self.xranges[i], *popt), '.', antialiased=True)
            ax2.grid(color='k', linestyle='--', alpha=0.2)
            plt.title('Residuals')

            ax3 = fig.add_subplot(2, 2, 3)
            ax3.plot(self.xranges[i],
                     ((self.yranges[i] - lingauss(self.xranges[i], *popt)) ** 2) / (np.sqrt(self.yranges[i])) ** 2,
                     '.', antialiased=True)
            ax3.grid(color='k', linestyle='--', alpha=0.2)
            plt.title('Normalised residuals')

            ax4 = fig.add_subplot(2, 2, 4)
            n, bins, patches = ax4.hist(self.yranges[i] - lingauss(self.xranges[i], *popt), bins=10)

            mdl = GaussianModel()
            bin_centre = []
            for t in range(0, len(bins) - 1):
                bin_centre.append((bins[t + 1] + bins[t]) / 2)
            bin_centre2 = np.asarray(bin_centre)
            pars = mdl.guess(n, x=bin_centre2)
            result2 = mdl.fit(n, pars, x=bin_centre2)
            corr_coeff = 1 - result2.residual.var() / np.var(n)
            at = AnchoredText("$R^2 = {:.3f}$".format(corr_coeff),
                              prop=dict(size=10), frameon=True,
                              loc=2,
                              )
            ax4.add_artist(at)
            ax4.plot(bin_centre2, result2.best_fit, antialiased=True)
            ax4.grid(color='k', linestyle='--', alpha=0.2)
            plt.title('Residual histogram')

            fig.tight_layout()
            fig.set_size_inches(16.5, 10.5)

            plt.show()


def DataConvert(datafolder, destinationfolder):
    for filename in os.listdir(datafolder):
        print(filename)
        name = filename.split('.')
        nam3 = name[0]
        if name[1] == 'txt':
            dframename1 = pd.read_csv(datafolder + "/" + filename, header=None, names=['Channel', 'Counts'], delim_whitespace=True)
            print(dframename1)
            print(nam3)
            dframename1.to_csv(destinationfolder + "/" + '{}.csv'.format(nam3), header=False, index=False)
        elif name[1] == 'csv':
            dframename2 = pd.read_csv(datafolder + "/" + filename, header=None, delimiter=',', usecols=[9, 10], engine='c')
            print(dframename2)
            print(nam3)
            dframename2.to_csv(destinationfolder + "/" + '{}.csv'.format(nam3), header=False, index=False)
        elif name[1] == 'xlsx':
            dframename3 = pd.read_excel(datafolder + "/" + filename, usecols='J:K')
            dframename3.to_csv(destinationfolder + "/" + '{}.csv'.format(nam3), header=False, index=False)


# print(DataRead(1, 1).dataset)
def doot(dataName, calib=None):
    fig, ax = plt.subplots()
    print(os.getcwd())
    dataRead = DataRead(dataName, calib)
    figure2, = ax.plot(dataRead.dataset[dataRead.IndependentVariable],
                       dataRead.dataset[dataRead.DependentVariable], 'x',
                       antialiased='True', color='#1c1c1c', mew=1.0, markersize=2.5)
    thing = RangeTool(ax, dataRead.dataset, DataRead(dataName,  calib).datafilename, figure2)
    plt.ylabel('Counts (a.u)')
    plt.xlabel('Energy (KeV)')
    ax.grid(color='k', linestyle='--', alpha=0.2)
    fig.set_size_inches(16.5, 10.5)

    print('----{}----'.format(time.time() - start_time))
    print('\n')
    if os.path.isfile('Ranges/{}.csv'.format(dataName)):
        dataRead.singleplot()
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    plt.legend()
    plt.show()


DataConvert('Data', 'Data')
fig, ax23 = plt.subplots()
name = "ShieldedTrinititeFIN2"
data = pd.read_csv("Data/{}.csv".format(name), names=["Energy (channels)", "Counts (a.u)"])
calibInfo = fullCalibrate(g4)
data["Energy (KeV)"] = calibInfo[3](data["Energy (channels)"], *calibInfo[1])
fig2 = ax23.plot(data["Energy (KeV)"], data["Counts (a.u)"])
thing = RangeTool(ax23, data, fig2, name)
plt.show()
# for key in Data:
#     fig, ax = plt.subplots()
#     figure2, = ax.plot(Data[key][IndependentVariable], Data[key][DependentVariable], '.')
#     thing = RangeTool(ax, Data, key, figure2)
#     fig.set_size_inches(16.5, 10.5)
#     figManager = plt.get_current_fig_manager()
#     figManager.window.showMaximized()
#     print('----{}----'.format(time.time() - start_time))
#     plt.show()
