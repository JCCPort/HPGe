import numpy as np
import os
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from lmfit.models import GaussianModel
import time
import HPGe_Calibration as clb
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

import datetime

today = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
print(today)

start_time = time.time()
os.chdir('C:\\Users\\Josh\\Desktop\HPGe2\ReadableData')

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

    def __init__(self, ax, data, key, figure2):
        self.ax = figure2.axes
        self.data = data
        self.key = key
        self.figure2 = figure2
        self.lx = ax.axhline(color='k')  # the horiz line
        self.ly = ax.axvline(color='k')  # the vert line
        self.lowers = np.array([])
        self.uppers = np.array([])
        self.IndependentVariable = "Energy (KeV)"
        self.DependentVariable = "Counts (a.u)"
        self.x = data[self.IndependentVariable]
        self.y = data[self.DependentVariable]
        self.ax.set_xlim(np.min(self.x), np.max(self.x))
        width = np.max(self.x) - np.min(self.x)
        height = np.max(self.y) - np.min(self.y)
        self.ax.set_ylim(np.min(self.y)-0.1*height, np.max(self.y)+0.1*height)
        # text location in axes coords
        self.txt = ax.text(0.7, 0.9, '', transform=ax.transAxes)
        self.cid1 = figure2.figure.canvas.mpl_connect('key_press_event', self.rangeselect)
        self.cid2 = figure2.figure.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        self.cid3 = figure2.figure.canvas.mpl_connect('key_press_event', self.rangeremove)
        self.cid4 = figure2.figure.canvas.mpl_connect('key_press_event', self.finishplot)
        self.Ranges = pd.DataFrame(columns=['Lower Bound', 'LowerIndex', 'Upper Bound', 'UpperIndex', 'Displayed'])
        self.il = 0
        self.iu = 0
        self.t = 0

    def __call__(self, event):
        print('click', event)
        print(event.xdata, event.ydata)
        if event.inaxes != self.figure2.axes: return

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
        #print('{},{}'.format(event.xdata, event.ydata))
        self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
        self.figure2.figure.canvas.draw_idle()
        #print('x=%1.2f, y=%1.2f' % (x, y))

    def rangeselect(self, event):
        x = event.xdata
        indx = min(np.searchsorted(self.x, [x])[0], len(self.x) - 1)
        x = self.x[indx]
        if event.key == 'tab':
            self.Ranges.at[self.il, 'Lower Bound'] = x
            self.Ranges.at[self.il, 'LowerIndex'] = indx
            self.il += 1
        if event.key == 'shift':
            self.Ranges.at[self.iu, 'Upper Bound'] = x
            self.Ranges.at[self.iu, 'UpperIndex'] = indx
            self.iu += 1
        if self.il == self.iu:
            try:
                if math.isnan(self.Ranges.at[self.il-1, 'Displayed']):
                    self.ax.axvspan(self.Ranges.at[self.il-1, 'Lower Bound'],
                                    self.Ranges.at[self.iu-1, 'Upper Bound'],
                                    alpha=0.1, edgecolor='k', linestyle='--')
                    self.Ranges.at[self.il-1, 'Displayed'] = 1
            except ValueError:
                pass
        self.cid3 = self.figure2.figure.canvas.mpl_connect('key_press_event', self.rangeremove)

    def rangeremove(self, event):
        if event.key == 'delete':
            if not self.Ranges.empty:
                self.figure2.figure.canvas.mpl_disconnect(self.cid1)
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
                self.cid1 = self.figure2.figure.canvas.mpl_connect('key_press_event', self.rangeselect)

    def finishplot(self, event):
        if event.key == 'enter':
            os.chdir('C:\\Users\\Josh\\Desktop\HPGe2\Ranges')
            self.Ranges.to_csv('{}.csv'.format(self.key), index=False, encoding='utf-8',
                               columns=['Lower Bound', 'LowerIndex', 'Upper Bound', 'UpperIndex'])
            plt.close()
            os.chdir('C:\\Users\\Josh\\Desktop\HPGe2\ReadableData')
        if event.key == 'escape':
            plt.close()
        #print('\n')
        #print('Ranges are \n {}'.format(self.Ranges))


Data = {}


class DataRead:
    def __init__(self, peak, run, calib=None):
        self.IndependentVariable = "Energy (KeV)"
        self.DependentVariable = "Counts (a.u)"
        self.peak = peak
        self.run = run
        self.files = []
        self.xrange = []
        self.yrange = []
        self.xranges = {}
        self.yranges = {}
        self.datpath = 'ReadableData'
        self.rangepath = 'Ranges'
        os.chdir('C:\\Users\\Josh\\Desktop\HPGe2\{}'.format(self.datpath))
        self.datafilename = '{}_{}'.format(self.peak, self.run)
        self.dataset = self.datafilename.split('.')[0]
        self.dataset = pd.read_csv('{}.csv'.format(self.datafilename), header=None, delimiter=',',
                                   names=[self.IndependentVariable, self.DependentVariable],
                                   float_precision='round_trip')
        if calib is not None:
            self.dataset[self.IndependentVariable] = clb.calibrate(calib)['Value']

    def range(self):
        for i in range(0, len(self.rangename)):
            self.xrange.append((self.dataset[self.IndependentVariable][self.rangename['LowerIndex'][i]
                                                                       :self.rangename['UpperIndex'][i]+1]).values)
            self.yrange.append((self.dataset[self.DependentVariable][self.rangename['LowerIndex'][i]
                                                                     :self.rangename['UpperIndex'][i]+1]).values)
        for i in range(0, len(self.xrange)):
            self.xranges[i] = self.xrange[i]
            self.yranges[i] = self.yrange[i]
        return self.xranges, self.yranges, self.xrange, self.yrange

    def singleplot(self):
        os.chdir('C:\\Users\\Josh\\Desktop\HPGe2\{}'.format(self.rangepath))
        self.rangename = pd.read_csv('{}.csv'.format(self.datafilename))
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
        os.chdir('C:\\Users\\Josh\\Desktop\HPGe2\{}'.format(self.rangepath))
        self.rangename = pd.read_csv('{}.csv'.format(self.datafilename))
        self.range()
        for i in range(0, len(self.xrange)):
            def lingauss(xvar, co_a, co_b, co_c, co_d, co_e):
                return co_a * np.exp(-((xvar - co_b) ** 2) / (2 * (co_c ** 2))) + co_d * xvar + co_e
            try:
                initial = [np.max(self.yranges[i]), self.xranges[i][(np.where(self.yranges[i] == np.max(self.yranges[i])))[0]], np.std(self.xranges[i]), -0.1, 100]
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

            ax2 = fig.add_subplot(2,2,2)
            ax2.plot(self.xranges[i], self.yranges[i]-lingauss(self.xranges[i], *popt), '.', antialiased=True)
            ax2.grid(color='k', linestyle='--', alpha=0.2)
            plt.title('Residuals')

            ax3 = fig.add_subplot(2, 2, 3)
            ax3.plot(self.xranges[i],
                     ((self.yranges[i]-lingauss(self.xranges[i], *popt))**2)/(np.sqrt(self.yranges[i]))**2,
                     '.', antialiased=True)
            ax3.grid(color='k', linestyle='--', alpha=0.2)
            plt.title('Normalised residuals')

            ax4 = fig.add_subplot(2, 2, 4)
            n, bins, patches = ax4.hist(self.yranges[i]-lingauss(self.xranges[i], *popt), bins=10)

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
    os.chdir('C:\\Users\\Josh\\Desktop\HPGe2\{}'.format(datafolder))
    for filename in os.listdir(os.getcwd()):
        print(filename)
        name = filename.split('.')[0]
        nam2 = name.split('_')
        nam3 = nam2[0] + '_' + ('R'+nam2[1])
        if filename.split('.')[1] == 'txt':
            dframename1 = pd.read_csv(filename, header=None, names=['Channel', 'Counts'], delim_whitespace=True)
            print(dframename1)
            print(nam3)
            os.chdir('C:\\Users\\Josh\\Desktop\HPGe2\{}'.format(destinationfolder))
            dframename1.to_csv('{}.csv'.format(nam3), header=False, index=False)
            os.chdir('C:\\Users\\Josh\\Desktop\HPGe2\{}'.format(datafolder))
        elif filename.split('.')[1] == 'csv':
            dframename2 = pd.read_csv(filename, header=None, delimiter=',', usecols=[9, 10], engine='c')
            print(dframename2)
            print(nam3)
            os.chdir('C:\\Users\\Josh\\Desktop\HPGe2\{}'.format(destinationfolder))
            dframename2.to_csv('{}.csv'.format(nam3), header=False, index=False)
            os.chdir('C:\\Users\\Josh\\Desktop\HPGe2\{}'.format(datafolder))
        elif filename.split('.')[1] == 'xlsx':
            dframename3 = pd.read_excel(filename, usecols='J:K')
            # print(dframename2)
            os.chdir('C:\\Users\\Josh\\Desktop\HPGe2\{}'.format(destinationfolder))
            dframename3.to_csv('{}.csv'.format(nam3), header=False, index=False)
            os.chdir('C:\\Users\\Josh\\Desktop\HPGe2\{}'.format(datafolder))


#print(DataRead(1, 1).dataset)
def doot(peak, run, calib=None):
    fig, ax = plt.subplots()
    figure2, = ax.plot(DataRead(peak, run, calib).dataset[DataRead(peak, run, calib).IndependentVariable],
                       DataRead(peak, run, calib).dataset[DataRead(peak, run, calib).DependentVariable], 'x',
                       antialiased='True', color='#1c1c1c', mew=1.0, markersize=2.5)
    thing = RangeTool(ax, DataRead(peak, run, calib).dataset, DataRead(peak, run, calib).datafilename, figure2)
    plt.ylabel('Counts (a.u)')
    plt.xlabel('Energy (KeV)')
    ax.grid(color='k', linestyle='--', alpha=0.2)
    fig.set_size_inches(16.5, 10.5)
    
    print('----{}----'.format(time.time() - start_time))
    print('\n')
    if os.path.isfile('C:\\Users\\Josh\\Desktop\HPGe2\Ranges\{}_{}.csv'.format(peak, run)):
        DataRead(peak, run, calib).singleplot()
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    plt.legend()
    plt.show()


def multidoot():
    fig, axes = plt.subplots(nrows=2, ncols=2)

# doot('mystery', '1')
# doot('sodium22', 'R071117', clb.g2)
# doot('sodium22', 'R071117')

# DataConvert('Data', 'ReadableData')
# for key in Data:
#     fig, ax = plt.subplots()
#     figure2, = ax.plot(Data[key][IndependentVariable], Data[key][DependentVariable], '.')
#     thing = RangeTool(ax, Data, key, figure2)
#     fig.set_size_inches(16.5, 10.5)
#     figManager = plt.get_current_fig_manager()
#     figManager.window.showMaximized()
#     print('----{}----'.format(time.time() - start_time))
#     plt.show()

print('Complete')
