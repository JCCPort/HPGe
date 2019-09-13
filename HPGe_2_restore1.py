import datetime
import math
import os
import time

import ComptonChecker as comp
import HPGe_Calibration as clb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uncertainties
from lmfit.models import GaussianModel
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from scipy.optimize import curve_fit

today = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
print(today)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [
    r'\usepackage{siunitx}',
    r'\usepackage{isotope}',
    r'\sisetup{detect-all}',
    r'\usepackage{fourier}',
    ]
plt.rcParams['ps.usedistiller'] = 'xpdf'
plt.rcParams['ps.distiller.res'] = '16000'

start_time = time.time()
os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2\Vals')
# analmystlist = np.genfromtxt('vals_mystery_R101117g2_20180314_013535.csv', delimiter=',', dtype='float')
analmystlist = np.genfromtxt('vals_mystery_R101117g2_20180325_002220.csv', delimiter=',', dtype='float')
# analmystlist = np.genfromtxt('vals_shieldedbackground_R061117g2_20180314_014320.csv', delimiter=',', dtype='float')
# analmystlist = np.genfromtxt('vals_shieldedbackground_R061117g2_20180321_151920.csv', delimiter=',', dtype='float')
# analmystlist = np.genfromtxt('vals_shieldedbackground_R061117g2_20180325_003450.csv', delimiter=',', dtype='float')

# analmystlist = np.genfromtxt('vals_unshielded_R141117g2_20180314_014304.csv', delimiter=',', dtype='float')
# analmystlist = np.genfromtxt('vals_unshielded_R141117g2_20180325_002834.csv', delimiter=',', dtype='float')

os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2\Ranges')

os.chdir('C:\\Users\\Josh\\Documents\\git\\HPGe2\\NewestVals')
# analmystlist_mys = np.genfromtxt('vals_mystery_R101117g2.csv', delimiter=',', dtype='float')
# analmystlist_sh = np.genfromtxt('vals_shieldedbackground_R061117g2.csv', delimiter=',', dtype='float')
# analmystlist_unsh = np.genfromtxt('vals_unshielded_R141117g2.csv', delimiter=',', dtype='float')
os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2\Ranges')

os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2')
xrays2 = np.genfromtxt('xrays2.csv', delimiter=',', dtype=('U60', 'float', 'float', 'float', 'float'))
energylist = np.genfromtxt('Energies7.csv', delimiter=',', dtype=('U60', 'float', 'float', 'float', 'float', 'U60'))
# energyescapelist = np.genfromtxt('Energies7_escapes.csv', delimiter=',', dtype=('U60', 'float', 'float', 'float', 'float'))

os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2\ReadableData')


def kleinnishina(E, theta):
    m0 = 510.9989461
    c = 2.99792458e8
    E_prime = E / (1 + (E / (m0)) * (1 - np.cos(theta)))
    return E_prime


class RangeTool(object):
    """
    Like Cursor but the crosshair snaps to the nearest x,y point
    For simplicity, I'm assuming x is sorted
    """

    def __init__(self, ax, data, key, figure2, calib=None):
        self.ax = figure2.axes
        self.calib = calib
        self.data = data
        self.key = key
        self.figure2 = figure2
        self.lx = ax.axhline(color='k')  # the horiz line
        self.ly = ax.axvline(color='k')  # the vert line
        self.lowers = np.array([])
        self.uppers = np.array([])
        self.IndependentVariable = "Energy (KeV)"
        self.DependentVariable = "Counts "
        self.x = data[self.IndependentVariable]
        self.y = data[self.DependentVariable]
        self.ax.set_xlim(np.min(self.x), np.max(self.x))
        width = np.max(self.x) - np.min(self.x)
        height = np.max(self.y) - np.min(self.y)
        self.ax.set_ylim(np.min(self.y) - 0.1 * height, np.max(self.y) + 0.1 * height)
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
        print('{}'.format(indx))
        self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
        self.figure2.figure.canvas.draw_idle()
        # print('x=%1.2f, y=%1.2f' % (x, y))

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
                if math.isnan(self.Ranges.at[self.il - 1, 'Displayed']):
                    self.ax.axvspan(self.Ranges.at[self.il - 1, 'Lower Bound'], self.Ranges.at[self.iu - 1, 'Upper Bound'], alpha=0.1, edgecolor='k',
                                    linestyle='--')
                    self.Ranges.at[self.il - 1, 'Displayed'] = 1
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
            os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2\Ranges')
            self.Ranges.to_csv('{}{}.csv'.format(self.key, self.calib.__name__), index=False, encoding='utf-8',
                               columns=['Lower Bound', 'LowerIndex', 'Upper Bound', 'UpperIndex'])
            plt.close()
            os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2\ReadableData')
        if event.key == 'escape':
            plt.close()
        # print('\n')
        # print('Ranges are \n {}'.format(self.Ranges))


Data = {}


class DataRead:
    def __init__(self, peak, run, calib=None):
        self.IndependentVariable = "Energy (KeV)"
        self.DependentVariable = "Counts "
        self.calib = calib
        self.peak = peak
        self.run = run
        self.files = []
        self.xrange = []
        self.yrange = []
        self.xranges = {}
        self.yranges = {}
        self.xrange2 = []
        self.yrange2 = []
        self.xranges2 = {}
        self.yranges2 = {}
        self.datpath = 'ReadableData'
        self.rangepath = 'Ranges'
        os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2\{}'.format(self.datpath))
        self.datafilename = '{}_{}'.format(self.peak, self.run)
        self.dataset = self.datafilename.split('.')[0]
        self.dataset = pd.read_csv('{}.csv'.format(self.datafilename), header=None, delimiter=',',
                                   names=[self.IndependentVariable, self.DependentVariable], float_precision='round_trip')
        if calib is not None:
            self.dataset[self.IndependentVariable] = clb.calibrate(calib)[0]['Value']

    def range(self):
        for i in range(0, len(self.rangename)):
            self.xrange.append((self.dataset[self.IndependentVariable][self.rangename['LowerIndex'][i]:self.rangename['UpperIndex'][i] + 1]).values)
            self.yrange.append((self.dataset[self.DependentVariable][self.rangename['LowerIndex'][i]:self.rangename['UpperIndex'][i] + 1]).values)
            self.xrange2.append((self.dataset[self.IndependentVariable][
                                 self.rangename['LowerIndex'][i] - 5:self.rangename['UpperIndex'][i] + 6]).values)
            self.yrange2.append((self.dataset[self.DependentVariable][
                                 self.rangename['LowerIndex'][i] - 5:self.rangename['UpperIndex'][i] + 6]).values)
        for i in range(0, len(self.xrange)):
            self.xranges[i] = self.xrange[i]
            self.yranges[i] = self.yrange[i]
            self.xranges2[i] = self.xrange2[i]
            self.yranges2[i] = self.yrange2[i]
        return self.xranges, self.yranges, self.xrange, self.yrange

    def singleplot(self):
        os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2\{}'.format(self.rangepath))
        try:
            self.rangename = pd.read_csv('{}.csv'.format(self.datafilename), engine='c')
        except AttributeError or FileNotFoundError:
            self.rangename = pd.read_csv('{}.csv'.format(self.datafilename), engine='c')
        self.range()
        plotvals = pd.DataFrame(columns=['ChiSq', 'Reduced ChiSq', 'Mean', 'Mean Error', 'Amplitude'])
        calib_opt = clb.calibrate(clb.g2)[1]
        for i in range(0, len(self.xranges)):
            def lingauss(xvar, co_a, co_b, co_c, co_d, co_e):
                return co_a * np.exp(-((xvar - co_b) ** 2) / (2 * (co_c ** 2))) + co_d * xvar + co_e

            print('donger')
            initial = (
                np.max(self.yranges[i]), self.xranges[i][(np.where(self.yranges[i] == np.max(self.yranges[i])))[0]][0], np.std(self.xranges[i]), -0.1, 100)
            bounds = ([np.max(self.yranges[i]) * 0.1, np.min(self.xranges[i]) * 0.8,
                       -np.std(self.xranges[i]) * 0.5, -500, -np.inf],
                      [np.max(self.yranges[i]) * 1.5, np.max(self.xranges[i]) * 1.2,
                       np.std(self.xranges[i]) * 3, 500, np.inf])
            sigmas = np.sqrt(abs(self.yranges[i] - 0.9 * np.min(self.yranges[i])))
            popt, pcov = curve_fit(lingauss, self.xranges[i], self.yranges[i],
                                   initial, bounds=bounds, sigma=sigmas,
                                   absolute_sigma=True, maxfev=10000000000)
            err = np.sqrt(abs((np.diag(pcov))))
            pointchisq = []

            def baseline(v):
                return v * popt[3] + popt[4]

            print(popt[1])
            for k in range(0, len(self.xranges[i])):
                print('Baseline count {}'.format(baseline(self.xranges[i][k])))
                print('Total count {}'.format(self.yranges[i][k]))
                print(
                        'Peak with baseline removed count {}'.format(self.yranges[i][k] - baseline(self.xranges[i][k]) + 4))
                print('\n')
                corr = baseline(self.xranges[i][k])
                gap = 0 - (self.yranges[i][k] - baseline(self.xranges[i][k])) + 0.1
                # alt_err = (corr**2)*self.yranges[i][k] + corr*(self.yranges[i][k])**2 + corr + self.yranges[i][k] - 2*((corr**2) + self.yranges[i][k]**2 - (self.yranges[i][k]-corr)**2)
                if self.yranges[i][k] - baseline(self.xranges[i][k]) <= 0:
                    print('gap: {}'.format(gap))
                    pointchisq.append((((self.yranges[i][k] - corr) - (
                            lingauss(self.xranges[i][k], *popt) - corr)) ** 2 / (
                                           np.sqrt(self.yranges[i][k] - corr + gap)) ** 2))
                else:
                    pointchisq.append((((self.yranges[i][k] - corr) - (
                            lingauss(self.xranges[i][k], *popt) - corr)) ** 2 / (
                                           np.sqrt(self.yranges[i][k] - corr)) ** 2))
            bl_avg = np.mean(baseline(self.xranges[i]))
            print(bl_avg)
            chisq1 = sum(pointchisq)
            print(chisq1)
            print('amp {}, max count {}'.format(popt[0], np.max(self.yranges[i]) - bl_avg))
            print('\n \n \n')
            redchisq1 = chisq1 / (len(self.xranges[i]) - 1)

            # try:
            pointerr = clb.point_errors(self.calib, popt[1])
            centre_index = (self.rangename['LowerIndex'][i] + self.rangename['UpperIndex'][i]) / 2.
            point_resolution = clb.g2(centre_index + 0.5, *calib_opt) - clb.g2(centre_index - 0.5, *calib_opt)
            meanerr = np.sqrt(err[1] ** 2 + pointerr ** 2 + point_resolution ** 2)
            # except TypeError:
            #     meanerr = np.sqrt(err[1]**2 + 0.5**2)
            # meanerr = np.sqrt(err[1]**2 + (popt[1]*5.89813289e-14)**2 + (7.28147622e-10)**2 + 0.17908847**2)
            #
            # plt.vlines(popt[1], ymin=1e0, ymax=1e5, linestyles='dashed', colors='r', linewidth=1.2, antialiased=True)
            # plt.text(popt[1], 1e5, '{:.2f}'.format(popt[1]), rotation=45, va='center', ha='center', bbox=dict(pad=0.1, ec='none', fc='w'), fontsize=32)
            plt.plot(np.linspace(self.xranges[i][0], self.xranges[i][-1], 500),
                     lingauss(np.linspace(self.xranges[i][0], self.xranges[i][-1], 500), *popt))
            if meanerr <= 2.5:
                plotvals = plotvals.append({'ChiSq': chisq1, 'Reduced ChiSq': redchisq1,
                                            'Mean':  popt[1], 'Mean Error': meanerr, 'Amplitude': popt[0]}, ignore_index=True)
        plt.show()
        plotvals['Amplitude'] = plotvals['Amplitude'].apply(lambda x: (x / sum(plotvals['Amplitude']) * 100))
        os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2\Vals')
        plotvals.to_csv('vals_{}_{}{}_{}.csv'.format(self.peak, self.run, self.calib.__name__, today), index=False,
                        header=False)
        os.chdir('C:\\Users\\Josh\\Documents\\git\\HPGe2\\NewestVals')
        plotvals.to_csv('vals_{}_{}{}.csv'.format(self.peak, self.run, self.calib.__name__), index=False, header=False)
        os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2\{}'.format(self.rangepath))

    def multiplot(self):
        os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2\{}'.format(self.rangepath))
        self.rangename = pd.read_csv('{}.csv'.format(self.datafilename))
        self.range()
        for i in range(0, len(self.xrange)):

            def lingauss(xvar, co_a, co_b, co_c, co_d, co_e):
                return co_a * np.exp(-((xvar - co_b) ** 2) / (2 * (co_c ** 2))) + co_d * xvar + co_e

            def gauss(xvar, co_a, co_b, co_c):
                return co_a * np.exp(-((xvar - co_b) ** 2) / (2 * (co_c ** 2)))

            initial = (
                np.max(self.yranges[i]), self.xranges[i][(np.where(self.yranges[i] == np.max(self.yranges[i])))[0]][0],
                np.std(self.xranges[i]), -0.1, 100)
            bounds = ([np.max(self.yranges[i]) * 0.1, np.min(self.xranges[i]) * 0.8,
                       -np.std(self.xranges[i]) * 0.5, -500, -np.inf],
                      [np.max(self.yranges[i]) * 1.5, np.max(self.xranges[i]) * 1.2,
                       np.std(self.xranges[i]) * 3, 500, np.inf])
            popt, pcov = curve_fit(lingauss, self.xranges[i], self.yranges[i],
                                   initial, bounds=bounds, sigma=np.sqrt(self.yranges[i]),
                                   absolute_sigma=True, maxfev=10000000000)
            if popt[1] > 960 and popt[1] < 980:
                size = 24
                size2 = 20
                fig = plt.figure(figsize=(12, 10))
                fig.subplots_adjust(top=0.948,
                                    bottom=0.102,
                                    left=0.102,
                                    right=0.989,
                                    hspace=0.286,
                                    wspace=0.402)
                plt.rc('xtick', labelsize=22)
                plt.rc('ytick', labelsize=22)
                ax1 = fig.add_subplot(2, 2, 1)
                values = np.linspace(np.min(self.xranges[i]), np.max(self.xranges[i]), 200)
                ax1.plot(values, lingauss(values, *popt), antialiased=True)
                ax1.plot(self.xranges2[i], self.yranges2[i], 'o', markerfacecolor="None", color='#050505',
                         markersize=3.5,
                         mew=1.4)
                dely = np.sqrt(lingauss(values, *popt))
                ax1.fill_between(values, lingauss(values, *popt) - dely, lingauss(values, *popt) + dely,
                                 color="#ABABAB",
                                 antialiased=True, alpha=0.4)
                ax1.grid(color='k', linestyle='--', alpha=0.3, antialiased=True)
                # plt.title(r'$\textit{Peak with 1 sigma error bands}$', fontsize=size2)
                plt.xlabel(r'$\textit{Energy} \textit{ (KeV)}$', fontsize=size)
                plt.ylabel(r'$\textit{Counts}$', fontsize=size)

                ax2 = fig.add_subplot(2, 2, 2)
                ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 1000))
                ax2.plot(self.xranges[i], self.yranges[i] - lingauss(self.xranges[i], *popt), 'o',
                         markerfacecolor="None", antialiased=True, color='#050505', markersize=3.5, mew=1.4)
                ax2.grid(color='k', linestyle='--', alpha=0.3, antialiased=True)
                # plt.title(r'$Residuals$', fontsize=size2)
                plt.ylabel(r'$C_m - C_f$', fontsize=size)
                plt.xlabel(r'$\textit{Energy} \textit{ (KeV)}$', fontsize=size)

                ax3 = fig.add_subplot(2, 2, 3)
                ax3.ticklabel_format(style='sci', axis='y', scilimits=(0, 1000))
                ax3.plot(self.xranges[i],
                         ((self.yranges[i] - lingauss(self.xranges[i], *popt)) ** 2) / (np.sqrt(self.yranges[i])) ** 2,
                         'o',
                         markerfacecolor="None", antialiased=True,
                         color='#050505', markersize=3.5, mew=1.4)
                ax3.grid(color='k', linestyle='--', alpha=0.3, antialiased=True)
                # plt.title(r'$\textit{Normalised residuals}$', fontsize=size2)
                plt.xlabel(r'$\textit{Energy} \textit{ (KeV)}$', fontsize=size)
                plt.ylabel(r'$(C_m - C_f)^2 \textbackslash \sigma^2$', fontsize=size)

                ax4 = fig.add_subplot(2, 2, 4)
                n, bins, patches = ax4.hist(self.yranges[i] - lingauss(self.xranges[i], *popt),
                                            bins=len(self.xranges[i]), antialiased=True, color="b",
                                            alpha=0.5, rwidth=0.8,
                                            edgecolor='#050505')

                mdl = GaussianModel()
                bin_centre = []
                for t in range(0, len(bins) - 1):
                    bin_centre.append((bins[t + 1] + bins[t]) / 2)
                bin_centre2 = np.asarray(bin_centre)
                pars = mdl.guess(n, x=bin_centre2)
                print(bin_centre2)
                # pars.add('center', value=np.median(bin_centre2),
                #            min=np.min(bin_centre2) * 0.7,
                #            max=np.max(bin_centre2) * 1.3)
                # pars.add('amplitude',
                #          min=0,
                #          max=100)
                result2 = mdl.fit(n, pars, x=bin_centre2)
                print(result2.fit_report())
                corr_coeff = 1 - result2.residual.var() / np.var(n)
                at = AnchoredText("$R^2 = {:.3f}$".format(corr_coeff),
                                  prop=dict(size=20), frameon=True,
                                  loc=2,
                                  )
                ax4.add_artist(at)
                xes = np.linspace(np.min(bin_centre2), np.max(bin_centre2), 100)
                if abs(corr_coeff) > 0.5:
                    y = mdl.eval(x=xes, params=result2.params)
                    ax4.plot(xes, y, antialiased=True)
                ax4.grid(color='k', linestyle='--', alpha=0.2)
                plt.xlabel(r'$C_m - C_f$', fontsize=size)
                plt.ylabel(r'$\textit{Counts}$', fontsize=size)
                # plt.title(r'$\textit{Residual histogram}$', fontsize=size2)

                fig_manager = plt.get_current_fig_manager()
                fig_manager.window.showMaximized()
                # fig.suptitle((r'{} {}, Run: {}'.format(dictdict2[self.peak], labels[i], self.run)), fontsize=18)
                os.chdir('C:\\Users\Josh\Documents\\git\HPGe2\Figures')
                plt.savefig(
                        'NEW{}_{:.3f}_{}_{}resid.png'.format(self.peak, popt[1], self.run, self.calib.__name__),
                        dpi=600)
                os.chdir('C:\\Users\Josh\Documents\\git\HPGe2\Ranges')
                plt.close(fig)
                # plt.show()


energymatch = []
mystmatch = []
energymatchuncertainty = []
mystmatchuncertainty = []
nuclidematch = []
intensitymatch = []
chisss = []
formattednuclide = []


def overlap(point1, point2, uncert1, uncert2, sigma):
    lower1 = point1 - uncert1 * sigma
    upper1 = point1 + uncert1 * sigma
    lower2 = point2 - uncert2 * sigma
    upper2 = point2 + uncert2 * sigma
    index1 = np.where((lower1 <= point2) and (upper1 >= point2) or (lower2 <= point1) and (upper2 >= point1))
    if index1[0] == 0:
        return index1[0]


def datamatch(measured, known, sigma, type):
    os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2\Latex')
    for i in range(0, len(known)):
        for k in range(0, len(measured)):
            b = overlap(known[i][1], measured[k][2], known[i][2], measured[k][3], sigma)
            if b == 0:
                energymatch.append(known[i][1])
                mystmatch.append(measured[k][2])
                energymatchuncertainty.append(known[i][2])
                mystmatchuncertainty.append(measured[k][3])
                nuclidematch.append(known[i][0])
                intensitymatch.append(measured[k][4])
                chisss.append(measured[k][1])
                num1 = known[i][0].split('-')
                try:
                    num2 = num1[2].split(' ')
                    formattednuclide.append(r'$\isotope[{}][{}]{{{}}}$ {} {}'.format(num2[0], num1[0], num1[1], num2[1], num2[2]))
                except IndexError:
                    try:
                        formattednuclide.append(r'$\isotope[{}][{}]{{{}}}$'.format(num1[2], num1[0], num1[1]))
                    except IndexError:
                        formattednuclide.append(known[i][0])
                continue
    match_frame = pd.DataFrame({'Accepted energy': energymatch, 'Accepted uncertainty': energymatchuncertainty,
                                'Measured energy': mystmatch, 'Measured uncertainty': mystmatchuncertainty,
                                'Nuclide':         formattednuclide, 'Height': intensitymatch})
    os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2')
    match_frame.to_csv('{}_{}.csv'.format(type, today))
    os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2\Latex')

    with open('{}Match_{}.tex'.format(type, today), 'w') as texfile:
        texfile.write('\\documentclass[a4paper,twoside]{article}\n')
        texfile.write('\\usepackage[margin=0in]{geometry}\n')
        texfile.write('\\usepackage{isotope}\n')
        texfile.write('\\usepackage{ltxtable}\n')
        texfile.write('\\usepackage{ltablex}\n')
        texfile.write('\\pagenumbering{gobble}\n')
        texfile.write('\\begin{document}\n')
        texfile.write(r'\ {\def\arraystretch{0.9}\tabcolsep=3pt')
        texfile.write('\\\n')
        texfile.write(r'\begin{tabularx}{\textwidth}{XXXXX}')
        texfile.write('\\\n')
        row_fields = ('$E_{Measured}$ (keV)', '$E_{Accepted}$ (keV)',
                      'Source', 'RI', '$\chi^2$')
        texfile.write(' {} & {} & {} & {} & {} \\\\ \n'.format(row_fields[0], row_fields[1],
                                                               row_fields[2], row_fields[3],
                                                               row_fields[4]))
        texfile.write('\hline ')
        for i in range(0, len(energymatch)):
            mystvals = uncertainties.ufloat(mystmatch[i], mystmatchuncertainty[i])
            knownvals = uncertainties.ufloat(energymatch[i], energymatchuncertainty[i])
            texfile.write(' ${:.2uL}$ & ${:.2uL}$ & {} & {:.2f} & {:.3f} \\\\ \n'.format(mystvals, knownvals,
                                                                                         formattednuclide[i],
                                                                                         intensitymatch[i], chisss[i]))
        texfile.write('\hline')
        texfile.write('\\end{tabularx}\n')
        texfile.write('\\ \ }\n')
        texfile.write('\\end{document}\n')

    items = np.unique(nuclidematch)
    match_count = pd.DataFrame(columns=['Isotope', 'Number'])
    for item in items:
        num = nuclidematch.count(item)
        num1 = item.split('-')
        try:
            num2 = num1[2].split(' ')
            isotope = r'$\isotope[{}][{}]{{{}}}$ {} {}'.format(num1[0], num2[0], num1[1], num2[1], num2[2])
        except IndexError:
            try:
                isotope = r'$\isotope[{}][{}]{{{}}}$'.format(num1[0], num1[2], num1[1])
            except IndexError:
                isotope = item
            pass
        match_count = match_count.append({'Isotope': isotope, 'Number': num}, ignore_index=True)
    with open('{}MatchList_{}.tex'.format(type, today), 'w') as texfile:
        texfile.write('\\documentclass[a4paper,twoside]{article}\n')
        texfile.write('\\usepackage[margin=0in]{geometry}\n')
        texfile.write('\\usepackage{mathtools}\n')
        texfile.write('\\usepackage[math]{cellspace}\n')
        texfile.write('\\usepackage{isotope}\n')
        texfile.write('\\pagenumbering{gobble}\n')
        texfile.write('\\cellspacetoplimit 4pt\n')
        texfile.write('\\cellspacebottomlimit 4pt\n')
        texfile.write('\n')
        texfile.write(r'\setlength{\topmargin}{1in}')
        texfile.write('\n')
        texfile.write('\\begin{document}\n')
        texfile.write(r'''\ {\def\arraystretch{1.2}\tabcolsep=3pt''')
        texfile.write('\\ \n')
        texfile.write('\\begin{tabular}[h!]{cc} \n')
        texfile.write('\hline')
        row_fields = ('Isotope', 'Hits')
        texfile.write('\\ {} & {} \\\\ \n'.format(row_fields[0], row_fields[1]))
        texfile.write('\hline ')
        for i in range(0, len(match_count)):
            texfile.write(' {} & {} \\\\ \n'.format(
                    match_count['Isotope'][i], match_count['Number'][i]))
        texfile.write('\hline')
        texfile.write('\\end{tabular}\n')
        texfile.write('\\ }\n')
        texfile.write('\\end{document}\n')

    os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2\\NewestLatex')

    with open('{}Match.tex'.format(type), 'w') as texfile:
        texfile.write('\\documentclass[a4paper,twoside]{article}\n')
        texfile.write('\\usepackage[margin=0in]{geometry}\n')
        texfile.write('\\usepackage{isotope}\n')
        texfile.write('\\pagenumbering{gobble}\n')
        texfile.write('\\begin{document}\n')
        # texfile.write('\\hskip-5.3cm\n')
        texfile.write('\\begin{tabular}[ht!]{ccccc} \n')
        row_fields = ('$E_{Measured}$ (keV)', '$E_{Accepted}$ (keV)',
                      'Source', 'RI', '$\chi^2$')
        texfile.write('\\  {} & {} & {} & {} & {} \\\\ \n'.format(row_fields[0], row_fields[1],
                                                                  row_fields[2], row_fields[3],
                                                                  row_fields[4]))
        texfile.write('\hline ')
        for i in range(0, len(energymatch)):
            mystvals = uncertainties.ufloat(mystmatch[i], mystmatchuncertainty[i])
            knownvals = uncertainties.ufloat(energymatch[i], energymatchuncertainty[i])
            texfile.write('\\ ${:.2uL}$ & ${:.2uL}$ & {} & {:.2f} & {:.3f} \\\\ \n'.format(mystvals, knownvals,
                                                                                           nuclidematch[i],
                                                                                           intensitymatch[i],
                                                                                           chisss[i]))
        texfile.write('\hline')
        texfile.write('\\end{tabular}\n')
        texfile.write('\\end{document}\n')

    items = np.unique(nuclidematch)
    match_count = pd.DataFrame(columns=['Isotope', 'Number'])
    for item in items:
        num = nuclidematch.count(item)
        num1 = item.split('-')
        try:
            num2 = num1[2].split(' ')
            isotope = r'$\isotope[{}][{}]{{{}}}$ {} {}'.format(num1[0], num2[0], num1[1], num2[1], num2[2])
        except IndexError:
            try:
                isotope = r'$\isotope[{}][{}]{{{}}}$'.format(num1[0], num1[2], num1[1])
            except IndexError:
                isotope = item
            pass
        match_count = match_count.append({'Isotope': isotope, 'Number': num}, ignore_index=True)
    with open('{}MatchList.tex'.format(type), 'w') as texfile:
        texfile.write('\\documentclass[a4paper,twoside]{article}\n')
        texfile.write('\\usepackage[margin=0in]{geometry}\n')
        texfile.write('\\usepackage{mathtools}\n')
        texfile.write('\\usepackage[math]{cellspace}\n')
        texfile.write('\\usepackage{isotope}\n')
        texfile.write('\\pagenumbering{gobble}\n')
        texfile.write('\\cellspacetoplimit 4pt\n')
        texfile.write('\\cellspacebottomlimit 4pt\n')
        texfile.write('\n')
        texfile.write(r'\setlength{\topmargin}{1in}')
        texfile.write('\n')
        texfile.write('\\begin{document}\n')
        texfile.write(r'''\ {\def\arraystretch{1.2}\tabcolsep=3pt''')
        texfile.write('\\ \n')
        texfile.write('\\begin{tabular}[h!]{|c|c|} \n')
        texfile.write('\hline')
        row_fields = ('Isotope', 'Hits')
        texfile.write('\\ {} & {} \\\\ \n'.format(row_fields[0], row_fields[1]))
        texfile.write('\hline \hline')
        for i in range(0, len(match_count)):
            texfile.write('\\ {} & {} \\\\ \n'.format(
                    match_count['Isotope'][i], match_count['Number'][i]))
        texfile.write('\hline')
        texfile.write('\\end{tabular}\n')
        texfile.write('\\ }\n')
        texfile.write('\\end{document}\n')


def xraymatch(measured, sigma, type):
    os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2\ReadableData')
    knownss = pd.read_csv('xrays.csv', names=['L', 'U', 'I'], float_precision='round_trip')
    print(knownss)
    match = pd.DataFrame(
            columns=['Measured value', 'Measured uncertainty', 'Lower Range', 'Upper Range', 'Source', 'RI', 'Chi'])
    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0
    for i in range(0, len(knownss)):
        for k in range(0, len(measured)):
            t1 = 0
            t2 = 0
            t3 = 0
            t4 = 0
            if measured[k][2] + measured[k][3] * sigma >= knownss['L'][i]:
                t1 = 1
            if measured[k][2] + measured[k][3] * sigma <= knownss['U'][i]:
                t2 = 1
                print(t1, t2, t3, t4)
                print('Measured: {}+-{}, range:{} - {}'.format(measured[k][2], measured[k][3], knownss['L'][i],
                                                               knownss['U'][i]))
            if measured[k][2] - measured[k][3] * sigma <= knownss['U'][i]:
                t3 = 1
                print(t1, t2, t3, t4)
                print('Measured: {}+-{}, range:{} - {}'.format(measured[k][2], measured[k][3], knownss['L'][i],
                                                               knownss['U'][i]))
            if measured[k][2] - measured[k][3] * sigma >= knownss['L'][i]:
                t4 = 1
            if t1 == 1 and t2 == 1 or t3 == 1 and t4 == 1:
                print('True')
                print('\n')

                num1 = knownss['I'][i].split('-')
                try:
                    num2 = num1[2].split(' ')
                    num = r'$\isotope[{}][{}]{{{}}}$ {} {} {} {}'.format(num2[0], num1[0], num1[1], num2[1], num2[2],
                                                                         num2[3], num2[4])
                except IndexError:
                    try:
                        num = r'$\isotope[{}][{}]{{{}}}$ {} {}'.format(num2[0], num1[0], num1[1], num2[1], num2[2])

                    except IndexError:
                        try:
                            num = r'$\isotope[{}][{}]{{{}}}$'.format(num1[2], num1[0], num1[1])
                        except IndexError:
                            num = knownss['I'][i]
                match.loc[len(match)] = [measured[k][2],
                                         measured[k][3],
                                         knownss['L'][i],
                                         knownss['U'][i],
                                         num,
                                         measured[k][4],
                                         measured[k][1]]
                t1 = 0
                t2 = 0
                t3 = 0
                t4 = 0
                num2 = 'ERR'
                num1 = 'ERR'
                num = 'ERR'
    print(match)
    os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2\Latex')
    print('bing')
    with open('{}XrayMatch_{}.tex'.format(type, today), 'w') as texfile:
        texfile.write('\\documentclass[a4paper,twoside]{article}\n')
        texfile.write('\\usepackage[margin=0in]{geometry}\n')
        texfile.write('\\usepackage{isotope}\n')
        texfile.write('\\usepackage{ltxtable}\n')
        texfile.write('\\usepackage{ltablex}\n')
        texfile.write('\\pagenumbering{gobble}\n')
        texfile.write('\\begin{document}\n')
        texfile.write(r'\ {\def\arraystretch{0.9}\tabcolsep=3pt')
        texfile.write('\\\n')
        texfile.write(r'\begin{tabularx}{\textwidth}{XXXXX}')
        texfile.write('\\\n')
        row_fields = ('$E_{Measured}$ (keV)', 'Accepted range (keV)',
                      'Source', 'RI', '$\chi^2$')
        texfile.write(' {} & {} & {} & {} & {} \\\\ \n'.format(row_fields[0], row_fields[1],
                                                               row_fields[2], row_fields[3],
                                                               row_fields[4]))
        texfile.write('\hline ')
        for i in range(0, len(match)):
            mystvals = uncertainties.ufloat(match['Measured value'][i], match['Measured uncertainty'][i])
            texfile.write(
                    ' ${:.2uL}$ &  {} - {} & {} & {:.3f} & {:.3f} \\\\ \n'.format(mystvals, match['Lower Range'][i],
                                                                                  match['Upper Range'][i],
                                                                                  match['Source'][i],
                                                                                  match['RI'][i],
                                                                                  match['Chi'][i]))
        texfile.write('\hline')
        texfile.write('\\end{tabularx}\n')
        texfile.write('\\ \ }\n')
        texfile.write('\\end{document}\n')
    print('bong')


def DataConvert(datafolder, destinationfolder):
    os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2\{}'.format(datafolder))
    for filename in os.listdir(os.getcwd()):
        print(filename)
        name = filename.split('.')[0]
        nam2 = name.split('_')
        nam3 = nam2[0] + '_' + ('R' + nam2[1])
        if filename.split('.')[1] == 'txt':
            dframename1 = pd.read_csv(filename, header=None, names=['Channel', 'Counts'], delim_whitespace=True)
            print(dframename1)
            print(nam3)
            os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2\{}'.format(destinationfolder))
            dframename1.to_csv('{}.csv'.format(nam3), header=False, index=False)
            os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2\{}'.format(datafolder))
        elif filename.split('.')[1] == 'csv':
            dframename2 = pd.read_csv(filename, header=None, delimiter=',', usecols=[9, 10], engine='c')
            print(dframename2)
            print(nam3)
            os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2\{}'.format(destinationfolder))
            dframename2.to_csv('{}.csv'.format(nam3), header=False, index=False)
            os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2\{}'.format(datafolder))
        elif filename.split('.')[1] == 'xlsx':
            dframename3 = pd.read_excel(filename, usecols='J:K')
            # print(dframename2)
            os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2\{}'.format(destinationfolder))
            dframename3.to_csv('{}.csv'.format(nam3), header=False, index=False)
            os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2\{}'.format(datafolder))


def energyconv():
    nuclide = []
    os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2')
    energies = pd.read_csv('Energies7.csv',
                           names=['Source', 'Energy (keV)', '$\sigma_E$ ($keV$)', 'Probability', '$\sigma_P$',
                                  'Comment'],
                           engine='c', float_precision='round_trip')
    for i in range(0, len(energies)):
        num1 = energies['Source'][i].split('-')
        # print(energies, energies['Source'][i])
        try:
            num2 = num1[2].split(' ')
            nuclide.append(r'$\isotope[{}][{}]{{{}}}$ {} {}'.format(num2[0], num1[0], num1[1], num2[1], num2[2]))
        except IndexError:
            try:
                nuclide.append(r'$\isotope[{}][{}]{{{}}}$'.format(num1[2], num1[0], num1[1]))
            except IndexError:
                nuclide.append(energies['Source'][i][0])
        continue
    energies['Source'] = nuclide
    energies = energies.sort_values('Energy (keV)')
    energies.index = pd.RangeIndex(len(energies.index))
    print(energies)
    with open('energies.tex', 'w') as texfile:
        texfile.write('\\documentclass{article}\n')
        texfile.write('\\usepackage[margin=0in]{geometry}\n')
        texfile.write('\\usepackage{isotope}\n')
        texfile.write('\\usepackage{tabularx}\n')
        texfile.write('\\usepackage{ltablex}\n')
        texfile.write('\\pagenumbering{gobble}\n')
        texfile.write('\\begin{document}\n')
        # texfile.write('\\hskip-5.3cm\n')
        texfile.write(r'''\ {\def\arraystretch{1.2}\tabcolsep=10pt''')
        texfile.write(r'\begin{tabularx}{1.0\textwidth}{XXXXX}')
        texfile.write('\n')
        row_fields = ('Source', 'Energy (keV)', '$\sigma_E$ ($keV$)', 'Probability', '$\sigma_P$')
        texfile.write('\t \\  {} & {} & {} & {} & {} \\\\ \n'.format(row_fields[0], row_fields[1],
                                                                     row_fields[2], row_fields[3],
                                                                     row_fields[4]))
        texfile.write('\t \hline \endhead \n')
        for i in range(0, len(energies)):
            if len(energies['Source'][i].split(' ')) <= 2:
                if energies['Source'][i] == '$\isotope[110m][47]{Ag}$':
                    print(energies['Source'][i], 'wank')
                    texfile.write(
                            '\t {} & {} & {} & {} & {} \\\\ \n'.format(energies['Source'][i], energies['Energy (keV)'][i],
                                                                       energies['$\sigma_E$ ($keV$)'][i],
                                                                       energies['Probability'][i],
                                                                       energies['$\sigma_P$'][i]))
        texfile.write('\hline \n')
        texfile.write('\\end{tabularx}\n')
        texfile.write('\\ }\n')
        texfile.write('\\end{document}\n')
    print(energies)


# print(DataRead(1, 1).dataset)
def doot(peak, run, calib=None):
    plt.switch_backend('QT5Agg')
    fig, ax = plt.subplots()
    pars = DataRead(peak, run, calib)
    figure2, = ax.plot(pars.dataset[pars.IndependentVariable],
                       pars.dataset[pars.DependentVariable], 'o', antialiased='True', color='#1c1c1c', mew=1.0,
                       markersize=4.0)
    # thing = RangeTool(ax, pars.dataset, pars.datafilename, figure2, calib)
    ax.plot(pars.dataset[pars.IndependentVariable],
            pars.dataset[pars.DependentVariable], antialiased='True', color='#1c1c1c')
    # ax.vlines(1475.78, ymin=0., ymax=1000, linestyles='dashed', colors='r', linewidth=0.95, antialiased=True)
    plt.ylabel('Counts ', fontsize=30)
    plt.xlabel('Energy (keV)', fontsize=28)
    plt.yticks(fontsize=22)
    plt.xticks(fontsize=22)
    # plt.xlim([100, 475])
    # plt.yscale('log')
    # plt.ylim([1e0, 1e6])
    # plt.ylim([0, 23000])
    # plt.yscale('log')
    # os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2')
    # matches = pd.read_csv('sh_20180321_153149.csv')
    # matches['ID'] = matches.groupby(['Nuclide']).ngroup()
    # print(matches)
    # texts = []
    # # bars = ax.bar(matches['Measured energy'], 5000000, width=0.5)
    # duplicate_array1 = matches[matches.duplicated('Measured energy', keep=False)]
    # duplicate_array2 = duplicate_array1.groupby('Measured energy').apply(lambda x: tuple(x.index)).tolist()
    # indexer = 0
    # for k in range(0, len(duplicate_array2)):
    #     str_holder = ""
    #     for o in range(0, len(duplicate_array2[k])):
    #         if o != len(duplicate_array2[k])-1:
    #             print('{}'.format(matches['ID'][duplicate_array2[k][o]]), type('{}'.format(matches['ID'][duplicate_array2[k][o]])))
    #             print(matches.loc[duplicate_array2[k][0], 'ID'], type(matches.loc[duplicate_array2[k][0], 'ID']))
    #             str_holder += '{},'.format(matches['ID'][duplicate_array2[k][o]])
    #         elif o == len(duplicate_array2[k])-1:
    #             str_holder += '{}'.format(matches['ID'][duplicate_array2[k][o]])
    #     matches.loc[duplicate_array2[k][0], 'ID'] = str_holder
    # drop_indexes = []
    # for k in range(0, len(duplicate_array2)):
    #     for o in range(1, len(duplicate_array2[k])):
    #         drop_indexes.append(matches.index[duplicate_array2[k][o]])
    # for k in range(0, len(drop_indexes)):
    #     matches = matches.drop(drop_indexes[k])
    #
    #
    # matches.index = pd.RangeIndex(len(matches.index))
    # print(matches)
    # ys = np.logspace(3, 11, len(matches))
    # random.shuffle(ys)
    # print(ys)
    # for h in range(0, len(matches)):
    #     # bars.append(ax.bar(matches['Measured energy'][h], 5000000, width=0.5))
    #     if len('{}'.format(matches['ID'][h]).split(',')) <= 10:
    #         texts.append(ax.text(matches['Measured energy'][h], ys[h], '{}'.format(matches['ID'][h]),
    #                              rotation=0, va='center', ha='center', bbox=dict(pad=0.1, ec='none', fc='w'), fontsize=16))
    #     else:
    #         texts.append(ax.text(matches['Measured energy'][h], ys[h], '*',
    #                              rotation=0, va='center', ha='center', bbox=dict(pad=0.1, ec='none', fc='w'), fontsize=16))
    #         at = AnchoredText("*: {}".format(matches['ID'][h]),
    #                           prop=dict(size=16), frameon=True,
    #                           loc=3,
    #                           )
    #         ax.add_artist(at)
    #
    # adjust_text(texts, force_text=200, force_objects=0, force_points=0, expand_text=(0.4, 0.9), ha='center',
    #             only_move={'points': 'y', 'text': 'y', 'objects': 'y'}, lim=100000)
    # for h in range(0, len(texts)):
    #     ax.vlines(matches['Measured energy'][h], ymin=0., ymax=texts[h].get_position()[1], linestyles='dashed', colors='r', linewidth=0.95, antialiased=True)
    fig.subplots_adjust(top=0.953,
                        bottom=0.162,
                        left=0.152,
                        right=0.974,
                        hspace=0.286,
                        wspace=0.402)
    ax.grid(color='k', linestyle='--', alpha=0.2)
    # fig.set_size_inches(13.5, 10.5)

    print('----{}----'.format(time.time() - start_time))
    print('\n')
    # print('{}_{}_{}'.format(peak, run, calib.__name__))
    if os.path.isfile('C:\\Users\\Josh\\Documents\\git\HPGe2\Ranges\{}_{}.csv'.format(peak, run)):
        pars.multiplot()
    # plt.legend()
    os.chdir('C:\\Users\Josh\Documents\\git\HPGe2\Long Report\Figures')
    # plt.savefig('Co_60.png', dpi=600)
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    # plt.show()


escaped = []
double_escaped = []


def escapematch(measured, known, sigma, type):
    os.chdir('Latex')
    for i in range(0, len(known)):
        for k in range(0, len(measured)):
            b = overlap(known[i][1], measured[k][2], known[i][2], measured[k][3], sigma)
            if b == 0:
                energymatch.append(known[i][1])
                mystmatch.append(measured[k][2])
                energymatchuncertainty.append(known[i][2])
                mystmatchuncertainty.append(measured[k][3])
                nuclidematch.append(known[i][0])
                intensitymatch.append(measured[k][4])
                chisss.append(measured[k][1])
                num1 = known[i][0].split('-')
                try:
                    num2 = num1[2].split(' ')
                    formattednuclide.append(
                            r'$\isotope[{}][{}]{{{}}}$ {} {}'.format(num2[0], num1[0], num1[1], num2[1], num2[2]))
                except IndexError:
                    try:
                        formattednuclide.append(r'$\isotope[{}][{}]{{{}}}$'.format(num1[2], num1[0], num1[1]))
                    except IndexError:
                        formattednuclide.append(known[i][0])
                continue

    for i in range(0, len(nuclidematch)):
        escaped.append(mystmatch[i] - 511)
        double_escaped.append(mystmatch[i] - (511 * 2))
    measured_dataframe = pd.DataFrame(measured)
    print(measured_dataframe)
    match_frame = pd.DataFrame({'Accepted energy': energymatch, 'Accepted uncertainty': energymatchuncertainty,
                                'Measured energy': mystmatch, 'Measured uncertainty': mystmatchuncertainty,
                                'Nuclide':         formattednuclide, 'Height': intensitymatch, '1 Escape': escaped,
                                '2 Escape':        double_escaped, 'chis': chisss})
    print(match_frame)
    os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2')
    match_frame.to_csv('{}_{}.csv'.format(type, today))
    os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2\Latex')
    energymatch2 = []
    mystmatch2 = []
    energymatchuncertainty2 = []
    mystmatchuncertainty2 = []
    nuclidematch2 = []
    intensitymatch2 = []
    chisss2 = []
    datframe1 = pd.DataFrame(
            columns=['E_acc', 'E_meas', 'unc_acc', 'unc_meas', 'source', 'type', 'matches', 'RI', 'chis'])

    tempframe = pd.DataFrame(columns=['Escape energy', 'Escape uncertainty', 'Match energy', 'Match uncertainty',
                                      'Match isotope', 'Accepted energy', 'Accepted uncertainty', 'FEPP', 'FEPP source',
                                      '1 or 2', 'RI', 'chi_s'])
    for i in range(0, len(match_frame)):
        for k in range(0, len(match_frame)):
            b = overlap(match_frame['Measured energy'][i], match_frame['1 Escape'][k],
                        match_frame['Measured uncertainty'][i] * 2, match_frame['Measured uncertainty'][k], sigma)
            if b == 0:
                tempframe.loc[len(tempframe)] = [match_frame['1 Escape'][k],
                                                 match_frame['Measured uncertainty'][k],
                                                 match_frame['Measured energy'][i],
                                                 match_frame['Measured uncertainty'][i],
                                                 match_frame['Nuclide'][i],
                                                 match_frame['Accepted energy'][i],
                                                 match_frame['Accepted uncertainty'][i],
                                                 match_frame['Measured energy'][k],
                                                 match_frame['Nuclide'][k],
                                                 '1',
                                                 match_frame['Height'][k],
                                                 match_frame['chis'][k]]
    for i in range(0, len(match_frame)):
        for k in range(0, len(match_frame)):
            b = overlap(match_frame['Measured energy'][i], match_frame['2 Escape'][k],
                        match_frame['Measured uncertainty'][i] * 2, match_frame['Measured uncertainty'][k], sigma)
            if b == 0:
                tempframe.loc[len(tempframe)] = [match_frame['2 Escape'][k],
                                                 match_frame['Measured uncertainty'][k],
                                                 match_frame['Measured energy'][i],
                                                 match_frame['Measured uncertainty'][i],
                                                 match_frame['Nuclide'][i],
                                                 match_frame['Accepted energy'][i],
                                                 match_frame['Accepted uncertainty'][i],
                                                 match_frame['Measured energy'][k],
                                                 match_frame['Nuclide'][k],
                                                 '2',
                                                 match_frame['Height'][k],
                                                 match_frame['chis'][k]]
                # print(tempframe)
    for i in range(0, len(tempframe)):
        if tempframe['1 or 2'][i] == '1':
            tempframe.at[i, 'FEPP source'] = '{} Single'.format(tempframe.at[i, 'FEPP source'])
        if tempframe['1 or 2'][i] == '2':
            tempframe.at[i, 'FEPP source'] = '{} Double'.format(tempframe.at[i, 'FEPP source'])
    tempframe.drop(columns=['1 or 2'], inplace=True)
    tempframe['Escape energy'] = tempframe['Escape energy'].astype('str')
    tempframe['Match energy'] = tempframe['Match energy'].astype('str')
    tempframe['FEPP'] = tempframe['FEPP'].astype('str')
    with open('{}Match_{}.tex'.format(type, today), 'w') as texfile:
        texfile.write('\\documentclass[a4paper,twoside]{article}\n')
        texfile.write('\\usepackage[margin=0in]{geometry}\n')
        texfile.write('\\usepackage{isotope}\n')
        texfile.write('\\pagenumbering{gobble}\n')
        texfile.write('\\begin{document}\n')
        # texfile.write('\\hskip-5.3cm\n')
        texfile.write('\\begin{tabular}[ht!]{cccccccc} \n')
        row_fields = (
            '$E_{Escape}$ (keV)', '$E_{Measured}$ (keV)', 'Match Isotope', 'Accepted energy (keV)', 'FEPP (keV)',
            'FEPP Source', 'RI', '$\chi^2$')
        texfile.write('\\  {} & {} & {} & {} & {} & {} & {} & {} \\\\ \n'.format(row_fields[0], row_fields[1],
                                                                                 row_fields[2], row_fields[3],
                                                                                 row_fields[4], row_fields[5],
                                                                                 row_fields[6],
                                                                                 row_fields[7]))
        texfile.write('\hline')
        for i in range(0, len(tempframe)):
            escp = uncertainties.ufloat(tempframe.at[i, 'Escape energy'], tempframe.at[i, 'Escape uncertainty'])
            mtch = uncertainties.ufloat(tempframe.at[i, 'Match energy'], tempframe.at[i, 'Match uncertainty'])
            fepp = uncertainties.ufloat(tempframe.at[i, 'FEPP'], tempframe.at[i, 'Escape uncertainty'])
            acc = uncertainties.ufloat(tempframe.at[i, 'Accepted energy'], tempframe.at[i, 'Accepted uncertainty'])
            if float(tempframe.at[i, 'FEPP']) >= 510.9 * 2:
                texfile.write(
                        '\\ ${:.2uL}$ & ${:.2uL}$ & {} & ${:.2uL}$ & ${:.2uL}$ & {} & {:.2f} & {:.3f} \\\\ \n'.format(escp,
                                                                                                                      mtch,
                                                                                                                      tempframe.at[
                                                                                                                          i, 'Match isotope'],
                                                                                                                      acc,
                                                                                                                      fepp,
                                                                                                                      tempframe.at[
                                                                                                                          i, 'FEPP source'],
                                                                                                                      tempframe.at[
                                                                                                                          i, 'RI'],
                                                                                                                      tempframe.at[
                                                                                                                          i, 'chi_s']))
        texfile.write('\hline')
        texfile.write('\\end{tabular}\n')
        texfile.write('\\end{document}\n')
    #
    # print(nuclidematch)
    # print(nuclidematch2)


def comptonmatch(measured, known, sigma, type):
    compedge = []
    backscatter = []
    for i in range(0, len(known)):
        for k in range(0, len(measured)):
            b = overlap(known[i][1], measured[k][2], known[i][2], measured[k][3], sigma)
            if b == 0:
                energymatch.append(known[i][1])
                mystmatch.append(measured[k][2])
                energymatchuncertainty.append(known[i][2])
                mystmatchuncertainty.append(measured[k][3])
                nuclidematch.append(known[i][0])
                intensitymatch.append(measured[k][4])
                chisss.append(measured[k][1])
                num1 = known[i][0].split('-')
                try:
                    num2 = num1[2].split(' ')
                    formattednuclide.append(
                            r'$\isotope[{}][{}]{{{}}}$ {} {}'.format(num2[0], num1[0], num1[1], num2[1], num2[2]))
                except IndexError:
                    try:
                        formattednuclide.append(r'$\isotope[{}][{}]{{{}}}$'.format(num1[2], num1[0], num1[1]))
                    except IndexError:
                        formattednuclide.append(known[i][0])
                continue

    for i in range(0, len(nuclidematch)):
        compedge.append(kleinnishina(mystmatch[i], np.pi))
        backscatter.append(comp.bcksctr(mystmatch[i]))
        print(comp.bcksctr(mystmatch[i]))
    measured_dataframe = pd.DataFrame(measured)
    print(measured_dataframe)
    match_frame = pd.DataFrame({'Accepted energy':  energymatch, 'Accepted uncertainty': energymatchuncertainty,
                                'Measured energy':  mystmatch, 'Measured uncertainty': mystmatchuncertainty,
                                'Nuclide':          formattednuclide, 'Height': intensitymatch, 'Compton Edge': compedge,
                                'Backscatter peak': backscatter, 'chis': chisss})
    print(match_frame)
    # os.chdir('C:\\Users\\Josh\\Documents\\git\HPGe2')
    match_frame.to_csv('{}_{}.csv'.format(type, today))
    os.chdir('Latex')
    datframe1 = pd.DataFrame(
            columns=['E_acc', 'E_meas', 'unc_acc', 'unc_meas', 'source', 'type', 'matches', 'RI', 'chis'])

    tempframe = pd.DataFrame(columns=['Comp. energy', 'Comp. uncertainty', 'Match energy', 'Match uncertainty',
                                      'Match isotope', 'Accepted energy', 'Accepted uncertainty', 'FEPP', 'FEPP source',
                                      'Edge or Backscatter', 'RI', 'chi_s'])
    for i in range(0, len(match_frame)):
        for k in range(0, len(match_frame)):
            b = overlap(match_frame['Measured energy'][i], match_frame['Backscatter peak'][k],
                        match_frame['Measured uncertainty'][i], match_frame['Measured uncertainty'][k], sigma)
            if b == 0 and match_frame['Measured energy'][i] > 100:
                tempframe.loc[len(tempframe)] = [match_frame['Backscatter peak'][k],
                                                 match_frame['Measured uncertainty'][k],
                                                 match_frame['Measured energy'][i],
                                                 match_frame['Measured uncertainty'][i],
                                                 match_frame['Nuclide'][i],
                                                 match_frame['Accepted energy'][i],
                                                 match_frame['Accepted uncertainty'][i],
                                                 match_frame['Measured energy'][k],
                                                 match_frame['Nuclide'][k],
                                                 'Backscatter peak',
                                                 match_frame['Height'][k],
                                                 match_frame['chis'][k]]
                # print(tempframe)
    for i in range(0, len(tempframe)):
        if tempframe['Edge or Backscatter'][i] == 'Backscatter peak':
            tempframe.at[i, 'FEPP source'] = '{} Backscatter peak'.format(tempframe.at[i, 'FEPP source'])
    for i in range(0, len(tempframe)):
        if tempframe['Edge or Backscatter'][i] == 'Compton Edge':
            tempframe.drop(tempframe.index[i], inplace=True)
    tempframe.drop(columns=['Edge or Backscatter'], inplace=True)
    tempframe['Comp. energy'] = tempframe['Comp. energy'].astype('str')
    tempframe['Match energy'] = tempframe['Match energy'].astype('str')
    tempframe['FEPP'] = tempframe['FEPP'].astype('str')
    with open('{}Match_{}.tex'.format(type, today), 'w') as texfile:
        texfile.write('\\documentclass[a4paper,twoside]{article}\n')
        texfile.write('\\usepackage[margin=0in]{geometry}\n')
        texfile.write('\\usepackage{isotope}\n')
        texfile.write('\\pagenumbering{gobble}\n')
        texfile.write('\\begin{document}\n')
        # texfile.write('\\hskip-5.3cm\n')
        texfile.write('\\begin{tabular}[ht!]{cccccccc} \n')
        row_fields = (
            '$E_{Comp}$ (keV)', '$E_{Measured}$ (keV)', 'Match Isotope', 'Accepted energy (keV)', 'FEPP (keV)',
            'FEPP Source', 'RI', '$\chi^2$')
        texfile.write('\\  {} & {} & {} & {} & {} & {} & {} & {} \\\\ \n'.format(row_fields[0], row_fields[1],
                                                                                 row_fields[2], row_fields[3],
                                                                                 row_fields[4], row_fields[5],
                                                                                 row_fields[6],
                                                                                 row_fields[7]))
        texfile.write('\hline')
        for i in range(0, len(tempframe)):
            escp = uncertainties.ufloat(tempframe.at[i, 'Comp. energy'], tempframe.at[i, 'Comp. uncertainty'])
            mtch = uncertainties.ufloat(tempframe.at[i, 'Match energy'], tempframe.at[i, 'Match uncertainty'])
            fepp = uncertainties.ufloat(tempframe.at[i, 'FEPP'], tempframe.at[i, 'Comp. uncertainty'])
            acc = uncertainties.ufloat(tempframe.at[i, 'Accepted energy'], tempframe.at[i, 'Accepted uncertainty'])
            texfile.write(
                    '\\ ${:.2uL}$ & ${:.2uL}$ & {} & ${:.2uL}$ & ${:.2uL}$ & {} & {:.2f} & {:.3f} \\\\ \n'.format(escp,
                                                                                                                  mtch,
                                                                                                                  tempframe.at[
                                                                                                                      i, 'Match isotope'],
                                                                                                                  acc,
                                                                                                                  fepp,
                                                                                                                  tempframe.at[
                                                                                                                      i, 'FEPP source'],
                                                                                                                  tempframe.at[
                                                                                                                      i, 'RI'],
                                                                                                                  tempframe.at[
                                                                                                                      i, 'chi_s']))
        texfile.write('\hline')
        texfile.write('\\end{tabular}\n')
        texfile.write('\\end{document}\n')


def coincidencematch(measured, known, sigma, type):
    os.chdir('Latex')
    for i in range(0, len(known)):
        for k in range(0, len(measured)):
            b = overlap(known[i][1], measured[k][2], known[i][2], measured[k][3], sigma)
            if b == 0:
                energymatch.append(known[i][1])
                mystmatch.append(measured[k][2])
                energymatchuncertainty.append(known[i][2])
                mystmatchuncertainty.append(measured[k][3])
                nuclidematch.append(known[i][0])
                intensitymatch.append(measured[k][4])
                chisss.append(measured[k][1])
                num1 = known[i][0].split('-')
                try:
                    num2 = num1[2].split(' ')
                    formattednuclide.append(
                            r'$\isotope[{}][{}]{{{}}}$ {} {}'.format(num2[0], num1[0], num1[1], num2[1], num2[2]))
                except IndexError:
                    try:
                        formattednuclide.append(r'$\isotope[{}][{}]{{{}}}$'.format(num1[2], num1[0], num1[1]))
                    except IndexError:
                        formattednuclide.append(known[i][0])
                continue
    measured_dataframe = pd.DataFrame(measured)
    print(measured_dataframe)
    match_frame = pd.DataFrame({'Accepted energy': energymatch, 'Accepted uncertainty': energymatchuncertainty,
                                'Measured energy': mystmatch, 'Measured uncertainty': mystmatchuncertainty,
                                'Nuclide':         formattednuclide, 'Height': intensitymatch, 'chis': chisss})
    tempframe = pd.DataFrame(
            columns=['E_1', 'E_1 uncertainty', 'E_1 RI', 'Isotope 1', 'E_2', 'E_2 uncertainty', 'E_2 RI', 'Isotope 2',
                     'E_sum', 'E_sum uncertainty', 'Sum isotope', 'RI', 'chi_s'])
    for i in range(0, len(match_frame)):
        for j in range(0, len(match_frame)):
            for k in range(0, len(match_frame)):
                b = overlap(match_frame['Measured energy'][i],
                            match_frame['Measured energy'][j] + match_frame['Measured energy'][k],
                            match_frame['Measured uncertainty'][i],
                            match_frame['Measured uncertainty'][j] + match_frame['Measured uncertainty'][k], sigma)
                if b == 0:
                    if match_frame['Height'][i] <= 0.55 * (1 + match_frame['Height'][j] + match_frame['Height'][k]):
                        if match_frame['Height'][j] >= 1 and match_frame['Height'][k] >= 1:
                            tempframe.loc[len(tempframe)] = [match_frame['Measured energy'][j],
                                                             match_frame['Measured uncertainty'][j],
                                                             match_frame['Height'][j],
                                                             match_frame['Nuclide'][j],
                                                             match_frame['Measured energy'][k],
                                                             match_frame['Measured uncertainty'][k],
                                                             match_frame['Height'][k],
                                                             match_frame['Nuclide'][k],
                                                             match_frame['Measured energy'][i],
                                                             match_frame['Measured uncertainty'][i],
                                                             match_frame['Nuclide'][i],
                                                             match_frame['Height'][i],
                                                             match_frame['chis'][i]]
    tempframe.drop(columns=['Isotope 1', 'Isotope 2'], inplace=True)
    tempframe.drop_duplicates(inplace=True)
    tempframe.index = pd.RangeIndex(len(tempframe.index))
    print(tempframe)
    with open('{}coincidence_{}.tex'.format(type, today), 'w') as texfile:
        texfile.write('\\documentclass[a4paper,twoside]{article}\n')
        texfile.write('\\usepackage[margin=0in]{geometry}\n')
        texfile.write('\\usepackage{isotope}\n')
        texfile.write('\\pagenumbering{gobble}\n')
        texfile.write('\\begin{document}\n')
        # texfile.write('\\hskip-5.3cm\n')
        texfile.write('\\begin{tabular}[ht!]{cccccccccc} \n')
        row_fields = ('$E_1$', '$RI_1$', '$E_2$', '$RI_2$',
                      '$E_sum$', '$RI_S$', 'Sum isotope', r'$\chi^2_{\nu}$')
        texfile.write('\\  {} & {} & {} & {} & {} & {} & {} & {}  \\\\ \n'.format(row_fields[0], row_fields[1],
                                                                                  row_fields[2], row_fields[3],
                                                                                  row_fields[4], row_fields[5],
                                                                                  row_fields[6],
                                                                                  row_fields[7]))
        texfile.write('\hline')
        for i in range(0, len(tempframe)):
            print(i)

            e1 = uncertainties.ufloat(tempframe.at[i, 'E_1'], tempframe.at[i, 'E_1 uncertainty'])
            e2 = uncertainties.ufloat(tempframe.at[i, 'E_2'], tempframe.at[i, 'E_2 uncertainty'])
            es = uncertainties.ufloat(tempframe.at[i, 'E_sum'], tempframe.at[i, 'E_sum uncertainty'])

            texfile.write(
                    '\\ ${:.2uL}$ & ${:.2f}$ & ${:.2uL}$ & ${:.2f}$ & ${:.2uL}$ & ${:.2f}$ & {} & ${:.3f}$ \\\\ \n'.format(
                            e1,
                            tempframe['E_1 RI'][i],
                            e2,
                            tempframe['E_2 RI'][i],
                            es,
                            tempframe['RI'][i],
                            tempframe['Sum isotope'][i],
                            tempframe['chi_s'][i]))
        texfile.write(r'\hline')
        texfile.write('\\end{tabular}\n')
        texfile.write('\\end{document}\n')


# xraymatch(analmystlist, 1, 'ShieldedXray')

# escapematch(analmystlist, energylist, 1, 'UnshMystery')
# comptonmatch(analmystlist, energylist, 3, 'comptonunshielded')
# coincidencematch(analmystlist, energylist, 1, 'myst')
# datamatch(analmystlist, xrays2, 1, 'myst')
# doot('mystery', '1')
# doot('mystery', 'R101117', clb.g2)
# doot('barium133', 'R071117')
# doot('cobalt60', 'R031117', clb.g2)
# doot('sodium22', 'R071117', clb.g2)

doot('unshielded', 'R141117', clb.g2)
# doot('shieldedbackground', 'R061117', clb.g2)
# DataConvert('Data', 'ReadableData')

# energyconv()


print('Complete')
