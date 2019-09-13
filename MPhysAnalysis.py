import numpy as np
import matplotlib.pylab as plt
import os
from scipy.optimize import curve_fit
import csv
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

os.chdir('C:\\Users\Joshua\Desktop\HPGe')

with open('axis.csv', "r") as f:
    reader = csv.reader(f, delimiter=',')
    for line in reader:
        line_list = line
f.close()
line_list = [float(i) for i in line_list]

rangelist2 = np.genfromtxt('m sranges1.csv', delimiter=',', dtype='float')
energylist = np.genfromtxt('Energies7.csv', delimiter=',', dtype=('U60', 'float', 'float', 'float', 'float', 'U60'))
mystlist = np.genfromtxt('Mystery_2.csv', delimiter=',', dtype='float')

energymatch = []
mystmatch = []
energymatchuncertainty = []
mystmatchuncertainty = []
nuclidematch = []
intensitymatch = []
chisss = []

def peakdetect(x, y):

def overlap(point1, point2, uncert1, uncert2, sigma):
    lower1 = point1 - uncert1*sigma
    upper1 = point1 + uncert1*sigma
    lower2 = point2 - uncert2*sigma
    upper2 = point2 + uncert2*sigma
    index1 = np.where((lower1 <= point2) and (upper1 >= point2) or (lower2 <= point1) and (upper2 >= point1))
    # index1 = set.intersection({lower2, upper2}, {lower1, upper1})
    if index1[0] == 0:
        return index1[0]


def datamatch(measured, known, sigma):
    for i in range(0, len(known)):
        for k in range(0, len(measured)):
            global b
            b = overlap(known[i][1], measured[k][2], known[i][2], measured[k][3], sigma)
            if b == 0:
                energymatch.append(known[i][1])
                mystmatch.append(measured[k][2])
                energymatchuncertainty.append(known[i][2])
                mystmatchuncertainty.append(measured[k][3])
                nuclidematch.append(known[i][0])
                intensitymatch.append(measured[k][4])
                chisss.append(measured[k][1])
        if b == 0:
            continue
    
        with open('shitfuck.tex', 'w') as texfile:
            texfile.write('\\documentclass[a4paper,twoside]{article}\n')
            texfile.write('\\usepackage[margin=0in]{geometry}\n')
            texfile.write('\\pagenumbering{gobble}\n')
            texfile.write('\\begin{document}\n')
            #texfile.write('\\hskip-5.3cm\n')
            texfile.write('\\begin{tabular}[ht!]{c|c|c|c|c|c|c} \n')
            row_fields = ('$E_{Measured}$ (keV)', '$\sigma_{E-M}$ (keV)', '$E_{Accepted}$ (keV)',
                          '$\sigma_{E-A}$ (keV)', 'Source', 'RI', '$\chi^2$')
            texfile.write('\\  {} & {} & {} & {} & {} & {} & {} \\\\ \n'.format(row_fields[0], row_fields[1],
                                                                      row_fields[2], row_fields[3],
                                                                      row_fields[4], row_fields[5], row_fields[6]))
            texfile.write('\hline \hline')
            for i in range(0, len(energymatch)):
                if mystmatchuncertainty[i] < 20:
                    texfile.write('\\ {} & {} & {} & {} & {} & {} & {} \\\\ \n'.format(mystmatch[i], mystmatchuncertainty[i], energymatch[i],
                                                                             energymatchuncertainty[i],
                                                                             nuclidematch[i], intensitymatch[i], chisss[i]))
            texfile.write('\hline')
            texfile.write('\\end{tabular}\n')
            texfile.write('\\end{document}\n')

def find_closest(A, target):
    #  A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx

# print('Available datasets: Na22, Co60, Ba133, Background, Mystery')
# purpose = input('View graph or extract data?')
# if purpose == 'view' or purpose == 'graph' or purpose == 'view graph':
#     cur = input('Cursor?')
#     gau = input('Fit Gaussians?')
# run = input('Source:')
# ver = input('Which run?')
# item = '%s %s' % (run, ver)


class SnaptoCursor(object):
    """
    Like Cursor but the crosshair snaps to the nearest x,y point
    For simplicity, I'm assuming x is sorted
    """

    def __init__(self, ax, x, y):
        self.ax = ax
        self.lx = ax.axhline(color='k')  # the horiz line
        self.ly = ax.axvline(color='k')  # the vert line
        self.x = x
        self.y = y
        # text location in axes coords
        self.txt = ax.text(0.7, 0.9, '', transform=ax.transAxes)

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

        self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
        #print('x=%1.2f, y=%1.2f' % (x, y))
        plt.draw()


fig, ax = plt.subplots()


arrowargs = dict(arrowprops=dict(
            facecolor='black', shrink=0.1, width=1.0, headwidth=4.0, headlength=5.0))
plotargs = dict(antialiased='True', color='k', mew=1.0, markersize=1.5)

scaledlist = ['scaled', 'scale', 'Scaled', 'Scale', 's', 'S']
mysterylist = ['mystery', 'Mystery', 'm', 'M']


def gauss(xvar, co_a, co_b, co_c):
    return co_a*(np.exp(-((xvar-co_b)**2)/(2*(co_c**2))))


def lingauss(xvar, co_a, co_b, co_c, co_d, co_e):
    return co_a*np.exp(-((xvar-co_b)**2)/(2*(co_c**2))) + co_d*xvar + co_e


Na22_1 = dict(xvals=line_list, yvals=count, source='Na22', runnum='1')
Na22_2 = dict(xvals=line_list, yvals=countB_2, source='Na22', runnum='2')
Na22_3 = dict(xvals=line_list, yvals=countB_3, source='Na22', runnum='3')
Co60_1 = dict(xvals=line_list, yvals=countD, source='Co60', runnum='1')
Co60_2 = dict(xvals=line_list, yvals=countD_2, source='Co60', runnum='1')
Ba133_1 = dict(xvals=line_list, yvals=countE, source='Ba133', runnum='1')
Ba133_2 = dict(xvals=line_list, yvals=countE_2, source='Ba133', runnum='1')
Mystery_1 = dict(xvals=line_list, yvals=c3s, source='Mystery', runnum='')
Mystery_2 = dict(xvals=line_list, yvals=countM_1, source='Mystery', runnum='')
Background_3 = dict(xvals=line_list, yvals=countC_3, source='Background', runnum='3')
Background_4 = dict(xvals=line_list, yvals=countC_4, source='Background', runnum='4')

models = []
chis = []
redchis = []
sources = []
runnums = []
peaks = []
meanerror = []
amplitude = []

def gaussfit(model, u, l, xvals, yvals, source='blank', runnum='blank'):
    l3 = xvals.index(min(xvals, key=lambda k: abs(k - l)))
    u3 = xvals.index(min(xvals, key=lambda o: abs(o - u)))
    l2 = np.min((l3, u3))
    u2 = np.max((l3, u3))
    chrange = xvals[l2:u2]
    corange = yvals[l2:u2]
    roughmean = float(sum(chrange)/len(corange))
    if model == gauss:
        modelll = 'Gaussian'
        initial = [max(corange), roughmean, np.std(chrange)]
    if model == lingauss:
        modelll = 'Gaussian + Line'
        initial = [max(corange), roughmean, np.std(chrange), -0.1, 100]
    try:
        popt, pcov = curve_fit(model, chrange, corange, initial, sigma=(np.sqrt((np.sqrt(corange))**2) + ((1/1.37290058)/2)**2), absolute_sigma=True,
                               maxfev=10000)
        mean = popt[1]
        pointchisq = []
        for i in range(0, len(chrange)):
            pointchisq.append(((corange[i] - model(chrange[i], *popt)) / (np.sqrt(corange[i]))) ** 2)
        chisq = sum(pointchisq)
        redchisq = chisq / len(chrange)
        err = np.sqrt(abs((np.diag(pcov))))
        return popt, pcov, l2, u2, chisq, redchisq, mean, err, chrange, corange, modelll, initial
    except TypeError:
        try:
            print('strike one', l2, u2)
            l2 = l2 - 1
            u2 = u2 + 1
            chrange = xvals[l2:u2]
            corange = yvals[l2:u2]
            roughmean = float(sum(chrange) / len(corange))
            initial = [max(corange), roughmean, np.std(chrange), -0.1, 100]
            popt, pcov = curve_fit(model, chrange, corange, initial, sigma=(np.sqrt((np.sqrt(corange))**2) + ((1/1.37290058)/2)**2), absolute_sigma=True,
                                   maxfev=10000)
            mean = popt[1]
            pointchisq = []
            for i in range(0, len(chrange)):
                pointchisq.append(((corange[i] - model(chrange[i], *popt)) / (np.sqrt(corange[i]))) ** 2)
            chisq = sum(pointchisq)
            redchisq = chisq / len(chrange)
            err = np.sqrt(abs((np.diag(pcov))))
            return popt, pcov, l2, u2, chisq, redchisq, mean, err, chrange, corange, modelll, initial
        except TypeError:
            try:
                print('strike two', l2, u2)
                l2 = l2 - 1
                u2 = u2 + 1
                chrange = xvals[l2:u2]
                corange = yvals[l2:u2]
                roughmean = float(sum(chrange) / len(corange))
                initial = [max(corange), roughmean, np.std(chrange), -0.1, 100]
                popt, pcov = curve_fit(model, chrange, corange, initial, sigma=(np.sqrt((np.sqrt(corange))**2) + ((1/1.37290058)/2)**2), absolute_sigma=True,
                                       maxfev=10000)
                mean = popt[1]
                pointchisq = []
                for i in range(0, len(chrange)):
                    pointchisq.append(((corange[i] - model(chrange[i], *popt)) / (np.sqrt(corange[i]))) ** 2)
                chisq = sum(pointchisq)
                redchisq = chisq / len(chrange)
                err = np.sqrt(abs((np.diag(pcov))))
                return popt, pcov, l2, u2, chisq, redchisq, mean, err, chrange, corange, modelll, initial
            finally:
                print('fuck')

def data_output(model, range3, **data):
    u = range3[0]
    l = range3[1]
    popt, pcov, l2, u2, chisq, redchisq, mean, err, chrange, corange, modelll, initial = gaussfit(model, u, l, **data)
    print(l, l2, mean, popt[1], u, u2)
    models.append('{}'.format(modelll))
    chis.append('{}'.format(float(chisq)))
    redchis.append('{}'.format(float(redchisq)))
    peaks.append('{}'.format(float(popt[1])))
    meanerror.append('{:.2E}'.format(err[1]))
    amplitude.append(int(max(corange)))

def data_display(run, ver, model, range4, **data):
    for i in range(0, len(range4)):
        try:
            data_output(model, range4[i], **data)
        # except TypeError:
        #     print('TypeError', range4[i])
        #     continue
        except RuntimeError:
            print('data_display RunTimeError', range4[i])
            continue
        except ZeroDivisionError:
            print('data_display ZeroDivisionError', range4[i])
            continue
    count_sum = np.sum(amplitude)
    tempdata_array = np.array([chis, redchis, peaks, meanerror, amplitude])
    data_array = np.transpose(tempdata_array)
    data_array2 = [[], [], [], [], []]
    #print(data_array2)
    for i in range(0, len(peaks)):
        data_array2[0].append('%.3g' % float(data_array[i, 0]))  # chis
        data_array2[1].append('%.3g' % float(data_array[i, 1]))  # redchis
        length = len('%.2G' % (float(np.sqrt((float(data_array[i, 3]))**2) + ((1/1.37290058)/2)**2)))
        tempmean = round(float(data_array[i, 2]), length - 2)
        if float(data_array[i, 3]) > 100:
            count_sum = count_sum - float(data_array[i, 4])
            print('Large Error detected', tempmean)
        norm_amp = round((amplitude[i]/count_sum)*100, 3)
        data_array2[2].append(tempmean)  # peaks
        data_array2[3].append('%.2g' % (float(np.sqrt((float(data_array[i, 3]))**2) + ((1/1.37290058)/2)**2)))  # peak error NOTE THE ADDITION OF RESOLUTION
        data_array2[4].append(norm_amp)  # amplitude
    with open('{}_{}.tex'.format(run, ver), 'w') as texfile:
        texfile.write('\\documentclass[a4paper,twoside]{article}\n')
        texfile.write('\\pagenumbering{gobble}\n')
        texfile.write('\\usepackage[margin=0in]{geometry}\n')
        texfile.write('\\begin{document}\n')
        #texfile.write('\\hskip-4.0cm\n')
        texfile.write('\\begin{tabular}[ht!]{c|c|c|c|c} \n')
        row_fields = ('$\chi^2$', 'Reduced $\chi^2$', 'Energy (keV)',
                      '$\sigma_E$ (keV)', 'Normalised Intensity')
        texfile.write('\\  {} & {} & {} & {} & {} \\\\ \n'.format(row_fields[0], row_fields[1],
                                                                                row_fields[2], row_fields[3],
                                                                                row_fields[4]))
        texfile.write('\hline \hline')
        for i in range(0, len(models)):
            if float(data_array2[3][i]) < 100:
                texfile.write('\\ {} & {} & {} & {} & {} \\\\ \n'.format(data_array2[0][i], data_array2[1][i],
                           data_array2[2][i], data_array2[3][i], data_array2[4][i]))
        texfile.write('\hline')
        texfile.write('\\end{tabular}\n')
        texfile.write('\\end{document}\n')

    with open('{}_{}.csv'.format(run, ver), 'w') as f2:
        writer = csv.writer(f2, delimiter=',')
    f.close()

    with open('{}_{}.csv'.format(run, ver), "a") as f2:
        writer = csv.writer(f2, delimiter=',')
        for i in range(0, len(models)):
            writer.writerow([data_array2[0][i], data_array2[1][i], data_array2[2][i], data_array2[3][i], data_array2[4][i]])

def gaussplot(subs, model, range2, cols, **data):
    for k in range(0, round(len(range2)/subs)):
        Rows = subs // cols
        Rows += subs % cols
        Position = range(1, subs + 1)
        for i in range(0, subs):
            try:
                u = range2[(k*subs)+i, 0]
                l = range2[(k*subs)+i, 1]
            except IndexError:
                print('gaussplot index error')
                break
            try:
                popt, pcov, l2, u2, chisq, redchisq, mean, err, chrange, corange, modelll, initial = \
                    gaussfit(model, u, l, **data)
            except TypeError:
                print('gaussplot type error')
                continue
            except RuntimeError:
                print('gaussplot runtime error')
                continue
            #print(l, mean, u)
            fig = plt.figure(k+2)
            ax = fig.add_subplot(Rows, cols, Position[i])
            ax.plot(np.linspace(l, u, 500), model(np.linspace(l, u, 500), *popt), antialiased='True', mew=1.0, linewidth=0.5, color='b')
            yerr = np.sqrt(corange)
            ax.errorbar(x=chrange, y=corange, yerr=yerr, fmt='.', color='k', capsize=2, capthick=1, markersize=0,
                                  elinewidth=1,
                                markeredgewidth=1)
            ax.plot(chrange, corange, '.', **plotargs)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
            plt.tight_layout()
            plt.xlabel('Energy (keV)')
            plt.ylabel('Counts')
            fig.set_size_inches(16.5, 10.5)
            fig.subplots_adjust(wspace=0.6, hspace=0.2)
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        #plt.savefig('Mystery{}.png'.format(k), dpi=600)
        plt.draw()
    plt.show()