import datetime
import math
import os
import time

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


def datamatch(measured, known, sigma, type, xraySigma):
    for i in range(0, len(known)):
        for k in range(0, len(measured)):
            b = None
            if i <= 329:
                b = overlap(known.at[i, "Energy"], measured.at[k, "Mean"], known.at[i, "EnergyUncert"], measured.at[k, "MeanError"], sigma)
            if i > 329:
                b = overlap(known.at[i, "Energy"], measured.at[k, "Mean"], known.at[i, "EnergyUncert"], measured.at[k, "MeanError"], xraySigma)
            if b == 0:
                energymatch.append(known.at[i, "Energy"])
                mystmatch.append(measured.at[k, "Mean"])
                energymatchuncertainty.append(known.at[i, "EnergyUncert"])
                mystmatchuncertainty.append(measured.at[k, "MeanError"])
                nuclidematch.append(known.at[i, "Source"])
                intensitymatch.append(measured.at[k, "Amplitude"])
                chisss.append(measured.at[k, "ReducedChisq"])
                num1 = known.at[i, "Source"].split('-')
                try:
                    num2 = num1[2].split(' ')
                    formattednuclide.append(r'$\isotope[{}][{}]{{{}}}$ {} {}'.format(num2[0], num1[0], num1[1], num2[1], num2[2]))
                except IndexError:
                    try:
                        formattednuclide.append(r'$\isotope[{}][{}]{{{}}}$'.format(num1[2], num1[0], num1[1]))
                    except IndexError:
                        formattednuclide.append(known.at[i, "Source"])
                continue
    match_frame = pd.DataFrame({'Accepted energy': energymatch, 'Accepted uncertainty': energymatchuncertainty,
                                'Measured energy': mystmatch, 'Measured uncertainty': mystmatchuncertainty,
                                'Nuclide':         formattednuclide, 'Height': intensitymatch})
    match_frame.to_csv('Matches/oldMatches/{}_{}.csv'.format(type, today))
    match_frame.to_csv('Matches/{}.csv'.format(type))

    with open('Latex/OldVersions/{}Match_{}.tex'.format(type, today), 'w') as texfile:
        texfile.write('\\documentclass[a4paper,twoside]{article}\n')
        texfile.write('\\usepackage[margin=0in]{geometry}\n')
        texfile.write('\\usepackage{isotope}\n')
        texfile.write('\\usepackage{ltxtable}\n')
        texfile.write('\\usepackage{ltablex}\n')
        texfile.write('\\usepackage{longtable}\n')
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
    with open('Latex/OldVersions/{}MatchList_{}.tex'.format(type, today), 'w') as texfile:
        texfile.write('\\documentclass[a4paper,twoside]{article}\n')
        texfile.write('\\usepackage[margin=0in]{geometry}\n')
        texfile.write('\\usepackage{mathtools}\n')
        texfile.write('\\usepackage[math]{cellspace}\n')
        texfile.write('\\usepackage{isotope}\n')
        texfile.write('\\usepackage{longtable}\n')
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

    with open('Latex/{}Match.tex'.format(type), 'w') as texfile:
        texfile.write('\\documentclass[a4paper,twoside]{article}\n')
        texfile.write('\\usepackage[margin=0in]{geometry}\n')
        texfile.write('\\usepackage{isotope}\n')
        texfile.write('\\usepackage{ltxtable}\n')
        texfile.write('\\usepackage{ltablex}\n')
        texfile.write('\\usepackage{longtable}\n')
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
    with open('Latex/{}MatchList.tex'.format(type), 'w') as texfile:
        texfile.write('\\documentclass[a4paper,twoside]{article}\n')
        texfile.write('\\usepackage[margin=0in]{geometry}\n')
        texfile.write('\\usepackage{mathtools}\n')
        texfile.write('\\usepackage[math]{cellspace}\n')
        texfile.write('\\usepackage{isotope}\n')
        texfile.write('\\usepackage{longtable}\n')
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


trinititePeaks = pd.read_csv("Peaks/ShieldedTrinititeFIN2.csv")
acceptedPeaks = pd.read_csv("Energies8.csv", names=["Source", "Energy", "EnergyUncert", "Prob", "ProbUncert", "Notes"])
acceptedPeaks.replace(r'^\s*$', np.nan, inplace=True, regex=True)
acceptedPeaks = acceptedPeaks[pd.notnull(acceptedPeaks['Energy'])]
acceptedPeaks = acceptedPeaks[pd.notnull(acceptedPeaks['EnergyUncert'])]
print(acceptedPeaks[pd.notnull(acceptedPeaks['EnergyUncert'])])
acceptedPeaks = acceptedPeaks[pd.notnull(acceptedPeaks['Prob'])]
acceptedPeaks = acceptedPeaks[pd.notnull(acceptedPeaks['ProbUncert'])]
acceptedPeaks.reset_index(inplace=True)
# acceptedPeaks = acceptedPeaks.replace('-', '-')
print(acceptedPeaks.iloc[570:580])
acceptedPeaks["Energy"] = pd.to_numeric(acceptedPeaks["Energy"])
acceptedPeaks["EnergyUncert"] = pd.to_numeric(acceptedPeaks["EnergyUncert"])
acceptedPeaks["Prob"] = pd.to_numeric(acceptedPeaks["Prob"])
acceptedPeaks["ProbUncert"] = pd.to_numeric(acceptedPeaks["ProbUncert"])
datamatch(trinititePeaks, acceptedPeaks, 2, "Trinitite_FIN2", 1)
