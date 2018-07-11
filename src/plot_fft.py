import os, glob, joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import signal

def detrend(a_):
    return signal.detrend(a_)

def get_fft_matrix(a_):
    target = detrend(np.array(a_))

    sp = np.abs(np.fft.rfft(target))
    freq = np.fft.rfftfreq(target.size, d=1)

    return 1/freq, sp

if __name__ == '__main__':
    c_, i_ = joblib.load('timeseries.pkl')

    #---- Plotting 
    fig = plt.figure(1, figsize=(4.5, 3))
    fig.clf()
    sns.set_context('paper')
    sns.set_style('ticks', 
        {
            'axes.grid': False, 
            'axes.linewidth': '0.75',
            'grid.color': '0.75',
            'grid.linestyle': u':',
            'legend.frameon': True,
        })
    plt.rc('text', usetex=True)
    plt.rc('font', family='Serif')

    ax = plt.subplot(1, 1, 1)
    plt.xlabel("Frequency [min]")

    cmap = sns.cubehelix_palette(start=1.2, hue=1, \
                                 light=1, rot=-1.05, as_cmap=True)

    xf, yf = get_fft_matrix(c_)
    plt.plot(xf, yf, '--o')
    plt.xlim([0, 180])

    plt.tight_layout(pad=0.5)
    figfile = 'png/{}.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=180, \
                facecolor='w', transparent=True)