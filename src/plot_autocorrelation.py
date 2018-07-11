import os, glob, joblib, textwrap
import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import get_cloud_size_dist as get_cloud
import generate_timeseries as gen_times
from scipy import signal

from sklearn import linear_model as lm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, \
                                            ExpSineSquared, RationalQuadratic

def find_nearest(array, value):
    ''' Find nearest value is an array '''
    idx = (np.abs(array-value)).argmin()
    return idx

def get_fft_matrix(a_):
    sp = np.abs(np.fft.rfft(a_))
    freq = np.fft.rfftfreq(a_.shape[-1], d=1)
    return freq, sp

def calc_circular_corr(x, y):
    return np.fft.ifft(np.fft.fft(x).conj() * np.fft.fft(y)).real

def save_timeseries():
    c_, i_ = [], []
    time = np.arange(0, 540)
    for t in time:
        df = pq.read_table(f'tracking/clouds_{t:08d}.pq', nthreads=6).to_pandas()
        df_size = get_cloud.get_cloud_area(df)
        model = get_cloud.calc_cloud_slope(df_size, mute=True)

        c_.append(-model.coef_[0])
        i_.append(model.intercept_)

    joblib.dump([c_, i_], 'timeseries.pkl')
    return np.array(c_), np.array(i_)

def load_timeseries():
    a_ = joblib.load('timeseries.pkl')
    return np.array(a_[0]), np.array(a_[1])

def get_slope_timeseries():
    try:
        c_, i_ = load_timeseries()
    except:
        c_, i_ = save_timeseries()

    # Advanced detrending 
    n_features = 4
    gp = lm.BayesianRidge()
    gp.fit(np.vander(np.linspace(0, 540, 540), n_features), c_)

    y_mean = gp.predict(np.vander(np.linspace(0, 540, 540), n_features))
    c_det = c_ - y_mean

    # kernel = 1.0 * RBF(length_scale_bounds=(1e3, 1e5)) \
    #         + 1.0 * WhiteKernel(noise_level=1e-5) \
    #         + 1.0 * ExpSineSquared(length_scale_bounds=(0.1, 10), periodicity=78)
    # gp = GaussianProcessRegressor(kernel=kernel,
    #                               normalize_y=False,
    #                               n_restarts_optimizer=0)

    X_ = np.arange(540)
    # gp.fit(X_[:, None], np.array(c_det[:540]))

    #---- Plotting 
    fig = plt.figure(1, figsize=(8, 9))
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

    plt.subplot(3, 1, 1)
    plt.title('Cloud Size Timeseries')
    plt.ylabel('Cloud Slope $b$')
    plt.xlabel('Time [min]')

    X = np.linspace(0, 539, 540)
    # y_mean, y_std = gp.predict(X[:, None], return_std=True)
    plt.plot(X_, c_det, '--o')
    # plt.plot(X, y_mean, 'k', lw=1, zorder=9)
    # plt.fill_between(X, y_mean - y_std, y_mean + y_std,
    #                  alpha=0.2, color='k')
    # print(gp.kernel_)
    # print(gp.log_marginal_likelihood_value_)
    y_mean = c_det
    # y_samples = gp.sample_y(X[:, None], 10)
    # plt.plot(X, y_samples, lw=0.5, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.title('Periodogram')
    plt.xlabel('Frequency [1/min]')
    plt.ylabel('Power')

    # xf, yf = get_fft_matrix(y_mean)
    xf, yf = signal.periodogram(y_mean)
    plt.plot(xf, yf, '--o')

    yp = yf[find_nearest(xf, 0.026)]
    plt.annotate('T = 45', xy=(0.026+0.002, yp-5e-3), xytext=(0.05, 0.4), 
                arrowprops=dict(), ha='center')

    plt.xlim([0, 0.2])

    candidates = 1/xf[np.argpartition(yf, -10)[-10:]]
    print(f'{candidates}')

    plt.subplot(3, 1, 3)
    plt.title('Autocorrelation')
    plt.xlabel('Periodicity [min]')
    plt.ylabel('Power')
    
    yf = calc_circular_corr(y_mean, y_mean)
    xf = np.arange(0, yf.size//2, 1)
    yf = yf[yf.size//2:]
    plt.plot(xf, yf, '--.')
    plt.xlim([0, 180])
    for item in candidates:
        plt.plot(item, yf[find_nearest(xf, item)], 'r.', ms=12)

    yp = yf[find_nearest(xf, 45)] + 0.1
    plt.annotate('T = 45', xy=(45, yp), xytext=(30, 1.5), ha='center',
                arrowprops=dict())
    
    yp = yf[find_nearest(xf, 78)]
    plt.annotate('T = 78', xy=(78, yp), xytext=(95, 1.5), ha='center',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0.3'))

    plt.ylim([-1.5, 2])

    plt.tight_layout(pad=0.5)
    figfile = 'png/{}.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=180, \
                facecolor='w', transparent=True)

if __name__ == '__main__':
    get_slope_timeseries()