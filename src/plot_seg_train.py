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

def save_timeseries():
    c_, i_ = [], []
    time = np.arange(0, 540)
    for t in time:
        df = pq.read_table(f'/nodessd/loh/tracking/BOMEX_12HR/clouds_{t:08d}.pq', nthreads=6).to_pandas()
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

    kernel = 1.0 * RBF(length_scale_bounds=(1e4, 1e5)) \
            + 1.0 * WhiteKernel(noise_level=1e-5) \
            + 1.0 * ExpSineSquared(length_scale_bounds=(0.1, 10), periodicity=78, periodicity_bounds=(75, 85)) \
            + 1.0 * ExpSineSquared(length_scale_bounds=(0.1, 10), periodicity=45, periodicity_bounds=(40, 55))
    gp = GaussianProcessRegressor(kernel=kernel,
                                  normalize_y=False,
                                  n_restarts_optimizer=5)

    # Every 60 minutes training, followed by 30 minutes of testing
    X_ = np.arange(540)
    m_ = np.ones_like(X_, dtype=np.bool_)
    m_[60:90] = False
    m_[180:210] = False
    m_[270:300] = False
    m_[360:390] = False
    m_[450:480] = False
    gp.fit(np.ma.array(X_[:, None], mask=~m_), np.ma.array(c_det[:540], mask=~m_))

    #---- Plotting 
    fig = plt.figure(1, figsize=(8, 3))
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
    plt.ylabel('Cloud Slope $b$')
    plt.xlabel('Time [min]')

    plt.plot(np.arange(540), c_det, '--o', lw=0.75)

    X = np.linspace(1, 540, 540)
    y_mean, y_std = gp.predict(X[:, None], return_std=True)
    plt.plot(X, np.ma.array(y_mean, mask=~m_), 'k', lw=1, zorder=9, label='Training Data')
    plt.plot(np.arange(60, 90), y_mean[60:90], 'r--', label='Testing Data')
    plt.plot(np.arange(180, 210), y_mean[180:210], 'r--')
    plt.plot(np.arange(270, 300), y_mean[270:300], 'r--')
    plt.plot(np.arange(360, 390), y_mean[360:390], 'r--')
    plt.plot(np.arange(450, 480), y_mean[450:480], 'r--')
    plt.fill_between(X, y_mean - y_std, y_mean + y_std,
                     alpha=0.2, color='k')
    print(gp.kernel_)
    print(gp.log_marginal_likelihood_value_)
    plt.legend()

    y_samples = gp.sample_y(X[:, None], 10)
    plt.plot(X, y_samples, lw=0.5, alpha=0.2)

    plt.tight_layout(pad=0.5)
    figfile = 'png/{}.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=180, \
                facecolor='w', transparent=True)

if __name__ == '__main__':
    get_slope_timeseries()