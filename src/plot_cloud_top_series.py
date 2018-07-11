import os, glob, joblib, ujson
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
from joblib import Parallel, delayed

import plot_fft as fft_

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, \
                                            ExpSineSquared, RationalQuadratic

cid = 3297

def get_cloud_top_height(file):
    df = pq.ParquetDataset(file).read(nthreads=6).to_pandas()
    df = df[(df.cid == cid) & (df.type == 4)]

    # Calculate z index from coordinates
    df['z'] = df.coord // (256 * 256)
    return np.max(df['z'])

def plot_cloud_top_series():
    loc = '/scratchSSD/loh/tracking'
    case_name = 'BOMEX'
    f_list = sorted(glob.glob(f'{loc}/{case_name}/clouds_*.pq'))

    with open('unique_clouds.json', 'r') as f:
        c_dict = ujson.load(f)
        
    c_list = [f_list[i] for i in c_dict[f'{cid}']]

    with Parallel(n_jobs=16) as Pr:
        result = Pr(delayed(get_cloud_top_height)(f) for f in c_list)

    #---- Plotting 
    fig = plt.figure(1, figsize=(6, 3))
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
    plt.rc('font', family='Helvetica')

    ax = plt.subplot(1, 1, 1)
    plt.ylabel(r'$d \hat{\mathcal{H}} / dt$')
    plt.xlabel('Time [min]')

    # xf, yf = fft_.get_fft_matrix(h_)
    # plt.plot(xf, yf, '--o')
    plt.plot(t_, h_, '--o', lw=0.75)

    # Plot GP regression result
    kernel = 1.0 * RBF(length_scale=1e5, length_scale_bounds=(1e3, 1e5)) \
            + 1.0 * WhiteKernel(noise_level=1e-2) \
            + 1.0 * ExpSineSquared(periodicity=15.8, periodicity_bounds=(5, 25))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
    gp.fit(t_[:, None], h_[:])

    X = np.linspace(t_[0], t_[-1], 180)
    y_mean, y_std = gp.predict(X[:, None], return_std=True)
    plt.plot(X, y_mean, 'k', lw=1, zorder=9)
    plt.fill_between(X, y_mean - y_std, y_mean + y_std,
                     alpha=0.2, color='k')
    print(gp.kernel_)
    print(gp.log_marginal_likelihood_value_)

    y_samples = gp.sample_y(X[:, None], 10)
    plt.plot(X, y_samples, lw=0.5, alpha=0.3)

    plt.tight_layout(pad=0.5)
    figfile = 'png/{}.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=180, \
                facecolor='w', transparent=True)

if __name__ == '__main__':
    plot_cloud_top_series()