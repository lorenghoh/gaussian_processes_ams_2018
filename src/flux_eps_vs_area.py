import os, glob, joblib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import Parallel, delayed

from sklearn import linear_model as lm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, \
                                            ExpSineSquared, RationalQuadratic

def save_timeseries(f_in):
    df = pq.ParquetDataset(f_in).read(nthreads=4).to_pandas()
    
    df['area_log'] = np.log(df.area)
    df['mf_log'] = np.log(df.mf)
    df['ent_log'] = np.log(df.ent)
    df['eps_log'] = np.log(df.ent / df.mf)

    df.area_log[df.area_log < 1] = np.nan

    for item in ['area_log', 'mf_log', 'ent_log', 'eps_log']:
        m_ = np.isfinite(df['area_log']) & np.isfinite(df[item])
        df[item] = df[item][m_]

    return df

def find_slope():
    pass

def plot_flux_timeseries():
    try:
        df = pq.ParquetDataset('core_eps.pq').read(nthreads=6).to_pandas()
    except:
        dump_list = sorted(glob.glob(('pq/core_dump_*.pq')))

        with Parallel(n_jobs=12) as Pr:
            result = Pr(delayed(save_timeseries)(f_in) for f_in in dump_list)
            df = pd.concat(result, ignore_index=True)
            df.to_parquet('core_eps.pq', engine='pyarrow')

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
    # plt.rc('text', usetex=True)
    plt.rc('font', family='Serif')

    df.eps_log[(df.eps_log < -8) & (df.eps_log > -2)] = np.nan

    df.plot.scatter(x='area_log', y='eps_log', ylim=(-8, -2))

    #---- Regression
    n_features = 4

    m_ = np.isfinite(df.area_log) & np.isfinite(df.eps_log)
    x_ = df.area_log[m_]
    y_ = df.eps_log[m_]

    gp = lm.BayesianRidge(compute_score=True)
    gp.fit(np.vander(x_, n_features), y_)

    print(gp.coef_)
    print(f'Intercept: {gp.intercept_}')
    print(gp.scores_)

    X = np.linspace(np.min(x_), np.max(x_), 35)
    y_mean, y_std = gp.predict(np.vander(X, n_features), return_std=True)
    plt.fill_between(X, y_mean - y_std, y_mean + y_std, alpha=0.05)

    plt.plot(X, y_mean, 'k', zorder=10, lw=2)

    plt.tight_layout(pad=0.5)
    figfile = 'png/{}.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=180, \
                facecolor='w', transparent=True)

if __name__ == '__main__':
    plot_flux_timeseries()