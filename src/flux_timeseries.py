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

    df.area_log[df.area_log < 1] = np.nan

    dd = {'mf':0, 'ent':0}
    for item in ['mf', 'ent']:
        m_ = np.isfinite(df['area_log']) & np.isfinite(df[f'{item}_log'])
        x_ = df['area_log'][m_]
        y_ = df[f'{item}_log'][m_]

        gp = lm.BayesianRidge(compute_score=True)
        gp.fit(np.vander(x_, 2), y_)

        dd[item] = gp.coef_[0]

    return pd.DataFrame([dd])

def find_slope():
    pass

def plot_flux_timeseries():
    try:
        df = pq.ParquetDataset('core_regression.pq').read(nthreads=6).to_pandas()
    except:
        dump_list = sorted(glob.glob(('pq/core_dump_*.pq')))

        with Parallel(n_jobs=6) as Pr:
            result = Pr(delayed(save_timeseries)(f_in) for f_in in dump_list)
            df = pd.concat(result, ignore_index=True)
            df.to_parquet('core_regression.pq', engine='pyarrow')

    #---- GP regression
    X_ = np.arange(540)
    c_ = df.mf

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

    n_features = 4
    gp = lm.BayesianRidge()
    gp.fit(np.vander(X_, n_features), c_)

    y_mean = gp.predict(np.vander(X_, n_features))
    c_det = c_ - y_mean
    plt.plot(X_, c_det, '--o', lw=0.75, label='Cloud Size Dist')

    a_ = joblib.load('timeseries.pkl')
    c_ = np.array(a_[0])

    n_features = 4
    gp = lm.BayesianRidge()
    gp.fit(np.vander(X_, n_features), c_)

    y_mean = gp.predict(np.vander(X_, n_features))
    c_det = c_ - y_mean

    plt.plot(X_, c_det, '--o', lw=0.75, label='Fractional Entrainment Dist')

    plt.xlabel('Time [min]')
    plt.legend(fontsize=12)

    plt.tight_layout(pad=0.5)
    figfile = 'png/{}.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=180, \
                facecolor='w', transparent=True)

if __name__ == '__main__':
    plot_flux_timeseries()