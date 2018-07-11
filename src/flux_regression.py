import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import glob, os, ujson

import numpy as np
import pyarrow.parquet as pq 
import pandas as pd

from sklearn import linear_model as lm

def flux_regression():
    #---- Plotting 
    fig = plt.figure(1, figsize=(4, 4))
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

    dump_list = sorted(glob.glob(('pq/*.pq')))
    df = pq.ParquetDataset(dump_list[500]).read(nthreads=6).to_pandas()

    df['area_log'] = np.log(df.area)
    df['mf_log'] = np.log(df.mf)

    df.area_log[df.area_log < 1] = np.nan
    # df.mf_log[df.mf_log < -5] = np.nan
    df.plot.hexbin(x='area_log', y='mf_log', gridsize=25, ylim=[-4, 8])

    plt.title(r'features = 2 (Linear)')
    plt.xlabel(r'$\log_{10}$ Area', fontsize=12)
    plt.ylabel(r'$\log_{10}$ MF', fontsize=12)

    #---- Regression
    n_features = 2

    m_ = np.isfinite(df.area_log) & np.isfinite(df.mf_log)
    x_ = df.area_log[m_]
    y_ = df.mf_log[m_]

    gp = lm.BayesianRidge(compute_score=True)
    gp.fit(np.vander(x_, n_features), y_)

    print(gp.coef_)
    print(f'Intercept: {gp.intercept_}')
    print(gp.scores_)

    X = np.linspace(np.min(x_), np.max(x_), 35)
    y_mean, y_std = gp.predict(np.vander(X, n_features), return_std=True)
    plt.fill_between(X, y_mean - y_std, y_mean + y_std,
                     alpha=0.05)

    plt.plot(X, y_mean, lw=2)

    plt.tight_layout(pad=0.5)
    figfile = 'png/{}.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=180, \
                facecolor='w', transparent=True)

if __name__ == '__main__':
    flux_regression()
