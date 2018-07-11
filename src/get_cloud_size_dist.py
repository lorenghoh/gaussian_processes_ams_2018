import os, glob
import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge

def get_cloud_area(df):
    df = df[df.type == 0]
    df['z'] = df.coord // (512 * 512)

    # Size as a function of (cid, z)
    # Return size as a function of (cid, z)
    df_size = df.groupby(['cid', 'z']).size().reset_index(name='area')
    df_size['area'].apply(lambda x: x * 25**2)

    return df_size

def calc_cloud_slope(df, mute=False):
    hist, bin_edges = np.histogram(df['area'], bins=135)

    m_ = (hist > 0)
    x, y = np.log10(bin_edges[1:][m_]), np.log10(hist[m_])

    model = Ridge(fit_intercept=True)
    X = x[:, np.newaxis]
    model.fit(X, y)

    if (mute == False):
        print(f"\t coef_: {model.coef_[0]:.05f}, int_: {model.intercept_:.05f}")

    return model

if __name__ == '__main__':
    # Use ridge regression method to estimate the 
    # slope of the cloud size distribution
    df = pq.read_table(f'tracking/clouds_{120:08d}.pq', nthreads=6).to_pandas()

    df_size = get_cloud_area(df)
    model = calc_cloud_slope(df_size)

    hist, bin_edges = np.histogram(df_size['area'], bins='fd')
    m_ = (hist > 0)
    x, y = np.log10(bin_edges[1:][m_]), np.log10(hist[m_])

    #---- Plotting 
    fig = plt.figure(1, figsize=(3, 3))
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
    plt.xlabel(r'$\log_{10}$ Area')
    plt.ylabel(r'$\log_{10}$ Count')

    cmap = sns.cubehelix_palette(start=1.2, hue=1, \
                                 light=1, rot=-1.05, as_cmap=True)

    plt.plot(x, y, '--o', lw=0.75)

    y_fit = model.predict(x[:, None])
    plt.plot(x, y_fit)

    plt.tight_layout(pad=0.5)
    figfile = 'png/{}.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=180, \
                facecolor='w', transparent=True)