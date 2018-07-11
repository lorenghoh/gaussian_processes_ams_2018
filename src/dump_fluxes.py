from netCDF4 import Dataset as nc
from collections import defaultdict as dd
import glob, os, ujson
import numpy as np

import pyarrow.parquet as pq 
import pandas as pd

import tempfile
from joblib import Parallel, delayed

MF = np.empty((512, 512, 128), dtype=np.float_)
ET = np.empty((512, 512, 128), dtype=np.float_)

def get_cloud_top_height(ind):
    dx = 512
    z = ind.coord // (dx * dx)
    xy = ind.coord % (dx * dx)
    y = xy // dx 
    x = xy % dx

    df = pd.DataFrame(columns=['area', 'mf', 'ent'])
    for k in np.unique(z):
        m_ = (z == k)
        
        area_ = np.sum(m_)
        mf_ = np.sum(MF[z[m_], y[m_], x[m_]])
        ent_ = np.sum(ET[z[m_], y[m_], x[m_]])
        df = df.append(
                pd.DataFrame([{'area': area_, 'mf': mf_, 'ent': ent_}]))
    return df

def main():
    loc = '/Howard16TB/data/loh/BOMEX_12HR'
    core_list = sorted(glob.glob(f'{loc}/core_entrain/*.nc'))
    var_list = sorted(glob.glob(f'{loc}/variables/*.nc'))

    loc = '/scratchSSD/loh/tracking/BOMEX_12HR'
    tracking_list = sorted(glob.glob(f'{loc}/clouds_*.pq'))
    try:
        time_steps = np.arange(len(tracking_list))
        if len(time_steps) <= 0:
            raise ValueError("No tracking data found")
    except:
        raise

    for t, ncf in enumerate(tracking_list):
        # Initialize dictionary
        data = dd(list)

        with nc(core_list[t]) as core_file:
            df = pq.ParquetDataset(tracking_list[t]).read(nthreads=6).to_pandas()
            df = df[(df.type == 4)]

            cids = df.cid.value_counts(sort=False)
            cids = cids.index[cids > 4]

            global MF
            global ET

            MF = core_file['MFTETCOR'][0, :]
            ET = core_file['ETETCOR'][0, :]

        with Parallel(n_jobs=16) as Pr:
            result = Pr(delayed(get_cloud_top_height)
                (df.loc[df['cid'] == cid]) for cid in cids)
            df = pd.concat(result, ignore_index=True)
            df.to_parquet(f'pq/core_dump_{t:03d}.pq', engine='pyarrow')

if __name__ == "__main__":
    main()
