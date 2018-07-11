import os, glob, joblib
import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

import get_cloud_size_dist as get_cloud
from joblib import Parallel, delayed

def estimate_slope(file):
    df = pq.read_table(file, nthreads=4).to_pandas()
    df_size = get_cloud.get_cloud_area(df)
    model = get_cloud.calc_cloud_slope(df_size, mute=True)

    return -model.coef_[0], model.intercept_

def save_timeseries(loc, case_name):
    c_, i_ = [], []
    
    f_list = sorted(glob.glob(f'{loc}/{case_name}/clouds_*.pq'))
    with Parallel(n_jobs=8) as Pr:
        result = Pr(delayed(estimate_slope)(f) for f in f_list)
        df = pd.DataFrame.from_records(result, columns=['c_', 'i_'])

    joblib.dump([df.c_.values, df.i_.values], f'timeseries_{case_name}.pkl')
    return df

if __name__ == '__main__':
    loc = '/scratchSSD/loh/tracking'
    case_name = 'BOMEX'
    df = save_timeseries(loc, case_name)

    # Test output
    a_ = joblib.load('timeseries.pkl')
    print(pd.DataFrame({'c_': a_[0], 'i_': a_[1]}).head())
