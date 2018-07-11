import os, glob, itertools, ujson
import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

from joblib import Parallel, delayed
from collections import defaultdict

def get_unique_clouds(time, file):
    df = pq.ParquetDataset(file).read(nthreads=6).to_pandas()
    return time, df.cid.unique()

if __name__ == '__main__':
    # Return unique list of cids, parallelized by joblib
    filelist = sorted(glob.glob(f'tracking/clouds_*.pq'))

    with Parallel(n_jobs=8) as Pr:
        result = Pr(delayed(get_unique_clouds)(time, file) 
                    for time, file in enumerate(filelist))
    c_dict = dict(result)

    # Take cids of 30 largest clouds 
    counts = np.unique(list(itertools.chain(*list(c_dict.values()))), 
                       return_counts=True)
    cids = counts[0][counts[1].argsort()[::-1][:30]]

    d = defaultdict(list)
    for key in c_dict.keys():
        matches = set(cids).intersection(c_dict[key])
        for i in matches:
            d[i].append(key)

    with open('unique_clouds.json', 'w') as out_f:
        ujson.dump(d, out_f, sort_keys=True, indent=4)