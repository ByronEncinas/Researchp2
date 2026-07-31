import numpy as np
import pandas as pd
from src.library import *
import pickle
import asyncio
import sys, os, glob

async def merge_and_save(abspath, identifier):
    __dense_cloud__ = identifier[0]
    _id_            = identifier[1:]
    
    loop = asyncio.get_event_loop()
    print(f'{abspath}/tmp_{_id_}_rank*.pkl')

    stat_files = sorted(glob.glob(f'{abspath}/tmp_{_id_}_rank*.pkl'))
    print(stat_files)

    if stat_files == []:
        print(f"Number of files available: {len(stat_files)}.", flush=True)
        return -1


    # Read all rank files concurrently
    def load_pickle(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    stat_tasks = [loop.run_in_executor(None, load_pickle, f) for f in stat_files]

    all_stats = await asyncio.gather(*stat_tasks)

    # Merge stats
    merged_stats = {}
    for d in all_stats:
        merged_stats.update(d)
        

    # Save final output
    df = pd.DataFrame.from_dict(merged_stats, orient='index')\
        .reset_index().rename(columns={'index': 'snapshot'})
    df.to_pickle(f'{abspath}/data_{__dense_cloud__}{_id_}.pkl')

    print(f"Merged {len(stat_files)} rank files successfully.", flush=True)

    for f in stat_files:
        #os.remove(f)
        print(f)

tmp_abspath = sys.argv[1]  # tmp/
identifier  = sys.argv[2]  # 6i0

asyncio.run(merge_and_save(tmp_abspath, identifier))