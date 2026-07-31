from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import csv, glob, os, sys, time, h5py, gc
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.spatial import cKDTree
#import src.tmp_library as tml
import src.mult_clouds as mcls
import warnings
from mpi4py import MPI         
import pickle
import asyncio
"""
global __alloc_slots__, __sample_size__, __rloc__, FLAG0, FLAG1, FLAG2, FLAG3
global __dense_cloud__, __threshold__, __start_time__, __start_snap__

__start_snap__  = 0
__start_time__  = 0.0 # Myrs
__alloc_slots__ = 1000   #
__sample_size__ = 100    # N
__rloc__        = 1.0e-1 # parsec
__dense_cloud__ = 1.0e+2 # per cm^3
__threshold__   = 1.0e+2 # per cm^3

FLAG0 = "-lin" # dump field lines
FLAG1 = "-exp" # rloc analysis
FLAG2 = "-all" # testing, use all snapshots
FLAG3 = "-weight" # use weighted uniform points instead of uniform
"""


def match_data_to_files(input_case, range_function = np.array(list(range(0,500)))):
    """
    range_function : contains a list with all available positions in of each cloud
    """
    if input_case == 'ideal':
        subdirectory = 'ideal_mhd'
        file_hdf5 = sorted(np.array(glob.glob(f'arepo_data/{subdirectory}/*.hdf5')))
    elif input_case == 'amb':
        subdirectory = 'ambipolar_diffusion'
        file_hdf5 = sorted(np.array(glob.glob(f'arepo_data/{subdirectory}/*.hdf5')))

    #assert os.path.exists(file_hdf5), f"[{file_hdf5} cloud data not present]"

    available_snapshot_files = np.zeros_like(range_function) == 1

    for snap_file in file_hdf5:
        snap_val = snap_file.split(".")
        available_snapshot_files[int(snap_val[-2][-3:])] = True
        
    return available_snapshot_files, np.array(file_hdf5)

def compute_reduction_factor_in_parallel(centers: np.array, snaps: np.array, file_hdf5: np.array, index_cloud: int, output_name: str) -> dict:
    
    print(f"""
    Initiating Analysis of Cloud: {index_cloud}
    Output pickle file name     : {output_name} 
    """, flush = True)

    start_time = time.time()

    # verify compataibility between data
    assert len(file_hdf5) == centers.shape[0] == len(snaps), "Arrays must all have the same length"

    # load hdf5 file
    survivors_fraction = np.zeros(file_hdf5.shape[0])
    df_stats = dict()
    df_fields= dict()
    
    for each, (center, snap, filename) in enumerate(zip(centers, snaps, file_hdf5)):
        mcls.mult_clouds_config_arepo(filename, center)
        print(mcls._time)

        #print(center, snap, filename, mcls.Density.shape)
        print(f"""
        Filename           : {filename}
        Snapshot Number    : {snap}
        In Simulation Time : {mcls._time}
        Cloud Coordinates  : {center}        
        """, flush = True)

        tree = cKDTree(mcls.Pos)

        try:
            x_input    = mcls.uniform_in_3d_tree_dependent(tree, mcls.__sample_size__, rloc=mcls.__rloc__, n_crit=mcls.__dense_cloud__)   
        except Exception as e:
            warnings.warn(f"[snap={snap}]", e)
            warnings.warn(f"[snap={snap}] At current snapshots, no cloud above {mcls.__dense_cloud__} cm-3", Warning)
            print(f"[Snap] snap {filename}: skipping", flush=True)
            mcls.config_arepo(filename, center, True)
            continue

        if x_input is None:
            print(f"[Snap] snap {snap}: skipping", flush=True)
            mcls.config_arepo(filename, center, True)
            continue

        directions=mcls.fibonacci_sphere(20)        # dodecahedron (12 faces), and icosahedron (20 faces)
        try:
            #__0, __1, mean_column, median_column = mcls.line_of_sight(x_init=x_input, directions=directions, n_crit=mcls.__threshold__)
            __0, __1, mean_column, median_column = mcls.edge_to_p_line_of_sight(x_init=x_input, directions=directions, n_crit=mcls.__threshold__)
        except Exception as e:
            warnings.warn(f"[snap={snap}]", e)
            print(f"[LOS] Invalid result from intergration: {snap}: skipping", flush=True)
            mcls.config_arepo(filename, center, True)
            continue
        
        mcls.__threshold__ = 10
        print(mcls.__threshold__)
        
        try:
            radius_vectors, magnetic_fields, numb_densities, follow_index, path_column, survivors1 = mcls.crs_path(x_init=x_input, n_crit=mcls.__threshold__)
            assert np.any(numb_densities > mcls.__threshold__), f"No values above threshold {mcls.__threshold__} cm-3"
        except Exception as e:
            warnings.warn(f"[snap={snap}]", e)
            print(f"[CRS] Invalid result from intergration: {snap}: skipping", flush=True)
            mcls.config_arepo(filename, center, True)
            continue

        #if np.log10(mcls.__threshold__) < 2: 
        r_u, n_rs, B_rs, survivors2 = mcls.eval_reduction(magnetic_fields, numb_densities, follow_index, mcls.__threshold__*10)
        r_l, _1, _2, _3 = mcls.eval_reduction(magnetic_fields, numb_densities, follow_index, mcls.__threshold__)

        
        survivors = np.logical_and(survivors1, survivors2)

        print(np.sum(survivors)/survivors.shape[0], " Survivor fraction", flush=True)

        print("__alloc_slots__: ", mcls.__alloc_slots__, flush=True)
        print("__used_slots__ : ",survivors1.shape, flush=True)

        survivors_fraction[each] = np.sum(survivors)/survivors.shape[0]
        u_input         = x_input[np.logical_not(survivors),:] # pc
        x_input         = x_input[survivors,:]                 # pc
        radius_vectors  = radius_vectors[:, survivors, :]      # pc
        numb_densities  = numb_densities[:, survivors]         # cm-3
        magnetic_fields = magnetic_fields[:, survivors]*mcls.gauss_code_to_gauss_cgs # Gauss CGS
        mean_column     = mean_column[survivors]               # cm-2
        median_column   = median_column[survivors]             # cm-2
        path_column     = path_column[survivors]               # cm-2
        n_rs            = n_rs[survivors]                      # cm-3
        B_rs            = B_rs[survivors]*mcls.gauss_code_to_gauss_cgs # Gauss CGS
        r_u             = r_u[survivors]                       # Adim
        r_l             = r_l[survivors]                       # Adim
        

        mean_r_u, median_r_u, skew_r_u, kurt_r_u = mcls.describes(r_u)
        mean_r_l, median_r_l, skew_r_l, kurt_r_l = mcls.describes(r_l)

        stats_dict = {
            "time": mcls._time, 
            "x_input": x_input,
            "n_rs": n_rs,
            "B_rs": B_rs,
            "n_path": path_column,
            "n_los0": mean_column,
            "n_los1": median_column,
            "surv_fraction": survivors_fraction[each],
            "r_u": r_u,
            "r_l": r_l
        }

        df_stats[str(snap)]  = stats_dict
        
        if 'Pos' in globals():
            print("\nPos is global", flush=True)

        mcls.get_globals_memory()            
        # at the end of the loop, drop all that will be reasigned, to avoid memory overflow
        del tree, _1, _2, _3, __0, __1
        del radius_vectors, magnetic_fields, numb_densities

        gc.collect()
        mcls.config_arepo(filename, center, True)
        mcls.get_globals_memory()

        # create tmp_files

    elapsed_time =time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    return df_stats, df_fields


with open('./util/cloud_matches.pkl', 'rb') as f:
    data_cloud_pairs = pickle.load(f)

COORD_COLS = ["c_coord_X", "c_coord_Y", "c_coord_Z"]
CLOUD_SNAP_CENTER = ["snap"] + COORD_COLS

#mcls.config_ic(input_file)

ideal_mask, ideal_hdf5 = match_data_to_files("ideal", range_function = np.array(list(range(0,500))))
amb_mask, amb_hdf5   = match_data_to_files("amb", range_function = np.array(list(range(0,500))))

from concurrent.futures import ProcessPoolExecutor

os.makedirs("others", exist_ok=True)

def process_cloud(index_cloud, cloud_data):
    print(index_cloud)
    ideal_output_name = f"ideal_{index_cloud}th_cloud.pkl"
    amb_output_name   = f"amb_{index_cloud}th_cloud.pkl"

    ideal_data = cloud_data[0].sort_values("snap")[CLOUD_SNAP_CENTER]
    amb_data   = cloud_data[1].sort_values("snap")[CLOUD_SNAP_CENTER]

    ideal_centers = ideal_data[COORD_COLS][ideal_mask].to_numpy()
    ideal_snaps   = ideal_data["snap"][ideal_mask].to_numpy()
    amb_centers   = amb_data[COORD_COLS][amb_mask].to_numpy()
    amb_snaps     = amb_data["snap"][amb_mask].to_numpy()

    ideal_stats, ideal_fields = compute_reduction_factor_in_parallel(
        ideal_centers, ideal_snaps, ideal_hdf5, index_cloud, ideal_output_name
    )
    amb_stats, amb_fields = compute_reduction_factor_in_parallel(
        amb_centers, amb_snaps, amb_hdf5, index_cloud, amb_output_name
    )


    return index_cloud, ideal_stats, amb_stats

#result = process_cloud(data_cloud_pairs.keys(), data_cloud_pairs.values())
#print(data_cloud_pairs.keys())
#print(data_cloud_pairs.values())
with ProcessPoolExecutor(max_workers=2) as executor:
    df_stats = list(executor.map(
        process_cloud,
        data_cloud_pairs.keys(),
        data_cloud_pairs.values()
    ))

_id_ = "cldsia"

"""
list
└── tuple(4)
    ├── int                          # index
    ├── dict[str → dict]             # primary records
    │   └── {time, x_input, n_rs, B_rs, n_path,
    │          n_los0, n_los1, surv_fraction, r_u, r_l}
    ├── dict[str → dict]             # secondary records (same schema)
    └── dict                         # always empty {}
"""

print(df_stats[0])

if "HOSTNAME" in list(os.environ.keys()):
    os.makedirs(f"/work/bjencinasvelaz/series/{_id_}/", exist_ok=True)
    workdir = f"/work/bjencinasvelaz/series/{_id_}/tmp_{_id_}_rank.pkl"
    print("Output files saved at :", workdir, flush=True)

    with open(workdir, 'wb') as f:
        pickle.dump(df_stats, f)
        f.flush()
        os.fsync(f.fileno())
else:# -exp or -weight since each of this modify _id_ variable
    os.makedirs(f"./series/{_id_}/", exist_ok=True)
    workdir = f"./series/{_id_}/tmp_{_id_}.pkl"
    print("Output files saved at :", workdir, flush=True)

    with open(workdir, 'wb') as f:
        pickle.dump(df_stats, f)
        f.flush()
        os.fsync(f.fileno())

print(df_stats)

"""
import pickle
import numpy as np

def load_data(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

data = load_data("data.pkl")

# Quick sanity check
for tup in data:
    idx, primary, secondary, extra = tup
    print(f"[{idx}] primary keys: {list(primary.keys())}, secondary keys: {list(secondary.keys())}")
    for rec_id, rec in primary.items():
        print(f"  {rec_id}: time={rec['time']:.4f}, x_input={rec['x_input'].shape}, surv={rec['surv_fraction']}")

-----------------------------------------------------------------------------------------------------------------

if "HOSTNAME" in list(os.environ.keys()):
    os.makedirs(f"/work/bjencinasvelaz/series/{_id_}/", exist_ok=True)
    asyncio.run(mcls.merge_and_save(_id_, mcls.__dense_cloud__, f"/work/bjencinasvelaz/series/{_id_}/"))
else:
    os.makedirs(f"./series/{_id_}/", exist_ok=True)
    asyncio.run(mcls.merge_and_save(_id_, mcls.__dense_cloud__, f"./series/{_id_}"))

"""
