import glob, os, h5py, sys,csv
from functools import wraps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import asyncio
import warnings
from scipy.spatial import cKDTree
from copy import deepcopy
from scipy.stats import kurtosis, skew
from collections import Counter
# cache-ing spatial.cKDTree(Pos[:]).query(x, k=1)
_cached_tree = None
_cached_pos = None

def find_points_and_relative_positions(x, Pos, VoronoiPos):
    global _cached_tree, _cached_pos
    if _cached_tree is None or not np.array_equal(Pos, _cached_pos):
        _cached_tree = cKDTree(Pos)
        _cached_pos = Pos.copy()

    dist, cells = _cached_tree.query(x, k=1, workers=-1)
    rel_pos = VoronoiPos[cells] - x
    return dist, cells, rel_pos

def find_points_and_get_fields(x, Bfield, Density, Pos, VoronoiPos):
	dist, cells, rel_pos = find_points_and_relative_positions(x, Pos, VoronoiPos)
	local_fields = Bfield[cells] #get_magnetic_field_at_points(x, Bfield[cells], rel_pos) commented if no grad_bfields
	local_densities = Density[cells] #get_density_at_points(x, Density[cells], Density_grad[cells], rel_pos) commented if no grad_density
	abs_local_fields = np.sqrt(np.sum(local_fields**2,axis=1))
	return local_fields, abs_local_fields, local_densities, cells

def Heun_step(x, dx, Bfield, Density, Pos, VoronoiPos, Volume, bdirection=None, retrieve_cells = False):

    # campo en x, mangitud campo en x, densidad en x y ID de la celda
    local_fields_1, abs_local_fields_1, local_densities, cells = find_points_and_get_fields(
        x, Bfield, Density, Pos, VoronoiPos
    )

    # vector unitario en la dirección del campo en x
    local_fields_1 = local_fields_1 / np.tile(abs_local_fields_1, (3, 1)).T

    # Volume de la celda en la que está x
    CellVol = Volume[cells]

    # radio de esfera con volumen CellVol
    scaled_dx = dx * ((3/4) * CellVol / np.pi)**(1/3)

    # paso intermedio en x
    x_tilde = x + 0.5 * scaled_dx[:, np.newaxis] * local_fields_1

    # campo en x_intermedio, mangitud campo en x_intermedio, densidad en x_intermedio y ID de la celda intermedia
    local_fields_2, abs_local_fields_2, _, _ = find_points_and_get_fields(
        x_tilde, Bfield, Density, Pos, VoronoiPos
    )
    local_fields_2 = local_fields_2 / np.tile(abs_local_fields_2, (3, 1)).T

    x_final = x + 0.5 * scaled_dx[:, np.newaxis] * (local_fields_1 + local_fields_2)

    if retrieve_cells:
        return x_final, abs_local_fields_1, local_densities, CellVol, cells
    else:
        return x_final, abs_local_fields_1, local_densities, CellVol

mean_molecular_weight_ism = 2.35  # mean molecular weight of the ISM (Wilms, 2000)
density_code_to_nuclei_cm3 = 6.02214076e+23 / (2.35) * 6.771194847794873e-23  # Wilms, 2000 ; Padovani, 2018 ism mean molecular weight is # conversion from g/cm^3 to nuclei/cm^3
gauss_code_to_gauss_cgs = (4 * np.pi)**0.5   * (3.086e18)**(-1.5) * (1.99e33)**0.5 * 1e5 # cgs units
pc_to_cm = 3.086 * 1.0e+18  # cm per parsec
gauss_to_micro_gauss = 1e+6

# legacy
gr_cm3_to_nuclei_cm3 = 6.02214076e+23 / (2.35) * 6.771194847794873e-23  # Wilms, 2000 ; Padovani, 2018 ism mean molecular weight is # conversion from g/cm^3 to nuclei/cm^3

# global variables that can be modified from anywhere in the code
global FloatType, FloatType2, IntType

FloatType          = np.float64
FloatType2         = np.float128
IntType            = np.int32

def config_ic(input_file):
    global __rloc__, __sample_size__, __input_case__, __start_snap__
    global __dense_cloud__,__threshold__, __alloc_slots__, FLAG0, FLAG1, FLAG2, FLAG3
    
    FLAG0 = "-lin" # dump field lines
    FLAG1 = "-exp" # rloc analysis
    FLAG2 = "-all" # testing, use all snapshots
    FLAG3 = "-weight" # use weighted uniform points instead of uniform

    config = {}
    with open(input_file, 'r') as f:
        exec(f.read(), {}, config)
    globals().update(config) # This injects every key as a variable in your script

if len(sys.argv) > 1:

    config_ic(sys.argv[1])

def config_arepo(filename, center, close = False):
    if "snap" not in globals():
        global snap, __Boxsize__, Pos, Density, Mass
        global Volume, VoronoiPos, Bfield, data
        global MagneticFieldDivergence

        
    if close:
        data.close()
        return None
    
    snap = int(filename.split('.')[0][-3:])
    data = h5py.File(filename, 'r')
    __Boxsize__ = data['Header'].attrs['BoxSize']

    MagneticFieldDivergence = np.asarray(data['PartType0']['MagneticFieldDivergence'], dtype=FloatType)

    # units solar mass, parsec and km/s
    Pos = np.asarray(data['PartType0']['CenterOfMass'], dtype=FloatType)
    Density  = np.asarray(data['PartType0']['Density'], dtype=FloatType)
    Mass     = np.asarray(data['PartType0']['Masses'], dtype=FloatType)
    VoronoiPos = np.asarray(data['PartType0']['Coordinates'], dtype=FloatType)
    Bfield     = np.asarray(data['PartType0']['MagneticField'], dtype=FloatType)
    Volume   = Mass/Density

    # units cm-3
    Density    = Density*gr_cm3_to_nuclei_cm3 # density is in cm-3 from the get go

    VoronoiPos-=center
    Pos       -=center    

    for dim in range(3):
        pos_from_center = Pos[:, dim]
        too_high = pos_from_center > __Boxsize__ / 2
        too_low  = pos_from_center < -__Boxsize__ / 2
        Pos[too_high, dim] -= __Boxsize__
        Pos[too_low,  dim] += __Boxsize__

def fibonacci_sphere(samples=20):
    phi = np.pi * (3. - np.sqrt(5.))  # Golden angle
    y = np.linspace(1 - 1/samples, -1 + 1/samples, samples)  # Even spacing in y
    radius = np.sqrt(1 - y**2)  # Compute radius for each point
    theta = phi * np.arange(samples)  # Angle increment

    x = radius * np.cos(theta)
    z = radius * np.sin(theta)
    return np.vstack((x, y, z)).T  # Stack into a (N, 3) array

async def merge_and_save(_id_, __dense_cloud__, path = "series"):

    loop = asyncio.get_event_loop()
    stat_files = sorted(glob.glob(f'./{path}/tmp_{_id_}_rank*.pkl'))

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
    output_path = os.path.join(path, f"data_{int(np.log10(__dense_cloud__))}{_id_}.pkl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_pickle(output_path)
    
    print(f"Merged {len(stat_files)} rank files successfully.", flush=True)

    for f in stat_files:
        #os.remove(f)
        print(f)
    
def get_globals_memory() -> None:
    import sys

    total = 0
    for name, obj in globals().items():
        if name.startswith("__") and name.endswith("__"):
            continue  # skip built-in entries
        try:
            total += sys.getsizeof(obj)
        except TypeError:
            pass  # some objects might not report size

    # Convert bytes → gigabytes
    gb = total / (1024 ** 3)
    print(f"Memory used by globals: {gb:.6f} gigabytes", flush=True)

def timing(func):
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to run.")
        return result
    return wrapper

def count_cell_reentries(cell_indices, max_gap=50):
    # max_gap is the space between oscillations for which we add them
    #cell indices for a magnetic field line
    cells = list(cell_indices)
    if not cells:
        return {}
    #starts a dictonary to keep track of the visits, for example 
    # For example:

    #   [10, 10, 20, 20, 10]
    #   becomes:
    #   {
    #       10: [(0, 1), (4, 4)],
    #       20: [(2, 3)]
    #   }
    
    visits = {}
    visit_start = 0
    #iterate for all the cells indexes 
    for step in range(1, len(cells)):
        #if the cell index is not the same as the previous one, so that steps in the same field line are not considered to be visits
        if cells[step] != cells[step - 1]:
            visits.setdefault(cells[step - 1], []).append(
                (visit_start, step - 1)
            )
            visit_start = step
    visits.setdefault(cells[-1], []).append((visit_start, len(cells) - 1))

    #dictionary for the reentry counts
    reentry_counts = {}
    for cell_id, intervals in visits.items():
        best_count = 0
        current_count = 0

        for previous, current in zip(intervals, intervals[1:]):
            if current[0] - previous[1] <= max_gap:
                current_count += 1
                best_count = max(best_count, current_count)
            else:
                current_count = 0

        if best_count:
            reentry_counts[cell_id] = best_count

    return reentry_counts

def detect_oscillating_line(cell_indices, threshold=20, max_gap=50):
    reentry_counts = count_cell_reentries(cell_indices, max_gap=max_gap)
    return any(count >= threshold for count in reentry_counts.values())

@timing
def dense_segments_in_3d_tree_dependent(tree, Density, Pos, no_per_seg, rloc=1.0):

    # selected a randome sample of 1 k  for range 10^10 - 10^14
    sphere = Pos[:,0]*Pos[:,0] + Pos[:,1]*Pos[:,1]+Pos[:,2]*Pos[:,2] < rloc*rloc

    cellsAboveDensity = np.sum(Density[sphere] > 1.0e+2)

    print(cellsAboveDensity, np.log10(cellsAboveDensity))

    if (np.log10(cellsAboveDensity) < 3):
        print(f"Log of Cells in Core = ", np.log10(cellsAboveDensity), flush=True)
        print(f"Not enough cells to create a meaningful sample {sphere.shape[0] }")
        return None

    if (np.max(Density[sphere]) < 1.0e+2):
        print("No Densities above 1.0e+2 cm-3")
        return None

    max_den_snap = np.max(Density[sphere])
    min_den_snap = min(np.min(Density[sphere]), 100)
    no_step = 5
    width_step = (np.log10(max_den_snap)- np.log10(min_den_snap)) / no_step
    no_per_seg = no_per_seg // no_step
    
    print(f"Max density in r = {rloc} pc: ", np.max(Density[sphere]), flush=True)
    print(f"Min density in r = {rloc} pc: ", np.min(Density[sphere]), flush=True)

    for window in range(5):
        if window == 0:
            n_top_boundary = max_den_snap #np.max(Density[sphere])
            n_bottom_boundary = 10**(np.log10(max_den_snap) - width_step * (window+1))#10**(np.log10(n_top_boundary )- 2) # assuming max of Density is not < 2
        else:
            n_top_boundary = 10**(np.log10(max_den_snap) - width_step * (window))
            n_bottom_boundary = 10**(np.log10(max_den_snap) - width_step * (window+1)) #10**(np.log10(n_top_boundary )- 2) # assuming max of Density is not < 2
        
        print(np.log10(n_top_boundary), np.log10(n_bottom_boundary))
        n_above_boundary = np.logical_and(Density > n_bottom_boundary, Density < n_top_boundary) 
        mask = np.logical_and(n_above_boundary, sphere)

        cell_centers = Pos[mask,:]
        cell_densities = Density[mask]

        #sample1 = np.random.choice(cell_centers, size=1000, replace=False)
        try:
            idx = np.random.choice(len(cell_centers), size=no_per_seg + 1, replace=False)
            sample = cell_centers[idx]
            sample_dens = cell_densities[idx]
        except:
            # ValueError: Cannot take a larger sample than population when 'replace=False'
            # then take call
            sample = cell_centers
            sample_dens = cell_densities


        if window == 0:
            new_sample = np.concatenate([sample], axis=0)
            new_sample_dens = np.concatenate([sample_dens], axis=0)
            continue

        new_sample = np.concatenate([new_sample, sample], axis=0)
        new_sample_dens = np.concatenate([new_sample_dens, sample_dens], axis=0)


    return new_sample

@timing
def weighted_in_3d_tree_dependent(tree, Density, no, rloc=1.0, n_crit=1.0e+2):

    def xyz_gen(size):
        U1 = np.random.uniform(low=0.0, high=1.0, size=size)
        U1 = np.random.uniform(low=0.0, high=1.0, size=size)
        U2 = np.random.uniform(low=0.0, high=1.0, size=size)
        U3 = np.random.uniform(low=0.0, high=1.0, size=size)
        r = rloc*U1 #(U1**(1.5))# 
        #a = np.sqrt(U1* (2 - rloc ) / 2.0)
        #r = 1.0 - np.sqrt(1.0 - 2.0*a)
        #r = rloc*np.cbrt(U1)
        #r = rloc*np.exp(-U1)
        #r = rloc*np.log(1 + U1)
        #r = np.exp(U1 - 1) * (rloc + 1) - 1
        theta = np.arccos(2*U2-1)
        phi = 2*np.pi*U3
        x,y,z = r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)

        rho_cartesian = np.array([[a,b,c] for a,b,c in zip(x,y,z)])
        #rho_spherical = np.array([[a,b,c] for a,b,c in zip(r, theta, phi)])
        return rho_cartesian #, rho_spherical

    valid_vectors = []
    _rloc_ = deepcopy(rloc)
    flag = True
    while (len(valid_vectors) < no) and (flag):
        aux_vector = xyz_gen(no - len(valid_vectors)) # [[x,y,z], [x,y,z], ...] <= np array
        distances = np.linalg.norm(aux_vector, axis=1)
        inside_sphere = aux_vector[distances <= _rloc_]
        _, nearest_indices = tree.query(inside_sphere)
        valid_mask = Density[nearest_indices] > n_crit
        valid_points = inside_sphere[valid_mask]
        valid_vectors.extend(valid_points)
        #density_above_threshold = Density[nearest_indices] > n_crit
        #print(density_above_threshold.shape)

        if len(valid_points) == 0:
            #_rloc_ /=2
            flag = False
            warnings.warn(f"[snap] low probability of populating ")
            warnings.warn("Current valid vectors: 0", RuntimeWarning)
            warnings.warn(f"[snap] At current snapshots, no cloud above {n_crit} cm-3", Warning)
            return None
                
    return np.array(deepcopy(valid_vectors))

@timing
def uniform_in_3d_tree_dependent(tree, no, rloc=1.0, n_crit=1.0e+2):

    def xyz_gen(size):
        U1 = np.random.uniform(low=0.0, high=1.0, size=size)
        U2 = np.random.uniform(low=0.0, high=1.0, size=size)
        U3 = np.random.uniform(low=0.0, high=1.0, size=size)
        r = rloc*np.cbrt(U1)
        theta = np.arccos(2*U2-1)
        phi = 2*np.pi*U3
        x,y,z = r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)

        rho_cartesian = np.array([[a,b,c] for a,b,c in zip(x,y,z)])
        #rho_spherical = np.array([[a,b,c] for a,b,c in zip(r, theta, phi)])
        return rho_cartesian #, rho_spherical

    valid_vectors = []
    _rloc_ = deepcopy(rloc)
    flag = True
    while (len(valid_vectors) < no) and (flag):
        aux_vector = xyz_gen(no - len(valid_vectors)) # [[x,y,z], [x,y,z], ...] <= np array
        distances = np.linalg.norm(aux_vector, axis=1)
        inside_sphere = aux_vector[distances <= _rloc_]
        _, nearest_indices = tree.query(inside_sphere)
        valid_mask = Density[nearest_indices] > n_crit
        valid_points = inside_sphere[valid_mask]
        valid_vectors.extend(valid_points)
        #density_above_threshold = Density[nearest_indices] > n_crit
        #print(density_above_threshold.shape)

        if len(valid_points) == 0:
            #_rloc_ /=2
            flag = False
            warnings.warn("Current valid vectors: 0", RuntimeWarning)
            warnings.warn(f"[snap={snap}] At current snapshots, no points found in cloud above {n_crit} cm-3", Warning)
            return None
                
    return np.array(deepcopy(valid_vectors))

@timing
def crs_path(*args, **kwargs):

    x_init = kwargs.get('x_init', None)
    n_crit = kwargs.get('n_crit', 1.0e+2)

    
    """ deprecated
    Default density threshold is 100 cm^-3  but saves index for both 10 and 100 boundary. 
    This way, all data is part of a comparison between 10 and 100 
    """
    m = x_init.shape[0]

    line      = np.zeros((__alloc_slots__+1,m,3)) # from __alloc_slots__+1 elements to the double, since it propagates forward and backward
    bfields   = np.zeros((__alloc_slots__+1,m))
    densities = np.zeros((__alloc_slots__+1,m))
    pst_mask = np.zeros((m,)).astype(int) # one value for each
    
    line_rev=np.zeros((__alloc_slots__+1,m,3)) # from __alloc_slots__+1 elements to the double, since it propagates forward and backward
    bfields_rev = np.zeros((__alloc_slots__+1,m))
    densities_rev = np.zeros((__alloc_slots__+1,m))
    pst_mask_rev = np.zeros((m,)).astype(int) # one value for each

    cells_for = np.zeros((__alloc_slots__+1,m))
    cells_rev = np.zeros((__alloc_slots__+1,m))

    line[0,:,:]     = x_init
    line_rev[0,:,:] = x_init 
    
    x = x_init.copy()

    dummy, bfields[0,:], densities[0,:], cells = find_points_and_get_fields(x, Bfield, Density, Pos, VoronoiPos)

    cells_for[0, :] = cells

    vol = Volume[cells]
    #densities[0,:] = densities[0,:] * gr_cm3_to_nuclei_cm3
    dens = densities[0,:] # * gr_cm3_to_nuclei_cm3
    k=0

    mask2 = dens > n_crit
    un_masked2 = np.logical_not(mask2) # 1 if finished

    x_rev = x_init.copy()

    dummy_rev, bfields_rev[0,:], densities_rev[0,:], cells = find_points_and_get_fields(x_rev, Bfield, Density, Pos, VoronoiPos)

    cells_rev[0, :] = cells

    vol_rev = Volume[cells]

    #densities_rev[0,:] = densities_rev[0,:] * gr_cm3_to_nuclei_cm3
    dens_rev = densities_rev[0,:]  #* gr_cm3_to_nuclei_cm3
    
    k=0
    k_rev=0

    mask2_rev = dens > n_crit
    un_masked2_rev = np.logical_not(mask2_rev)

    print(cells_rev.shape, cells_for.shape)

    #while np.any(mask2) or np.any(mask2_rev): 
    while np.any(mask2) and (k + 1 < __alloc_slots__) or np.any(mask2_rev) and (k_rev + 1 < __alloc_slots__):

        mask2_rev = dens_rev > n_crit
        un_masked2_rev = np.logical_not(mask2_rev)

        if np.any(mask2_rev) and (k_rev + 1 < __alloc_slots__):
        
            x_rev_aux = x_rev[mask2_rev]

            x_rev_aux, bfield_aux_rev, dens_aux_rev, vol_rev, cells = Heun_step(x_rev_aux, -1.0, Bfield, Density, Pos, VoronoiPos, Volume, retrieve_cells = True)
            #dens_aux_rev = dens_aux_rev * gr_cm3_to_nuclei_cm3
            
            x_rev[mask2_rev] = x_rev_aux
            dens_rev[mask2_rev] = dens_aux_rev
            pst_mask_rev[mask2_rev] = bool(1)


            x_rev[un_masked2_rev] = 0
            dens_rev[un_masked2_rev] = 0
            pst_mask_rev[un_masked2_rev] = bool(0)

            cells_rev[k_rev+1,mask2_rev] = cells
            cells_rev[k_rev+1,un_masked2_rev] = -1
            
            #print(" alive lines? ",  np.any(mask2_rev), "k_rev + 1 < __alloc_slots__: ", k_rev + 1 < __alloc_slots__)

            line_rev[k_rev+1,mask2_rev,:] = x_rev_aux
            bfields_rev[k_rev+1,mask2_rev] = bfield_aux_rev
            densities_rev[k_rev+1,mask2_rev] = dens_aux_rev              

            k_rev += 1

        mask2 = dens > n_crit # above threshold
        un_masked2 = np.logical_not(mask2)
        
        if np.any(mask2) and (k + 1 < __alloc_slots__):

            x_aux = x[mask2]
            x_aux, bfield_aux, dens_aux, vol, cells = Heun_step(x_aux, 1.0, Bfield, Density, Pos, VoronoiPos, Volume, retrieve_cells = True)
            #dens_aux = dens_aux * gr_cm3_to_nuclei_cm3
            
            x[mask2]                   = x_aux
            dens[mask2]                = dens_aux
            pst_mask[mask2]            = bool(1)

            cells_for[k+1,mask2] = cells
            cells_for[k+1,un_masked2]      = -1

            x[un_masked2]              = 0
            dens[un_masked2]           = 0
            pst_mask[un_masked2]       = bool(0)
            #print("k + 1 < __alloc_slots__: ", k + 1 < __alloc_slots__," alive lines? ",  np.any(mask2))
            line[k + 1, mask2, :]      = x_aux
            bfields[k + 1, mask2]      = bfield_aux
            densities[k + 1, mask2]    = dens_aux

            k += 1
        
    print(np.logical_not((np.any(mask2_rev) and (k_rev + 1 < __alloc_slots__))), np.logical_not((np.any(mask2) and (k + 1 < __alloc_slots__))))
    #threshold = threshold.astype(int)

    survivors_mask = np.logical_not(np.logical_and(pst_mask, pst_mask_rev))

    percentage_of_survivors = np.sum(survivors_mask)*100/survivors_mask.shape[0]

    print("Percentage of Survivors at crs_path: ", percentage_of_survivors, " %", flush=True)
    print("k = ", k, " k_rev = ", k_rev, flush=True)
    
    nz_i    = k + 1
    nz_irev = k_rev + 1
    
    print(f"get_lines => threshold index for {__threshold__}cm-3: ", nz_i, nz_irev, flush=True)
    print(f"get_lines => original shapes ({2*__alloc_slots__+1} to {nz_i + nz_irev - 1})", flush=True)
    print(f"get_lines => p_r = {__alloc_slots__+1} to p_r = {nz_irev} for array with shapes ...", flush=True)

    radius_vectors   = np.append(line_rev[:nz_irev,:,:][::-1, :, :], line[1:nz_i,:,:], axis=0)
    magnetic_fields  = np.append(bfields_rev[:nz_irev,:][::-1, :], bfields[1:nz_i,:], axis=0)
    numb_densities   = np.append(densities_rev[:nz_irev,:][::-1, :], densities[1:nz_i,:], axis=0)
    cells_arr        = np.append(cells_rev[:nz_irev,:][::-1, :], cells_for[1:nz_i,:], axis=0)

    for j in range(cells_arr.shape[1]): # must be shape (2*N + 1, m)

        cell_indices = cells_arr[:, j]
        remove_negatives = np.where(cell_indices > -1)[0]
        first, last = remove_negatives[0], remove_negatives[-1]
        
        val = detect_oscillating_line(cell_indices[first: last + 1], threshold=10, max_gap=50)
        # if val is True - then didnt survived
        survivors_mask[j] = not val
    
    if False:
        # find field lines with recurring cycles along the trajectory
        for file in glob.glob("./lines/*.png"):
            os.remove(file)
            
        print("Cells array shape", cells_arr.shape, flush=True)
        for j in range(cells_arr.shape[1]): # must be shape (2*N + 1, m)

            cell_indices = cells_arr[:, j]
            remove_negatives = np.where(cell_indices > -1)[0]
            first, last = remove_negatives[0], remove_negatives[-1]

            suv = survivors_mask[j]

            val = detect_oscillating_line(cell_indices[first: last + 1], threshold=10, max_gap=50)
            print(f"field line {j} is oscilating/survivor?", val, suv, flush=True)

            if val:
                fig, ax = plt.subplots()
                ax.plot(magnetic_fields[first: last + 1, j])
                plt.title(f"oscilating/survivor? {val}, {suv}")
                plt.savefig(f"./lines/maglinesosc{j}.png")
                plt.close(fig)
            else:
                fig, ax = plt.subplots()
                ax.plot(magnetic_fields[first: last + 1, j])
                plt.title(f"oscilating/survivor? {val}, {suv}")
                plt.savefig(f"./lines/maglinessurv{j}.png")
                plt.close(fig)

    print("Radius vector shape:", radius_vectors.shape, flush=True)

    m = magnetic_fields.shape[1]
    
    path_column   = np.sum(numb_densities[1:, :] * np.linalg.norm(np.diff(radius_vectors, axis=0), axis=2), axis=0) * pc_to_cm

    return radius_vectors, magnetic_fields, numb_densities, nz_irev, path_column, survivors_mask#p_r #, [threshold, threshold2, threshold_rev, threshold2_rev]

@timing
def line_of_sight(*args, **kwargs):
    x_init = kwargs.get('x_init', None)
    directions = kwargs.get('directions', fibonacci_sphere(20))
    n_crit = kwargs.get('n_crit', 1.0e+2)

    """
    Default density threshold is 10 cm^-3  but saves index for both 10 and 100 boundary. 
    This way, all data is part of a comparison between 10 and 100 
    """
    directions = directions/np.linalg.norm(directions, axis=1)[:, np.newaxis]
    dx = 0.5
    """
    Here you need to 
    directions = its repeated version 'm' times
    directions = np.tile(directions, m)
    x_init     = figure out how to repeat according to the example
    """
    m0 = x_init.shape[0]
    l0 = directions.shape[0]
    directions = np.tile(directions, (m0, 1))
    x_init = np.repeat(x_init, l0, axis=0)
    m = x_init.shape[0]
    l = directions.shape[0]
    """
    Now, a new feature that might speed the while loop, can be to double the size of all arrays
    and start calculating backwards and forwards simultaneously. This creates a more difficult condition
    for the 'mask', nevertheless, for a large array 'x_init' it may not be as different and it will definitely scale efficiently in parallel
    """
    line      = np.zeros((__alloc_slots__+1,m,3)) # from __alloc_slots__+1 elements to the double, since it propagates forward and backward
    line_rev=np.zeros((__alloc_slots__+1,m,3)) # from __alloc_slots__+1 elements to the double, since it propagates forward and backward
    densities = np.zeros((__alloc_slots__+1,m))
    densities_rev = np.zeros((__alloc_slots__+1,m))
    threshold = np.zeros((m,))
    threshold_rev = np.zeros((m,))

    line[0,:,:]     = x_init.copy()
    line_rev[0,:,:] = x_init.copy()
    x = x_init.copy()
    dummy, _0, densities[0,:], cells = find_points_and_get_fields(x_init, Bfield, Density, Pos, VoronoiPos)

    vol = Volume[cells]
    #densities[0,:] = densities[0,:] * gr_cm3_to_nuclei_cm3
    dens = densities[0,:].copy()
    k=0
    x_rev = x_init.copy()
    densities_rev[0,:], cells_rev = densities[0,:].copy(), cells.copy()
    vol_rev = Volume[cells_rev]
    #densities_rev[0,:] = densities_rev[0,:]
    dens_rev = densities_rev[0,:].copy()
    k_rev=0
    mask  = dens > n_crit# 1 if not finished
    un_masked = np.logical_not(mask) # 1 if finished
    mask_rev = dens_rev > n_crit
    un_masked_rev = np.logical_not(mask_rev)

    while np.any(mask) and (k + 1 < __alloc_slots__) or np.any(mask_rev) and (k_rev + 1 < __alloc_slots__):
        mask = dens > n_crit              # still alive?
        un_masked = np.logical_not(mask)  # any deaths?
        mask_rev = dens_rev > n_crit     
        un_masked_rev = np.logical_not(mask_rev)

        if np.any(mask) and (k + 1 < __alloc_slots__): # any_alive? and below_allocation?
            # continue if still lines alive and below allocation

            _, bfield, dens, vol = Heun_step(x, 1.0, Bfield, Density, Pos, VoronoiPos, Volume)

            #dens *= gr_cm3_to_nuclei_cm3
            
            vol[un_masked] = 0               # artifically make cell volume of finished lines equal to cero

            dx_vec = ((4 / 3) * vol / np.pi) ** (1 / 3)

            threshold += mask.astype(int)  # Increment threshold count only for values still above 100

            x += dx_vec[:, np.newaxis] * directions

            line[k+1,:,:]    = x
            densities[k+1,:] = dens

            k += 1

        if np.any(mask_rev) and (k_rev + 1 < __alloc_slots__):
            # continue if still lines alive and below allocation

            _, bfield_rev, dens_rev, vol_rev = Heun_step(x_rev, 1.0, Bfield, Density, Pos, VoronoiPos, Volume)
            
            #dens_rev *= gr_cm3_to_nuclei_cm3
            
            vol_rev[un_masked_rev] = 0

            dx_vec = ((4 / 3) * vol_rev / np.pi) ** (1 / 3) 

            threshold_rev += mask_rev.astype(int)

            x_rev -= dx_vec[:, np.newaxis] * directions

            line_rev[k_rev+1,:,:]    = x_rev
            densities_rev[k_rev+1,:] = dens_rev

            k_rev += 1
        
    threshold = threshold.astype(int)
    threshold_rev = threshold_rev.astype(int)

    max_th     = np.max(threshold) + 1
    max_th_rev = np.max(threshold_rev) + 1 

    radius_vectors = np.append(line_rev[:max_th_rev,:,:][::-1, :, :], line[1:max_th,:,:], axis=0)*pc_to_cm
    numb_densities = np.append(densities_rev[:max_th_rev,:][::-1, :], densities[1:max_th,:], axis=0)
    column_densities = np.sum(numb_densities[1:, :] * np.linalg.norm(np.diff(radius_vectors, axis=0), axis=2), axis=0) # list of 'm' column densities
    
    mean_columns = np.zeros(m0)
    median_columns= np.zeros(m0)
    i = 0
    
    #print("null means good", np.sum(np.where(numb_densities == 0)), "\n") # if this prints null then we alright

    for x in range(0, l0*m0, l0): 

        mean_columns[i] = np.mean(column_densities[x:x+l0])
        median_columns[i] = np.median(column_densities[x:x+l0])
        i += 1

    if False: # for a single point in space with l0 directions
        fig0, ax0 = plt.subplots()
        h = 0
        ax0.set_xlabel(r'''$s$ [cm]''', fontsize=16)
        ax0.set_ylabel(r'$n_g$ [cm$^{-3}$]', fontsize=16)
        #ax0.set_yscale('log')
        ax0.set_xscale('log')
        
        for j in range(l0):

            s  = np.cumsum(np.linalg.norm(np.diff(radius_vectors[:, j, :], axis=0), axis=1))
            ng = numb_densities[1:, j] #* np.linalg.norm(np.diff(radius_vectors[:, j, :], axis=0), axis=1)

            ax0.plot(s, ng, '-',label=f'{j}')

        ax0.tick_params(axis='both')
        ax0.grid(True, which='both', alpha=0.3)
        ax0.legend(loc="upper left", fontsize=6)
        
        plt.title(r'Density [$n_g$] vs Distance [$s$]', fontsize=16)

        plt.savefig(f'./series/ng_vs_s.png',dpi=300)
        plt.close(fig0)

        fig1, ax1 = plt.subplots()

        # x0, ... 20 veces, x1, ..., 20 veces

        ax1.set_xlabel(r'''$s$ [cm]''', fontsize=16)
        ax1.set_ylabel(r'$N_{los}$ [cm$^{-2}$]', fontsize=16)
        ax1.set_yscale('log')
        ax1.set_xscale('log')

        for j in range(l0):

            c  = np.cumsum(numb_densities[1:, j]*np.linalg.norm(np.diff(radius_vectors[:, j, :], axis=0), axis=1))
            ax1.plot(s, c, '-', label=f'{j}',linewidth=1)

        ax1.tick_params(axis='both')
        ax1.grid(True, which='both', alpha=0.3)
        ax1.legend(loc="upper left", fontsize=6)
        
        plt.title(r'Column [$N_{los}$] vs Distance [$s$]', fontsize=16)

        plt.savefig(f'./series/c_vs_s.png',dpi=300)
        plt.close(fig1)

    return radius_vectors, numb_densities, mean_columns, median_columns

@timing
def edge_to_p_line_of_sight(*args, **kwargs):
    x_init = kwargs.get('x_init', None)
    directions = kwargs.get('directions', fibonacci_sphere(20))
    n_crit = kwargs.get('n_crit', 1.0e+2)

    """
    Default density threshold is 10 cm^-3  but saves index for both 10 and 100 boundary. 
    This way, all data is part of a comparison between 10 and 100 
    """
    directions = directions/np.linalg.norm(directions, axis=1)[:, np.newaxis]
    dx = 0.5
    """
    Here you need to 
    directions = its repeated version 'm' times
    directions = np.tile(directions, m)
    x_init     = figure out how to repeat according to the example
    """
    m0 = x_init.shape[0]
    l0 = directions.shape[0]
    directions = np.tile(directions, (m0, 1))
    x_init = np.repeat(x_init, l0, axis=0)
    m = x_init.shape[0]
    l = directions.shape[0]
    """
    Now, a new feature that might speed the while loop, can be to double the size of all arrays
    and start calculating backwards and forwards simultaneously. This creates a more difficult condition
    for the 'mask', nevertheless, for a large array 'x_init' it may not be as different and it will definitely scale efficiently in parallel
    """
    line      = np.zeros((__alloc_slots__+1,m,3)) # from __alloc_slots__+1 elements to the double, since it propagates forward and backward
    densities = np.zeros((__alloc_slots__+1,m))
    threshold = np.zeros((m,))

    line[0,:,:]     = x_init.copy()
    x = x_init.copy()
    dummy, _0, densities[0,:], cells = find_points_and_get_fields(x_init, Bfield, Density, Pos, VoronoiPos)

    vol = Volume[cells]
    #densities[0,:] = densities[0,:] * gr_cm3_to_nuclei_cm3
    dens = densities[0,:].copy()
    k=0

    mask  = dens > n_crit# 1 if not finished
    un_masked = np.logical_not(mask) # 1 if finished

    while np.any(mask) and (k + 1 < __alloc_slots__):
        mask = dens > n_crit              # still alive?
        un_masked = np.logical_not(mask)  # any deaths?


        # continue if still lines alive and below allocation

        _, bfield, dens, vol = Heun_step(x, 1.0, Bfield, Density, Pos, VoronoiPos, Volume)

        #dens *= gr_cm3_to_nuclei_cm3
        
        vol[un_masked] = 0               # artifically make cell volume of finished lines equal to cero

        dx_vec = ((4 / 3) * vol / np.pi) ** (1 / 3)

        threshold += mask.astype(int)  # Increment threshold count only for values still above 100

        x += dx_vec[:, np.newaxis] * directions

        line[k+1,:,:]    = x
        densities[k+1,:] = dens

        k += 1

        
    threshold = threshold.astype(int)
    max_th     = np.max(threshold) + 1

    radius_vectors = line[1:max_th,:,:]*pc_to_cm
    numb_densities = densities[1:max_th,:]
    column_densities = np.sum(numb_densities[1:, :] * np.linalg.norm(np.diff(radius_vectors, axis=0), axis=2), axis=0)
    
    mean_columns = np.zeros(m0)
    median_columns= np.zeros(m0)
    i = 0
    
    #print("null means good", np.sum(np.where(numb_densities == 0)), "\n") # if this prints null then we alright

    for x in range(0, l0*m0, l0): 

        mean_columns[i] = np.mean(column_densities[x:x+l0])
        median_columns[i] = np.median(column_densities[x:x+l0])
        i += 1

    return radius_vectors, numb_densities, mean_columns, median_columns

@timing
def match_files_to_data(__input_case__, __start_snap__):
    
    if __input_case__ in 'ideal_mhd':
        subdirectory = 'ideal_mhd'
        file_hdf5 = np.array(glob.glob(f'arepo_data/{subdirectory}/*.hdf5'))
        file_xyz       = f'./util/{__input_case__}_cloud_trajectory.txt'
    elif __input_case__ in 'ambipolar_diffusion':
        subdirectory = 'ambipolar_diffusion'
        file_hdf5 = np.array(glob.glob(f'arepo_data/{subdirectory}/*.hdf5'))
        file_xyz       = f'./util/{__input_case__}_cloud_trajectory.txt'

    assert os.path.exists(file_xyz), f"[{file_xyz} cloud data not present]"
 
    with open(file_xyz, mode='r') as file:
        clst = []; dlst = []; tlst = []; slst = []
        csv_reader = csv.reader(file)  # Use csv.reader to access rows directly
        next(csv_reader)  # Skip the header row
        
        for row in csv_reader:
            #print(row[0],row[1], np.log10(float(row[8])))
            clst.append([np.float64(row[2]),np.float64(row[3]),np.float64(row[4])])
            dlst.append(np.float64(row[8]))
            tlst.append(np.float64(row[1]))
            slst.append(int(row[0]))
            if (int(row[0]) < int(__start_snap__)):
                break

        
    assert len(slst) != 0, "Error accessing cloud data"

    clst, dlst, tlst, slst = [np.array(arr, dtype=FloatType) for arr in (clst[::-1], dlst[::-1], tlst[::-1], slst[::-1])]

    validated = np.logical_not(dlst); white_list = []

    for i, f in enumerate(file_hdf5):
        if int(f.split('.')[0][-3:]) in slst: 
            #print(int(f.split('.')[0][-3:]), np.where(slst == int(f.split('.')[0][-3:])))    
            validated[np.where(slst == int(f.split('.')[0][-3:]))] = True
            white_list += [True]
        else:
            white_list += [False]

    clst, dlst, tlst, slst = [arr[validated] for arr in (clst, dlst, tlst, slst)]

    file_hdf5 = np.sort(file_hdf5[np.array(white_list)][::-1])

    return clst, dlst, tlst, slst, file_hdf5

def describes(data, band=False, percent=False):

    # data shape: (50, 1000)
    mean_   = np.mean(data, axis=0)
    median_ = np.median(data, axis=0)
    skew_   = skew(data, axis=0)
    kurt_   = kurtosis(data, axis=0)

    if band:
        sigma1_low, sigma1_high = np.percentile(data, [16, 84], axis=0) # 1-sigma band
        return sigma1_low, sigma1_high
    if percent:
        p_25    = np.percentile(data, 25, axis=0)
        p_10    = np.percentile(data, 10, axis=0)
        p_5     = np.percentile(data, 5, axis=0)
        return p_25, p_10, p_5
    return mean_, median_, skew_, kurt_

def pocket_finder(bfield, numb, p_r, plot=False):
    #pocket_finder(bfield, p_r, B_r, img=i, plot=False)
    """  
    Finds peaks in a given magnetic field array.

    Args:
        bfield (array-like): Array or list of magnetic field magnitudes.
        cycle (int, optional): Cycle number for saving the plot. Defaults to 0.
        plot (bool, optional): Whether to plot the results. Defaults to False.

    Returns:
        tuple: Contains two tuples:
            - (indexes, peaks): Lists of peak indexes and corresponding peak values.
            - (index_global_max, upline): Indices and value of the global maximum.
    """
    bfield = np.array(bfield)  # Ensure input is a numpy array

    baseline = np.min(bfield)
    upline = np.max(bfield)
    index_global_max = np.where(bfield == upline)[0]
    try:
        idx = index_global_max[0]
    except Exception as e:
        warnings.warn("[pocket_finder]", e)
        idx = index_global_max
    upline == bfield[idx]
    ijk = np.argmax(bfield)
    bfield[ijk] = bfield[ijk]*1.001 # if global_max is found in flat region, choose one and scale it 0.001


    # Find left peaks
    Bi = 0.0
    lindex = []
    lpeaks = []
    for i, Bj in enumerate(bfield):
        if Bj < Bi and (len(lpeaks) == 0 or Bi > lpeaks[-1]):  # if True, then we have a peak
            lindex.append(i - 1)
            lpeaks.append(Bi)
        Bi = Bj

    # Find right peaks
    Bi = 0.0
    rindex = []
    rpeaks = []
    for i, Bj in enumerate(reversed(bfield)):
        if Bj < Bi and (len(rpeaks) == 0 or Bi > rpeaks[-1]):  # if True, then we have a peak
            rindex.append(len(bfield) - i)
            rpeaks.append(Bi)
        Bi = Bj

    peaks = lpeaks +  list(reversed(rpeaks))[1:]
    indexes = lindex + list(reversed(rindex))[1:]

    if plot:
        # Find threshold crossing points for 100 cm^-3
        mask = np.log10(numb) < 2  # log10(100) = 2
        slicebelow = mask[:p_r]
        sliceabove = mask[p_r:]
        peaks = np.array(peaks)
        indexes = np.array(indexes)

        try:
            above100 = np.where(sliceabove)[0][0] + p_r
        except IndexError:
            above100 = None

        try:
            below100 = np.where(slicebelow)[0][-1]
        except IndexError:
            below100 = None


    return (indexes, peaks), (index_global_max, upline)

def eval_reduction(field, numb, follow_index, threshold):

    R10      = []
    Numb100  = []
    B100     = []

    _, m = field.shape

    flag = False
    filter_mask = np.ones(m).astype(bool)
    dead = 0
    for i in range(m):

        mask10 = np.where(numb[:, i] > threshold)[0]
        if mask10.size > 0:
            start, end = mask10[0], mask10[-1]
            if start <= follow_index <= end:
                try:
                    numb10   = numb[start:end+1, i]
                    bfield10 = field[start:end+1, i]
                    p_r = follow_index - start
                    B_r = bfield10[p_r]
                    n_r = numb10[p_r]
                except IndexError:
                    raise ValueError(f"\nTrying to slice beyond bounds for column {i}. "
                                    f"start={start}, end={end}, shape={numb.shape}")
            else:
                print(f"\n[Info] follow_index {follow_index} outside threshold interval for column {i}.")
                if follow_index >= numb.shape[0]:
                    raise ValueError(f"follow_index {follow_index} is out of bounds for shape {numb.shape}")
                numb10   = np.array([numb[follow_index, i]])
                bfield10 = np.array([field[follow_index, i]])
                p_r = 0
                B_r = bfield10[p_r]
                n_r = numb10[p_r]
        else:
            print(f"\n[Info] No densities > {threshold} cm-3 found for column {i}. Using follow_index fallback.")
            if follow_index >= numb.shape[0]:
                raise ValueError(f"\nfollow_index {follow_index} is out of bounds for shape {numb.shape}")
            numb10   = np.array([numb[follow_index, i]])
            bfield10 = np.array([field[follow_index, i]])
            p_r = 0
            B_r = bfield10[p_r]
            n_r = numb10[p_r]

        #print("p_r: ", p_r)
        if not (0 <= p_r < bfield10.shape[0]):
            raise IndexError(f"\np_r={p_r} is out of bounds for bfield10 of length {len(bfield10)}")

        # pockets with density threshold of 10cm-3
        pocket, global_info = pocket_finder(bfield10, numb10, p_r, plot=False)
        index_pocket, field_pocket = pocket[0], pocket[1]
        flag = False
        p_i = np.searchsorted(index_pocket, p_r)

        most_common_value, count = Counter(bfield10.ravel()) .most_common(1)[0]
    
        if count > 20:
            R = 1.
            #print(f"Most common value: {most_common_value} (appears {count} times): R = ", R)
            R10.append(R)
            Numb100.append(n_r)
            B100.append(B_r)   
            flag = True
            filter_mask[i] = False
            dead +=1
            continue     

        try:
            closest_values = index_pocket[max(0, p_i - 1): min(len(index_pocket), p_i + 1)]
            B_l = min([bfield10[closest_values[0]], bfield10[closest_values[1]]])
            B_h = max([bfield10[closest_values[0]], bfield10[closest_values[1]]])
            # YES! 
            success = True  
        except:
            # NO :c
            R = 1.
            R10.append(R)
            Numb100.append(n_r)
            B100.append(B_r)
            success = False 
            continue

        if success:
            # Ok, our point is between local maxima, is inside a pocket?
            if B_r / B_l < 1:
                # double YES!
                R = 1. - np.sqrt(1 - B_r / B_l)
                R10.append(R)
                Numb100.append(n_r)
                B100.append(B_r)
            else:
                # NO!
                R = 1.
                R10.append(R)
                Numb100.append(n_r)
                B100.append(B_r)

    filter_mask = filter_mask.astype(bool)

    return np.array(R10), np.array(Numb100), np.array(B100), filter_mask
