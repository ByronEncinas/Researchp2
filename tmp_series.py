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
import src.tmp_library as tmplib
import src.sampling as sam
import warnings
from mpi4py import MPI          # Move to TOP of __main__
import pickle
import asyncio

start_time = time.time()

input_file = sys.argv[1]

tmplib.config_ic(input_file)

print(f"\n[{sys.argv[0]}]: started running...\n", flush=True)
print(f"[__input_case__] ", tmplib.__input_case__,flush=True)
# amb:   t > 3.0 Myrs snap > 225    
# ideal: t > 3.0 Myrs snap > 270
__start_time__  = 3.0 # Myrs

if tmplib.__input_case__ =='ideal':
    __start_snap__  = '270'

if tmplib.__input_case__ =='amb':
    __start_snap__  = '225'

if tmplib.FLAG2 in sys.argv:
    __start_snap__  = '000'
    

os.makedirs("./series/", exist_ok=True)

if __name__=='__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        clst, dlst, tlst, slst, file_hdf5 = tmplib.match_files_to_data(tmplib.__input_case__,__start_snap__)
        input_file_ = input_file.split('/')[-1]
        _id_ = str(input_file_.split('.')[0][0] + input_file_.split('.')[0][-1])

        if tmplib.FLAG1 in sys.argv:
            _id_ = str(input_file.split('.')[0][0] + input_file.split('.')[0][-3:])
        if tmplib.FLAG3 in sys.argv:
            _id_ = "w" + str(input_file_.split('.')[0][0] + input_file_.split('.')[0][-1])
        if "-dense" in sys.argv:
            _id_ = _id_ + "_dense"
    
        print("ID of series.py run is", _id_)
    else:
        clst = dlst = tlst = slst = file_hdf5 = None
        _id_ = None

    clst, dlst, tlst, slst, file_hdf5, _id_ = comm.bcast(
        (clst, dlst, tlst, slst, file_hdf5, _id_), root=0
    )
    
    assert len(file_hdf5) == len(clst) == len(tlst), "Arrays must all have the same length"

    survivors_fraction = np.zeros(file_hdf5.shape[0])
    
    df_stats = dict()
    df_fields= dict()
    print(file_hdf5)
    for each in range(rank, len(file_hdf5), size):
        filename = file_hdf5[each]
        center   = clst[each, :]
        _time    = tlst[each]

        tmplib.config_arepo(filename, center)

        table_data = [
            ["__curr_snap__", filename.split('/')[-1]],
            ["__cores_avail__", os.cpu_count()],
            ["__rloc__", f"{tmplib.__rloc__}" + " pc"],
            ["__threshold__", f"{tmplib.__threshold__}"+ " cm^-3"],
            ["__dense_cloud__", f"{tmplib.__dense_cloud__}"+ " cm^-3"],
            ["__alloc_slots__", tmplib.__alloc_slots__],
            ["__sample_size__", tmplib.__sample_size__]
            ]

        print(tabulate(table_data, headers=["Property", "Value"], tablefmt="grid"), '\n', flush=True)
        
        tree = cKDTree(tmplib.Pos)

        try:
            if "-dense" in sys.argv:
                x_input = tmplib.dense_segments_in_3d_tree_dependent(tree, tmplib.Density, tmplib.Pos, tmplib.__sample_size__, rloc=tmplib.__rloc__)
            elif tmplib.FLAG3 in sys.argv:
                print(f"Flag {tmplib.FLAG3} was used, therefore Random Variable $X_r \sim U_1$",flush = True)
                x_input    = tmplib.weighted_in_3d_tree_dependent(tree, tmplib.Density, tmplib.__sample_size__, rloc=0.5, n_crit=tmplib.__dense_cloud__)   
            else:
                x_input    = tmplib.uniform_in_3d_tree_dependent(tree, tmplib.__sample_size__, rloc=tmplib.__rloc__, n_crit=tmplib.__dense_cloud__)   
        except Exception as e:
            warnings.warn(f"[snap={tmplib.snap}]", RuntimeWarning)
            warnings.warn(f"[snap={tmplib.snap}] At current snapshots, no cloud above {tmplib.__dense_cloud__} cm-3", Warning)
            print(f"[Snap] snap {filename}: skipping", flush=True)
            tmplib.config_arepo(filename, center, True)
            continue

        if x_input is None or x_input.shape == (0,):
            print(f"[Snap] snap {tmplib.snap}: skipping", flush=True)
            tmplib.config_arepo(filename, center, True)
            continue

        print(x_input.shape)

        dist, cells, rel_pos = tmplib.find_points_and_relative_positions(x_input, tmplib.Pos, tmplib.VoronoiPos)
        sample_dens = tmplib.Density[cells]

        distances = np.linalg.norm(x_input, axis = 1)
        
        fig, ax = plt.subplots()

        ax.scatter(distances, sample_dens, s = 1)
        ax.set_yscale("log")
        plt.show()
        plt.close(fig)
        continue
        """

        mask = tmplib.Pos[:,0]*tmplib.Pos[:,0] + tmplib.Pos[:,1]*tmplib.Pos[:,1]+tmplib.Pos[:,2]*tmplib.Pos[:,2] < 0.1

        x_input =  tmplib.Pos[mask,:]

        dist, cells, rel_pos = tmplib.find_points_and_relative_positions(x_input, tmplib.Pos, tmplib.VoronoiPos)

        Distances = np.linalg.norm(x_input, axis = 1)

        maskA = tmplib.Density[cells] > 1.0e+10
        #maskA = np.logical_and(tmplib.Density[cells] > 1.0e+8, tmplib.Density[cells] < 1.0e+10)

        print(np.sum(maskA))

        #rl = 0.1

        fig, ax = plt.subplots()

        ax.scatter(Distances[maskA], tmplib.Density[cells][maskA], s=1, color = "red")
        #ax.hlines(100, -0.0, 0.1, linestyle='--', color="black")


        #ax.set_ylim(10**1.8, 10**14)
        #ax.set_xlim(-0.01, rl*1.01)
        ax.set_xlabel(r"$r$ [pc]")
        ax.set_ylabel(r"$\log_{10}(n_g / \rm{cm}^{-3})$")
        ax.set_title(rf"Weighted Sampling $X_r$")
        ax.set_yscale("log")
        #ax.set_xscale("log")
        #plt.savefig("./DensityVRadiusAbove10e2.png", dpi = 150)
        plt.show()
        plt.close(fig)
        
        fig, ax = plt.subplots()
        #ax.hist(np.log10(Density[cells][maskA]), bins = 80, color = "black")
        ax.hist(Distances[maskA], bins = 50, color = "black")
        #ax.plot(np.log10(Density[cells][maskA]), bins = 80, color = "black")
        #ax.hist(Density[cells][maskM], bins = 80, color = "black")
        #ax.vlines(100, -0.01, rl*1.01, linestyle='--', color="black")
        #ax.set_xlim(1, 14)
        #ax.set_xscale("log")
        ax.set_ylabel(r"Count")
        ax.set_xlabel(r"$r$ [pc]")
        ax.set_title(rf"Weighted Sampling $X_r$")
        ax.set_yscale("log")
        #plt.savefig("./HistogramDensityAbove10e2.png", dpi = 150)
        plt.show()
        plt.close(fig)
        break

        # Generated points and uniformity 
        sam.gkde_plt(x_input, _id_+str(tmplib.snap))
        print(np.log10(np.max(tmplib.Density)))
        continue

        """

        
        directions=tmplib.fibonacci_sphere(20) # dodecahedron (12 faces), and icosahedron (20 faces)
        try:
            #__0, __1, mean_column, median_column = tmplib.line_of_sight(x_init=x_input, directions=directions, n_crit=tmplib.__threshold__)
            __0, __1, mean_column, median_column = tmplib.edge_to_p_line_of_sight(x_init=x_input, directions=directions, n_crit=tmplib.__threshold__)
        except Exception as e:
            warnings.warn(f"[snap={tmplib.snap}]", RuntimeWarning)
            print(f"[LOS] Invalid result from intergration: {tmplib.snap}: skipping", flush=True)
            tmplib.config_arepo(filename, center, True)
            print(f"{e}", flush=True)
            continue

        tmplib.__threshold__ = 10

        try:
            radius_vectors, magnetic_fields, numb_densities, follow_index, path_column, survivors1 = tmplib.crs_path(x_init=x_input, n_crit=tmplib.__threshold__)
            assert np.any(numb_densities > tmplib.__threshold__), f"No values above threshold {tmplib.__threshold__} cm-3"
        except Exception as e:
            warnings.warn(f"[snap={tmplib.snap}]", RuntimeWarning)
            print(f"[CRS] Invalid result from intergration: {tmplib.snap}: skipping", flush=True)
            tmplib.config_arepo(filename, center, True)
            continue

        print("__alloc_slots__: ", tmplib.__alloc_slots__, flush=True)
        print("__used_slots__ : ",__0.shape, flush=True)

        r_u, n_rs, B_rs, survivors2 = tmplib.eval_reduction(magnetic_fields, numb_densities, follow_index, 1.0e+2)
        r_l, _1, _2, _3 = tmplib.eval_reduction(magnetic_fields, numb_densities, follow_index, 1.0e+1)

        
        survivors = np.logical_and(survivors1, survivors2)

        print(np.sum(survivors)/survivors.shape[0], " Survivor fraction", flush=True)

        if "-dense" in sys.argv:

            _dist_, cells, _rel_pos_ = tmplib.find_points_and_relative_positions(x_input, tmplib.Pos, tmplib.VoronoiPos)
            
            NormDivB = tmplib.MagneticFieldDivergence[cells] * (np.cbrt(3 * tmplib.Volume[cells] / (4*np.pi)))   / np.linalg.norm(tmplib.Bfield[cells,:], axis=1)

            survivors_fraction[each] = np.sum(survivors)/survivors.shape[0]
            magnetic_fields *= tmplib.gauss_code_to_gauss_cgs # Gauss CGS

            #u_input         = x_input[np.logical_not(survivors),:] # pc
            #x_input         = x_input [survivors,:]                 # pc
            #radius_vectors  = radius_vectors[:, survivors, :]      # pc
            #numb_densities  = numb_densities[:, survivors]         # cm-3
            #mean_column     = mean_column[survivors]               # cm-2
            #median_column   = median_column[survivors]             # cm-2
            #path_column     = path_column[survivors]               # cm-2
            #n_us            = n_rs[np.logical_not(survivors)]     # cm-3
            #n_rs            = n_rs[survivors]                      # cm-3
            #B_rs            = B_rs[survivors]*tmplib.gauss_code_to_gauss_cgs # Gauss CGS
            #r_u             = r_u[survivors]                       # Adim
            #r_l             = r_l[survivors]                       # Adim

            mean_r_u, median_r_u, skew_r_u, kurt_r_u = tmplib.describes(r_u)
            mean_r_l, median_r_l, skew_r_l, kurt_r_l = tmplib.describes(r_l)
            
            stats_dict = {
                "time": _time, 
                "surv_mask": survivors,
                "x_input": x_input,
                "norm_divb": NormDivB,
                "n_rs": n_rs,
                "B_rs": B_rs,
                "n_path": path_column,
                "n_los0": mean_column,
                "n_los1": median_column,
                "surv_fraction": survivors_fraction[each],
                "r_u": r_u,
                "r_l": r_l
            }

            df_stats[str(tmplib.snap)]  = stats_dict

        elif (tmplib.FLAG0 not in sys.argv):
            survivors_fraction[each] = np.sum(survivors)/survivors.shape[0]
            u_input         = x_input[np.logical_not(survivors),:] # pc
            x_input         = x_input[survivors,:]                 # pc
            radius_vectors  = radius_vectors[:, survivors, :]      # pc
            numb_densities  = numb_densities[:, survivors]         # cm-3
            magnetic_fields = magnetic_fields[:, survivors]*tmplib.gauss_code_to_gauss_cgs # Gauss CGS
            mean_column     = mean_column[survivors]               # cm-2
            median_column   = median_column[survivors]             # cm-2
            path_column     = path_column[survivors]               # cm-2
            n_us            = n_rs[np.logical_not(survivors)]     # cm-3
            n_rs            = n_rs[survivors]                      # cm-3
            B_rs            = B_rs[survivors]*tmplib.gauss_code_to_gauss_cgs # Gauss CGS
            r_u             = r_u[survivors]                       # Adim
            r_l             = r_l[survivors]                       # Adim

            #_dist_, cells, _rel_pos_ = tmplib.find_points_and_relative_positions(x_input, tmplib.Pos, tmplib.VoronoiPos)
            #NormDivB = tmplib.MagneticFieldDivergence[cells] * 3 * tmplib.Volume[cells] / (4*np.pi)  / np.linalg.norm(tmplib.Bfield[cells], axis=1)

            stats_dict = {
                "time": _time, 
                "x_input": x_input,
                #"norm_divb": NormDivB,
                "n_rs": n_rs,
                "B_rs": B_rs,
                "n_path": path_column,
                "n_los0": mean_column,
                "n_los1": median_column,
                "surv_fraction": survivors_fraction[each],
                "r_u": r_u,
                "r_l": r_l
            }

            df_stats[str(tmplib.snap)]  = stats_dict


        mean_r_u, median_r_u, skew_r_u, kurt_r_u = tmplib.describes(r_u)
        mean_r_l, median_r_l, skew_r_l, kurt_r_l = tmplib.describes(r_l)
        
        if tmplib.FLAG0 in sys.argv: # -lin
    
            _dist_, cells, _rel_pos_ = tmplib.find_points_and_relative_positions(x_input, tmplib.Pos, tmplib.VoronoiPos)
        
            NormDivB = tmplib.MagneticFieldDivergence[cells] * 3 * tmplib.Volume[cells] / (4*np.pi)  / np.linalg.norm(tmplib.Bfield[cells], axis=1)

            survivors_fraction[each] = np.sum(survivors)/survivors.shape[0]
            magnetic_fields *= tmplib.gauss_code_to_gauss_cgs # Gauss CGS

            field_dict = {
                "time": _time, 
                "surv_mask": survivors,
                "x_input": x_input,
                "n_rs": n_rs,
                "B_rs": B_rs,
                "norm_divb": NormDivB,
                "directions": directions,
                "B_s": magnetic_fields,
                "r_s": radius_vectors,
                "n_s": numb_densities,
                "surv_fraction": survivors_fraction[each],
            }
            print(field_dict.keys())
            
            #df_fields[str(tmplib.snap)]  = field_dict
            df_stats[str(tmplib.snap)]  = field_dict


        if 'Pos' in globals():
            print("\nPos is global", flush=True)

        tmplib.get_globals_memory()            
        # at the end of the loop, drop all that will be reasigned, to avoid memory overflow
        del tree, __0, __1, _1, _2, _3
        del radius_vectors, magnetic_fields, numb_densities

        gc.collect()
        tmplib.config_arepo(filename, center, True)
        tmplib.get_globals_memory()

        if "HOSTNAME" in list(os.environ.keys()):
            os.makedirs(f"/work/bjencinasvelaz/series/{_id_}/", exist_ok=True)
            workdir = f"/work/bjencinasvelaz/series/{_id_}/tmp_{_id_}_rank{rank}.pkl"
            print("Output files saved at :", workdir, flush=True)

            with open(workdir, 'wb') as f:
                pickle.dump(df_stats, f)
                f.flush()
                os.fsync(f.fileno())
        else:
            if tmplib.FLAG0 in sys.argv: # -lin
                os.makedirs("./lines", exist_ok=True)
                workdir = f"./lines/tmp_{_id_}_rank{rank}.pkl"
                print("Output files saved at :", workdir, flush=True)
                with open(workdir, 'wb') as f:
                    pickle.dump(df_stats, f)
                    f.flush()
                    os.fsync(f.fileno())
            else: # -exp or -weight since each of this modify _id_ variable
                os.makedirs(f"./series/{_id_}/", exist_ok=True)
                workdir = f"./series/{_id_}/tmp_{_id_}_rank{rank}.pkl"
                print("Output files saved at :", workdir, flush=True)

                with open(workdir, 'wb') as f:
                    pickle.dump(df_stats, f)
                    f.flush()
                    os.fsync(f.fileno())

    comm.Barrier()
    if rank == 0:
        expected = comm.Get_size()
        
        if "HOSTNAME" in list(os.environ.keys()):
            os.makedirs(f"/work/bjencinasvelaz/series/{_id_}/", exist_ok=True)
            asyncio.run(tmplib.merge_and_save(_id_, tmplib.__dense_cloud__, f"/work/bjencinasvelaz/series/{_id_}/"))
        else:
            if (tmplib.FLAG1 in sys.argv): # -all or -exp
                os.makedirs(f"./series/{_id_}/", exist_ok=True)
                asyncio.run(tmplib.merge_and_save(_id_, tmplib.__dense_cloud__, f"./series/{_id_}"))
            elif tmplib.FLAG0 in sys.argv:
                os.makedirs(f"./lines", exist_ok=True)
                asyncio.run(tmplib.merge_and_save(_id_, tmplib.__dense_cloud__, "./lines"))

    elapsed_time =time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    print(f"Elapsed time: {hours}h {minutes}m {seconds}s")
    
