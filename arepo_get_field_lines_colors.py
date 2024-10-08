from collections import defaultdict
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy import spatial
import healpy as hp
import numpy as np
import random
import shutil
import h5py
import json
import sys
import os

from library import *

import time

start_time = time.time()

"""  
Using Margo Data

Analysis of reduction factor

$$N(s) 1 - \sqrt{1-B(s)/B_l}$$

Where $B_l$ corresponds with (in region of randomly choosen point) the lowest between the highest peak at both left and right.
where $s$ is a random chosen point at original 128x128x128 grid.

1.- Randomly select a point in the 3D Grid. 
2.- Follow field lines until finding B_l, if non-existent then change point.
3.- Repeat 10k times
4.- Plot into a histogram.

contain results using at least 20 boxes that contain equally spaced intervals for the reduction factor.

# Calculating Histogram for Reduction Factor in Randomized Positions in the 128**3 Cube 

"""

"""
Parameters

- [N] default total number of steps in the simulation
- [dx] default 4/N of the rloc_boundary (radius of spherical region of interest) variable

"""
FloatType = np.float64
IntType = np.int32

if len(sys.argv)>2:
	# first argument is a number related to rloc_boundary
	N=int(sys.argv[1])
	rloc_boundary=float(sys.argv[2])
	max_cycles   =int(sys.argv[3])
else:
    N            =100
    rloc_boundary=256   # rloc_boundary for boundary region of the cloud
    max_cycles   =1

"""  B. Jesus Velazquez """

snap = '117'
filename = 'arepo_data/snap_'+ snap + '.hdf5'

data = h5py.File(filename, 'r')
Boxsize = data['Header'].attrs['BoxSize'] #

# Directly convert and cast to desired dtype
VoronoiPos = np.asarray(data['PartType0']['Coordinates'], dtype=FloatType)
Pos = np.asarray(data['PartType0']['CenterOfMass'], dtype=FloatType)
Bfield = np.asarray(data['PartType0']['MagneticField'], dtype=FloatType)
Density = np.asarray(data['PartType0']['Density'], dtype=FloatType)
Mass = np.asarray(data['PartType0']['Masses'], dtype=FloatType)

# Initialize gradients
Bfield_grad = np.zeros((len(Pos), 9))
Density_grad = np.zeros((len(Density), 3))

print(filename, "Loaded (1) :=: time ", (time.time()-start_time)/60.)

Volume   = Mass/Density

#Center= 0.5 * Boxsize * np.ones(3) # Center
#Center = np.array( [91,       -110,          -64.5]) #117
#Center = np.array( [96.6062303,140.98704002, 195.78020632]) #117
Center = Pos[np.argmax(Density),:] #430
print("Center before Centering", Center)

VoronoiPos-=Center
Pos-=Center

xPosFromCenter = Pos[:,0]
Pos[xPosFromCenter > Boxsize/2,0]       -= Boxsize
VoronoiPos[xPosFromCenter > Boxsize/2,0] -= Boxsize

def get_along_lines(x_init):

    m = x_init.shape[0]

    line      = np.zeros((N+1,m,3)) # from N+1 elements to the double, since it propagates forward and backward
    bfields   = np.zeros((N+1,m))
    densities = np.zeros((N+1,m))
    volumes   = np.zeros((N+1,m))
    threshold = np.zeros((m,)).astype(int) # one value for each

    line_rev=np.zeros((N+1,m,3)) # from N+1 elements to the double, since it propagates forward and backward
    bfields_rev = np.zeros((N+1,m))
    densities_rev = np.zeros((N+1,m))
    volumes_rev   = np.zeros((N+1,m))
    threshold_rev = np.zeros((m,)).astype(int) # one value for each

    line[0,:,:]     = x_init
    line_rev[0,:,:] = x_init

    x = x_init.copy()

    dummy, bfields[0,:], densities[0,:], cells = find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos, VoronoiPos)

    vol = Volume[cells]
    densities[0,:] = densities[0,:] * gr_cm3_to_nuclei_cm3
    dens = densities[0,:] * gr_cm3_to_nuclei_cm3

    k=0

    while np.any((dens > 100)):

        # Create a mask for values that are still above the threshold
        mask = dens > 100

        un_masked = np.logical_not(mask)

        aux = x[un_masked]

        x, bfield, dens, vol = Heun_step(x, +1, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)
        dens = dens * gr_cm3_to_nuclei_cm3

        threshold += mask.astype(int)  # Increment threshold count only for values still above 100
        x[un_masked] = aux
        print(threshold)

        line[k+1,:,:]    = x
        volumes[k+1,:]   = vol
        bfields[k+1,:]   = bfield
        densities[k+1,:] = dens
        
        if np.all(un_masked):
            print("All values are False: means all crossed the threshold")
            break

        k += 1
    
    threshold = threshold.astype(int)
    larger_cut = np.max(threshold)
    
    x = x_init.copy()

    dummy_rev, bfields_rev[0,:], densities_rev[0,:], cells = find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos, VoronoiPos)

    vol = Volume[cells]

    densities_rev[0,:] = densities_rev[0,:] * gr_cm3_to_nuclei_cm3
    dens = densities_rev[0,:] * gr_cm3_to_nuclei_cm3
    
    print(line_rev.shape)

    k=0

    while np.any((dens > 100)):

        # Create a mask for values that are still above the threshold
        mask = dens > 100

        un_masked = np.logical_not(mask)

        aux = x[un_masked]

        x, bfield, dens, vol = Heun_step(x, -1, Bfield, Density, Density_grad, Pos, VoronoiPos, Volume)
        dens = dens * gr_cm3_to_nuclei_cm3

        threshold_rev += mask.astype(int)  # Increment threshold count only for values still above 100
        x[un_masked] = aux

        line_rev[k+1,:,:] = x
        volumes_rev[k+1,:] = vol
        bfields_rev[k+1,:] = bfield
        densities_rev[k+1,:] = dens 
                    
        if np.all(un_masked):  # ~arr negates the array
            print("All values are False: means all crossed the threshold")
            break
        k += 1

    threshold = threshold.astype(int)
    threshold_rev = threshold_rev.astype(int)

    for cut in threshold:
        line[:cut,:,:] = line[cut-1,:,:]
        volumes[:cut,:,:]   = volumes[cut-1,:]
        bfields[:cut,:,:]   = bfields[cut-1,:]
        densities[:cut,:,:] = densities[cut-1,:]

    for cut in threshold_rev:
        line_rev[:cut,:,:] = line_rev[cut-1,:,:]
        volumes_rev[:cut,:]   = volumes_rev[cut-1,:]
        bfields_rev[:cut,:]   = bfields_rev[cut-1,:]
        densities_rev[:cut,:] = densities_rev[cut-1,:,:]

    radius_vector = np.append(line_rev[::-1, :, :], line, axis=0)
    magnetic_fields = np.append(bfields_rev[::-1, :], bfields, axis=0)
    numb_densities = np.append(densities_rev[::-1, :], densities, axis=0)
    volumes_all = np.append(volumes_rev[::-1, :], volumes, axis=0)

    #gas_densities   *= 1.0* 6.771194847794873e-23                      # M_sol/pc^3 to gram/cm^3
    #numb_densities   = gas_densities.copy() * 6.02214076e+23 / 1.00794 # from gram/cm^3 to Nucleus/cm^3
    
    # Initialize trajectory and radius_to_origin with the same shape
    trajectory      = np.zeros_like(magnetic_fields)
    radius_to_origin= np.zeros_like(magnetic_fields)

    print("Magnetic fields shape:", magnetic_fields.shape)
    print("Radius vector shape:", radius_vector.shape)
    print("Numb densities shape:", numb_densities.shape)
	
    for _n in range(m): # Iterate over the first dimension
        prev = radius_vector[0, _n, :]
        for k in range(magnetic_fields.shape[0]):  # Iterate over the first dimension
            radius_to_origin[k, _n] = magnitude(radius_vector[k, _n, :])
            cur = radius_vector[k, _n, :]
            diff_rj_ri = magnitude(cur, prev)
            trajectory[k,_n] = trajectory[k-1,_n] + diff_rj_ri            
            prev = radius_vector[k, _n, :]
    
    trajectory[0,:]  = 0.0

    radius_vector   *= 1.0* 3.086e+18                                # from Parsec to cm
    trajectory      *= 1.0* 3.086e+18                                # from Parsec to cm
    magnetic_fields *= 1.0* (1.99e+33/(3.086e+18*100_000.0))**(-1/2) # in Gauss (cgs)
    
    return bfields[0,:], radius_vector, trajectory, magnetic_fields, numb_densities, volumes, radius_to_origin

rloc_center      = np.array([float(random.uniform(0,rloc_boundary)) for l in range(max_cycles)])
nside = max_cycles     # sets number of cells sampling the spherical boundary layers = 12*nside**2
npix  = 12 * nside ** 2
ipix_center       = np.arange(npix)
xx,yy,zz = hp.pixelfunc.pix2vec(nside, ipix_center)

xx = np.array(random.sample(list(xx), max_cycles))
yy = np.array(random.sample(list(yy), max_cycles))
zz = np.array(random.sample(list(zz), max_cycles))

m = len(zz) # amount of values that hold which_up_down

x_init = np.zeros((m,3))

x_init[:,0]      = rloc_center * xx[:]
x_init[:,1]      = rloc_center * yy[:]
x_init[:,2]      = rloc_center * zz[:]

lmn = N

print("Cores Used         : ", os.cpu_count())
print("Steps in Simulation: ", 2*N)
print("rloc_boundary      : ", rloc_boundary)
print("rloc_center        :\n ", rloc_center)
print("max_cycles         : ", max_cycles)
print("Boxsize            : ", Boxsize) # 256
print("Center             : ", Center) # 256
print("Posit Max Density  : ", Pos[np.argmax(Density),:]) # 256
print("Smallest Volume    : ", Volume[np.argmin(Volume)]) # 256
print("Biggest  Volume    : ", Volume[np.argmax(Volume)]) # 256
print(f"Smallest Density  : {Density[np.argmin(Density)]}")
print(f"Biggest  Density  : {Density[np.argmax(Density)]}")

__, radius_vector, trajectory, magnetic_fields, numb_densities, volumes, radius_to_origin = get_along_lines(x_init)

print("Elapsed Time: ", (time.time() - start_time)/60.)

with open('output', 'w') as file:
    file.write(f"{filename}\n")
    file.write(f"Cores Used: {os.cpu_count()}\n")
    file.write(f"Steps in Simulation: {2 * N}\n")
    file.write(f"rloc_boundary (Pc) : {rloc_boundary}\n")
    file.write(f"rloc_center (Pc)   :\n {rloc_center}\n")
    file.write(f"max_cycles         : {max_cycles}\n")
    file.write(f"Boxsize (Pc)       : {Boxsize} Pc\n")
    file.write(f"Center (Pc, Pc, Pc): {Center[0]}, {Center[1]}, {Center[2]} \n")
    file.write(f"Posit Max Density (Pc, Pc, Pc): {Pos[np.argmax(Density), :]}\n")
    file.write(f"Smallest Volume (Pc^3)   : {Volume[np.argmin(Volume)]} \n")
    file.write(f"Biggest  Volume (Pc^3)   : {Volume[np.argmax(Volume)]}\n")
    file.write(f"Smallest Density (M☉/Pc^3)  : {Density[np.argmin(Volume)]} \n")
    file.write(f"Biggest  Density (M☉/Pc^3) : {Density[np.argmax(Volume)]}\n")
    file.write(f"Elapsed Time (Minutes)     : {(time.time() - start_time)/60.}\n")

for i in range(m):

    pocket, global_info = pocket_finder(magnetic_fields[:,i], i, plot=True) # this plots

    np.save(f"arepo_npys/ArePositions{i}.npy", radius_vector[:,i,:])
    np.save(f"arepo_npys/ArepoTrajectory{i}.npy", trajectory[:,i])
    np.save(f"arepo_npys/ArepoNumberDensities{i}.npy", numb_densities[:,i])
    np.save(f"arepo_npys/ArepoMagneticFields{i}.npy", magnetic_fields[:,i])
	
    print(f"finished line {i+1}/{max_cycles}",(time.time()-start_time)/60)

    if True:
        # Create a figure and axes for the subplot layout
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))
        
        axs[0,0].plot(trajectory[:,i], magnetic_fields[:,i], linestyle="--", color="m")
        axs[0,0].scatter(trajectory[:,i], magnetic_fields[:,i], marker="+", color="m")
        axs[0,0].scatter(trajectory[lmn,i], magnetic_fields[lmn,i], marker="x", color="black")
        axs[0,0].set_xlabel("s (cm)")
        axs[0,0].set_ylabel("$B(s)$ Gauss (cgs)")
        axs[0,0].set_title("Magnetic FIeld")
        axs[0,0].grid(True)
		
        axs[0,1].plot(trajectory[:,i], radius_to_origin[:,i], linestyle="--", color="m")
        axs[0,1].scatter(trajectory[:,i], radius_to_origin[:,i], marker="+", color="m")
        axs[0,1].scatter(trajectory[lmn,i], radius_to_origin[lmn,i], marker="x", color="black")
        axs[0,1].set_xlabel("s (cm)")
        axs[0,1].set_ylabel("$r$ cm (cgs)")
        axs[0,1].set_title("Distance Away of MaxDensityCoord $r$ ")
        axs[0,1].grid(True)

        axs[1,0].plot(trajectory[:,i], numb_densities[:,i], linestyle="--", color="m")
        axs[1,0].scatter(trajectory[:,i], numb_densities[:,i], marker="+", color="m")
        axs[1,0].scatter(trajectory[lmn,i], numb_densities[lmn,i], marker="x", color="black")
        axs[1,0].set_yscale('log')
        axs[1,0].set_xlabel("s (cm)")
        axs[1,0].set_ylabel("$n_g(s)$ gr/cm^3 (cgs)")
        axs[1,0].set_title("Number Density (Nucleons/cm^3) ")
        axs[1,0].grid(True)
		
        axs[1,1].plot(volumes[:,i], linestyle="-", color="black")
        axs[1,1].set_yscale('log')
        axs[1,1].set_xlabel("steps")
        axs[1,1].set_ylabel("$V(s)$ cm^3 (cgs)")
        axs[1,1].set_title("Cells Volume along Path")
        axs[1,1].grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the figure
        plt.savefig(f"field_shapes/shapes_{i}.png")

        # Close the plot
        plt.close(fig)
		
if True:
	ax = plt.figure().add_subplot(projection='3d')

	for k in range(m):
		x=radius_vector[:,k,0]/ 1.496e13                            
		y=radius_vector[:,k,1]/ 1.496e13
		z=radius_vector[:,k,2]/ 1.496e13
		
		for l in range(len(radius_vector[:,0,0])):
			ax.plot(x[l:l+2], y[l:l+2], z[l:l+2], color="m",linewidth=0.3)

		#ax.scatter(x_init[0], x_init[1], x_init[2], marker="v",color="m",s=10)
		ax.scatter(x[0], y[0], z[0], marker="x",color="g",s=6)
		ax.scatter(x[lmn], y[lmn], z[lmn], marker="x", color="black",s=6)
		ax.scatter(x[-1], y[-1], z[-1], marker="x", color="r",s=6)
		
	zoom = 256#np.max(radius_to_origin)
	ax.set_xlim(-zoom,zoom)
	ax.set_ylim(-zoom,zoom)
	ax.set_zlim(-zoom,zoom)
	ax.set_xlabel('x [Pc]')
	ax.set_ylabel('y [Pc]')
	ax.set_zlabel('z [Pc]')
	ax.set_title('Magnetic field morphology')

	plt.savefig(f'field_shapes/MagneticFieldTopology.png', bbox_inches='tight')

	plt.show()

if False:

    from mpl_toolkits.mplot3d import Axes3D

    x_center, y_center, z_center = Center[0], Center[1], Center[2]

    region_size = 1  # Pc

    # Filter the data to keep only the points within 10 units of the center
    mask = (
        (np.abs(Pos[:, 0] - x_center) < region_size) &
        (np.abs(Pos[:, 1] - y_center) < region_size) &
        (np.abs(Pos[:, 2] - z_center) < region_size)
    )

    # Apply the mask to filter Coordinates and magnetic field components
    x_filtered = Pos[mask, 0]
    y_filtered = Pos[mask, 1]
    z_filtered = Pos[mask, 2]
    Bx_filtered = Bfield[mask, 0]
    By_filtered = Bfield[mask, 1]
    Bz_filtered = Bfield[mask, 2]

    # Plot the filtered data in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the magnetic field using quiver plot in 3D
    ax.quiver(x_filtered, y_filtered, z_filtered, Bx_filtered, By_filtered, Bz_filtered, length=0.1, normalize=True)

    # Set axis labels and title
    ax.set_xlabel('X [Pc]')
    ax.set_ylabel('Y [Pc]')
    ax.set_zlabel('Z [Pc]')
    plt.title('3D Magnetic Field Direction in Center Region')

    # Show the plot
    plt.show()