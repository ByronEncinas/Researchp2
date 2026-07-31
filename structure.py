import matplotlib
matplotlib.use("Agg") 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import skew, kurtosis
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib as mpl
from copy import deepcopy
#from src.library import *
import pandas as pd
import numpy as np
import os, glob, sys, time
import numpy as np
import cv2
import glob
from matplotlib.colors import LogNorm, Normalize

mpl.rcParams['text.usetex'] = True

gauss_code_to_gauss_cgs = (4 * np.pi)**0.5   * (3.086e18)**(-1.5) * (1.99e33)**0.5 * 1e5 # cgs units
pc_to_cm = 3.086 * 1.0e+18  # cm per parsec
gauss_to_micro_gauss = 1e+6

mpl.rcParams['text.latex.preamble'] = r'\usepackage{mathrsfs}'
mpl.rcParams['text.usetex'] = True
MARKERS = ['v', 'o']
COLORS  = ["#8E2BAF", "#148A02"]
ALPHA   = 0.9
SIZE    = 8
FONTSIZE = 18
GRID_ALPHA = 0.5

from src.library import *
import pandas as pd
import numpy as np
import os, glob, sys

def imporfromfile(file, identifier):

    df = pd.read_pickle(file)
    df.index.name = 'snapshot'
    df.index = df.index.astype(int)
    df = df.sort_index()
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    
    if df.empty:
        return False
        
    t     = df["time"].to_numpy()
    _ = np.argsort(t)
    t     = df["time"].to_numpy()[_]
    x     = df["x_input"].to_numpy()[_]
    r     = df["r_s"].to_numpy()[_]
    n     = df["n_s"].to_numpy()[_]
    B     = df["B_s"].to_numpy()[_]

    radius = np.ceil(np.max([np.max(ris) for ris in x])*100)/100

    if identifier[0] == '6'  and identifier[-1] == '4':
        radius = np.ceil(np.max([np.max(ris) for ris in x])*100)/100
    elif identifier == '4i3' or identifier == '4a3':
        radius = 0.1
    elif identifier == '2i2' or identifier == '2a2':
        radius = 0.5
    elif identifier == '2i1' or identifier == '2a1':
        radius = 0.2
    elif identifier == '2i0' or identifier == '2a0':
        radius = 0.1
        
    if "e-" in identifier:
        int_extract = float(identifier.split("e-")[-1][0])
        radius = 10**int_extract

    return {
        "id": identifier,
        "t":     t,
        "r":     r,
        "x":     x,
        "n":     n,
        "B":     B
    }

def _fieldstructure(r, b, out = ""):

    from matplotlib import cm
    from matplotlib.colors import Normalize
    m = b.shape[1]
    zoom = 0.8
    zoom2= 1.2
    ax = plt.figure().add_subplot(projection='3d')

    for k in range(0,m,1):
        # mask makes sure that start and ending point arent the zero vector
        #x0=r0[k, 0]
        #y0=r0[k, 1]
        #z0=r0[k, 2]
        #ax.scatter(x0, y0, z0, marker="x",color="black",s=1,alpha=0.05)   
        x=r[:,k, 0]
        y=r[:,k, 1]
        z=r[:,k, 2]
        mk0  = b[:, k] > 0
        mk1  = x*x + y*y + z*z < (zoom2)**2
        mk = np.logical_and(mk0,mk1)
        x=r[mk,k, 0]
        y=r[mk,k, 1]
        z=r[mk,k, 2]
        bhat = b[mk, k]
        bhat /= np.max(bhat)
        norm = LogNorm(vmin=np.min(bhat), vmax=np.max(bhat))
        cmap = cm.viridis

        #ax.scatter(x0, y0, z0, marker="x",color="g",s=1, alpha=0.5, label="X")            
        for l in range(len(x) - 1):
            color = cmap(norm(bhat[l]))
            ax.plot(x[l:l+2], y[l:l+2], z[l:l+2], color=color, linewidth=0.3)
            if np.all(x[l:l+2]) == 0.0:
                break
    #ax.scatter(rot_plane[:,0], rot_plane[:,1], rot_plane[:,2], marker="x",color="r",s=3, alpha=0.25)            
    #ax.quiver(0,0,0, r_rxb_z[0],r_rxb_z[1],r_rxb_z[2],color="black")
    ax.set_xlim(-zoom,zoom)
    ax.set_ylim(-zoom,zoom)
    ax.set_zlim(-zoom,zoom)
    ax.set_xlabel('x [Pc]')
    ax.set_ylabel('y [Pc]')
    ax.set_zlabel('z [Pc]')
    ax.set_title('Magnetic field morphology')
    #ax.view_init(elev=elevation, azim=azimuth)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Arbitrary Units')
    plt.savefig(f"./FieldStructure{out}.png", bbox_inches='tight', dpi=300)
    print(f"./FieldStructure{out}.png <--- saved")

def fieldstructure(r, b, out="", elev=30, azim=-60):

    from matplotlib import cm
    from matplotlib.colors import LogNorm
    import matplotlib.pyplot as plt
    import numpy as np

    m = b.shape[1]
    zoom  = 1.0
    zoom2 = 1.5

    fig = plt.figure()
    ax  = fig.add_subplot(projection='3d')
    ax.set_proj_type('persp', focal_length=0.4)  # stronger perspective 0.4
    ax.view_init(elev=elev, azim=azim)

    # approximate camera direction from viewing angles
    elev_r = np.radians(elev)
    azim_r = np.radians(azim)
    cam = np.array([
        np.cos(elev_r) * np.cos(azim_r),
        np.cos(elev_r) * np.sin(azim_r),
        np.sin(elev_r)
    ]) * zoom2 * 3  # place camera outside the scene

    # precompute max possible depth for normalization
    max_depth = np.linalg.norm(cam) + zoom2

    cmap = cm.viridis
    norm = None  # will be set in first valid line

    for k in range(0, m, 2):
        x = r[:, k, 0]
        y = r[:, k, 1]
        z = r[:, k, 2]
        mk0 = b[:, k] > 0
        mk1 = x*x + y*y + z*z < zoom2**2
        mk  = np.logical_and(mk0, mk1)
        x, y, z = r[mk, k, 0], r[mk, k, 1], r[mk, k, 2]
        bhat = b[mk, k]
        if len(bhat) == 0:
            continue
        bhat = bhat / np.max(bhat)
        norm = Normalize(vmin=np.min(bhat), vmax=np.max(bhat))

        for l in range(len(x) - 1):
            if np.all(x[l:l+2] == 0.0):
                break

            # depth cueing: distance of segment midpoint from camera
            mid   = np.array([x[l], y[l], z[l]])
            depth = np.linalg.norm(mid - cam)
            t     = np.clip(depth / max_depth, 0.0, 1.0)  # 0=close, 1=far

            alpha = float(1.0 - 0.75 * t)   # 1.0 (close) → 0.25 (far)
            lw    = float(0.8 - 0.55 * t)   # 0.8 (close) → 0.25 (far)

            color = cmap(norm(bhat[l]))
            ax.plot(x[l:l+2], y[l:l+2], z[l:l+2],
                    color=color, linewidth=lw, alpha=alpha)

    ax.set_xlim(-zoom, zoom)
    ax.set_ylim(-zoom, zoom)
    ax.set_zlim(-zoom, zoom)
    ax.set_xlabel('x [Pc]')
    ax.set_ylabel('y [Pc]')
    ax.set_zlabel('z [Pc]')
    ax.set_title('Magnetic field morphology')

    if norm is not None:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Arbitrary Units')

    plt.savefig(f"./{out}.png", bbox_inches='tight', dpi=300)
    print(f"./FieldStructure{out}.png <--- saved")
    plt.close(fig)

def calc_distance(r, b, n, jth=""):

    m = b.shape[1]
    distances = []
    
    plt.figure(figsize=(8, 4))
    
    for k in range(m):
        x  = r[:, k, 0]
        y  = r[:, k, 1]
        z  = r[:, k, 2]
        rk = r[:, k, :]
        
        #mk0 = b[:, k] > 0        
        mk0 = n[:, k] > 0
        aux = np.where(n[:, k] >  10**2)
        bk = b[aux[0][0]:aux[0][-1],k]
        if (np.sum(aux) == 0):
            continue

        nk = n[aux[0][0]:aux[0][-1],k]
        distances += [np.sum(np.linalg.norm(np.diff(rk[aux[0][0]:aux[0][-1]], axis=0), axis=1))]
        distance = np.cumsum(np.linalg.norm(np.diff(rk[aux[0][0]:aux[0][-1]], axis=0), axis=1))
        plt.plot(distance, bk[1:], '-' , linewidth=1, markersize=4)
        plt.plot(distance, nk[1:], '--', linewidth=1, markersize=4)
    #plt.plot(range(m), distances, marker='o', linewidth=1, markersize=4)
    #plt.xlabel("arbitrary ID (k)")
    plt.xlabel("Distance (pc)")
    plt.ylabel("B (Gauss) | n_{gas} (cm$^{-3}$)")
    plt.yscale("log")
    plt.xscale("log")
    plt.title("Distance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./owo{jth}.jpeg")

def _proj_plot(r, b, n, jth=""):

    m = b.shape[1]
    distances = []
    
    plt.figure(figsize=(8, 4))
    
    for k in range(m):
        x  = r[:, k, 0]
        y  = r[:, k, 1]
        z  = r[:, k, 2]
        rk = r[:, k, :]
        
        #mk0 = b[:, k] > 0        
        mk0 = n[:, k] > 0
        aux = np.where(n[:, k] > 3 * 10**2)
        bk = b[aux[0][0]:aux[0][-1],k]*gauss_to_micro_gauss
        if (np.sum(aux) == 0):
            continue
        plt.scatter(x, y, s= 1, color="black", alpha = 0.5)
    plt.xlabel("X [pc])")
    plt.ylabel("Y [pc]")

    plt.title("Distance per B-line Iteration")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./{jth}.jpeg")

def proj_plot(r, b, n, jth=""):
    import matplotlib.colors as mcolors

    m = b.shape[1]
    distances = []

    fig, ax = plt.subplots(figsize=(8, 4))

    # Collect all n values across active field lines first, for a shared colormap range
    all_n_vals = []
    for k in range(m):
        aux = np.where(n[:, k] > 3 * 10**2)
        if np.sum(aux) == 0:
            continue
        all_n_vals.append(n[aux[0][0]:aux[0][-1], k])

    if not all_n_vals:
        print("No active field lines found.")
        return

    n_concat = np.concatenate(all_n_vals)
    vmin, vmax = n_concat.min(), n_concat.max()
    norm = mcolors.LogNorm(vmin=max(vmin, 1e-10), vmax=vmax)  # log scale suits density
    cmap = plt.cm.plasma

    sc = None
    for k in range(m):
        aux = np.where(n[:, k] > 1.0e+2)
        if np.sum(aux) == 0:
            continue

        idx = slice(aux[0][0], aux[0][-1])
        x  = r[idx, k, 0]
        y  = r[idx, k, 1]
        nk = n[idx, k]

        sc = ax.scatter(x, y, c=nk, cmap=cmap, norm=norm, s=2, alpha=0.8)

    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Density n [cm$^{-3}$]")  # adjust units as needed

    ax.set_xlabel("X [pc]")
    ax.set_ylabel("Y [pc]")
    ax.set_xlim(-1/2,1/2)
    ax.set_ylim(-1/2,1/2)
    ax.set_title("Distance per B-line Iteration")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"./{jth}.jpeg", dpi=150)
    plt.close(fig)

#files = sorted(glob.glob(f'../m-c-data/series/rloc/*.pkl'))
files = sorted(glob.glob(f'./lines/data*.pkl'))
#files = sorted(glob.glob(f'./series/data*.pkl'))
print(files)
df = pd.DataFrame()  # empty DataFrame

for index, file in enumerate(files[::-1]):
    ID = file.split("/")[-1].split(".")[0][-3:]
    print(ID)
    data = imporfromfile(file, ID)
    if not data:
        continue
    info = {
        "identifier": ID,
        "times"  : data["t"].shape[0],
        "radius" : data["r"],
        "lines"  : data["x"],
        "field"  : data["B"],
        "dens"   : data["n"]
    }
    for i, t in enumerate(data["t"]):
        #calc_distance(data["r"][i],data["B"][i],data["n"][i], f"profile-{i}-{ID}")
        #proj_plot(data["r"][i],data["B"][i],data["n"][i], f"projplt-{i}-{ID}")
        fieldstructure(data["r"][i], data["B"][i], out = ID + f"fieldtop-{i}-{ID}")

    if index == 0:
        df = pd.DataFrame([info])
    else:
        df = pd.concat([df, pd.DataFrame([info])], ignore_index=True)
