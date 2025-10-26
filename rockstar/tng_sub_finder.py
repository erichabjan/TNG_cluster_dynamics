import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

import subprocess
import argparse
import os
import struct
import ctypes
from pathlib import Path

import sys
sys.path.append("/home/habjan.e/TNG/Codes/TNG_workshop")
sys.path.append("/home/habjan.e/TNG/TNG_cluster_dynamics")

import iapi_TNG as iapi
import h5py
import TNG_DA

import multiprocessing
import os
print(f"OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS')}")
print(f"Detected CPUs (multiprocessing): {multiprocessing.cpu_count()}")

import time
start = time.time()

### Cluster ID 

parser = argparse.ArgumentParser(description="TNG ROCKSTAR Script")
parser.add_argument("cluster_ID", type=str, help="ID of the cluster to process")
args = parser.parse_args()
cluster_id = args.cluster_ID
print('Processing Cluster ' + cluster_id)

### Download particle data from TNG
halo_cutout_url = f'http://www.tng-project.org/api/TNG300-1/snapshots/99/halos/' + cluster_id + '/cutout.hdf5' 
params={'dm':'Coordinates,ParticleIDs,Velocities'}
fName = '/projects/mccleary_group/habjan.e/TNG/Data/TNG_data/halo_cutouts_dm_' + cluster_id
cutout = iapi.get(halo_cutout_url, params = params, fName = fName)

### Import downloaded cluster

with h5py.File(fName+'.hdf5', 'r') as f:

    coordinates = f['PartType1']['Coordinates'][:]
    velocities = f['PartType1']['Velocities'][:]
    ids = f['PartType1']['ParticleIDs'][:]

### Hard-coded particle DM mass
masses = np.zeros(coordinates.shape[0]) + 5.9 * 10**7
coordinates = coordinates

### Correct coordinates for TNG simulation coordiantes
cluster_id = np.int64(cluster_id)
coordinates = TNG_DA.coord_cm_corr(cluster_ind = cluster_id, coordinates = coordinates) 
coordinates = coordinates * 10**-3     #Convert to Mpc/h


### Load shared ROCKSTAR library

rockstar_path = Path("/home/habjan.e/TNG/Codes/rockstar/librockstar.so")
lib = ctypes.CDLL(str(rockstar_path))

### Define Particle class to make C-readable particle data

class Particle(ctypes.Structure):
    _fields_ = [
        ("id",   ctypes.c_int64),
        ("pos",  ctypes.c_float * 6),
        ("mass", ctypes.c_float),
    ]

### Make particle structure in NumPy similar to C structure

particle_dtype = np.dtype([
    ("id",   np.int64),
    ("pos",  np.float32, (6,)),
    ("mass", np.float32),
], align=True)

N = coordinates.shape[0]
structured = np.empty(N, dtype=particle_dtype)

structured["id"] = ids[:N]

# c Mpc / h 
structured["pos"][:, 0:3] = coordinates[:N].astype(np.float32)
# km / s
structured["pos"][:, 3:6] = velocities[:N].astype(np.float32)
# solar masses
structured["mass"] = masses[:N].astype(np.float32)

assert ctypes.sizeof(Particle) == structured.dtype.itemsize == 40

### Make particle structure for ROCKSTAR input

ParticleArray = Particle * N
particles = ParticleArray()

# Efficient memory copy from NumPy to ctypes array
ctypes.memmove(
    ctypes.addressof(particles),
    structured.ctypes.data,
    structured.nbytes
)

print("ctypes sizeof:", ctypes.sizeof(Particle))
print("numpy itemsize:", structured.dtype.itemsize)

### import `rockstar_analyze_fof_group`

lib.rockstar_analyze_fof_group.argtypes = [ctypes.POINTER(Particle), ctypes.c_int64, ctypes.c_int, 
                                           ctypes.c_double, ctypes.c_char_p, ctypes.c_char_p, 
                                           ctypes.c_int, ctypes.c_double, ctypes.c_double, 
                                           ctypes.c_double, ctypes.c_double]
lib.rockstar_analyze_fof_group.restype = ctypes.c_int

### Additional arugments to run the rockstar code

# Number of particles in the FoF halo

num_particles = coordinates.shape[0]

# Mass of dark matter particles in solar masses

dark_matter_particle_mass = masses[0]

# Output file names

suffix = ''

subhalo_fname = f"rockstar_subhalos_{cluster_id}" + suffix +".list"
member_fname = f"rockstar_subhalo_members_{cluster_id}" + suffix +".list"

subhalo_fname_b  = subhalo_fname.encode("utf-8")
member_fname_b   = member_fname.encode("utf-8")

# Minimum number of particles in a subhalo (mass resolution matched with bahamas)

min_particles_in_subhalo = 4661

# FoF fraction

fof_fraction = 0.7

# DM particle mass in comoving solar masses 

dm_mass_h = masses[0] * 0.667

# TNG softening length in Mpc / h (from Nelson et al. 2019)

softening_in_Mpc_over_h = (1.48 * 10**-3) / 0.667

# Scale factor at z = 0

a_scale_factor = 1

### Run the code

status = lib.rockstar_analyze_fof_group(particles, num_particles, 1, 
                                        dark_matter_particle_mass, subhalo_fname_b, 
                                        member_fname_b, min_particles_in_subhalo, 
                                        fof_fraction, dm_mass_h,
                                        softening_in_Mpc_over_h, a_scale_factor)
print("Rockstar returned:", status)

end = time.time()
print(f"Elapsed time: {(end - start)/60:.2f} minutes")