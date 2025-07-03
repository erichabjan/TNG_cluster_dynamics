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
sys.path.append("/home/habjan.e/TNG/Codes")
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

parser = argparse.ArgumentParser(description="Script that accepts a string input.")
parser.add_argument("input_string", type=str, help="The input string to process")
args = parser.parse_args()
cluster_id = args.input_string
print('Processing Cluster ' + cluster_id)

### Download particle data from TNG

halo_cutout_url = f'http://www.tng-project.org/api/TNG300-1/snapshots/99/halos/' + cluster_id + '/cutout.hdf5' 
params={'dm':'Coordinates,ParticleIDs,Velocities'}
fName = '/home/habjan.e/TNG/Data/TNG_data/halo_cutouts_dm_' + cluster_id
cutout = iapi.get(halo_cutout_url, params = params, fName = fName)

### Import downloaded cluster

with h5py.File(fName+'.hdf5', 'r') as f:

    coordinates = f['PartType1']['Coordinates'][:]
    velocities = f['PartType1']['Velocities'][:]
    ids = f['PartType1']['ParticleIDs'][:]

### Hard-coded particle DM mass
masses = np.zeros(coordinates.shape[0]) + 5.9 * 10**7

### Take a fraction of the data

#frac = 0.001

#num_particles = coordinates.shape[0]

#part_frac = int(num_particles * frac)

#coordinates = coordinates[:part_frac, :]
#velocities = velocities[:part_frac, :]
#ids = ids[:part_frac]
#masses = masses[:part_frac]

### Correct coordinates for TNG simulation coordiantes
cluster_id = np.int64(cluster_id)
coordinates = TNG_DA.coord_cm_corr(cluster_ind = cluster_id, coordinates = coordinates) 


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
structured["pos"][:, 0:3] = coordinates[:N].astype(np.float32)
structured["pos"][:, 3:6] = velocities[:N].astype(np.float32)
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

lib.rockstar_analyze_fof_group.argtypes = [ctypes.POINTER(Particle), ctypes.c_int64, ctypes.c_int, ctypes.c_int64]
lib.rockstar_analyze_fof_group.restype = ctypes.c_int

### Run the code

num_particles = coordinates.shape[0]

status = lib.rockstar_analyze_fof_group(particles, num_particles, 1, cluster_id)
print("Rockstar returned:", status)

end = time.time()
print(f"Elapsed time: {(end - start)/60:.2f} minutes")