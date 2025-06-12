import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

import os
import struct
import ctypes
from pathlib import Path

import sys
sys.path.append("/home/habjan.e/TNG/Codes")

import iapi_TNG as iapi
import h5py

import multiprocessing
import os
print(f"OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS')}")
print(f"Detected CPUs (multiprocessing): {multiprocessing.cpu_count()}")

import time
start = time.time()

### Import downloaded cluster

fName = '/home/habjan.e/TNG/Data/TNG_data/halo_cutouts_dm'

with h5py.File(fName+'.hdf5', 'r') as f:

    coordinates = f['PartType1']['Coordinates'][:]
    velocities = f['PartType1']['Velocities'][:]
    ids = f['PartType1']['ParticleIDs'][:]

### Hard-coded particle DM mass
masses = np.zeros(coordinates.shape[0]) + 5.9 * 10**7

### Take a fraction of the data

frac = 0.01

num_particles = coordinates.shape[0]

part_frac = int(num_particles * frac)

coordinates = coordinates[:part_frac, :]
velocities = velocities[:part_frac, :]
ids = ids[:part_frac]
masses = masses[:part_frac]


### Load shared ROCKSTAR library

rockstar_path = Path("/home/habjan.e/TNG/Codes/rockstar/librockstar.so")
lib = ctypes.CDLL(str(rockstar_path))

### Define Particle class to make C-readable particle data

class Particle(ctypes.Structure):
    _fields_ = [
        ("pos", ctypes.c_float * 3),
        ("vel", ctypes.c_float * 3),
        ("mass", ctypes.c_float),
        ("id", ctypes.c_float),
    ]

# particles structure

### Make particle structure in NumPy similar to C structure

particle_dtype = np.dtype([
    ("pos", np.float32, (3,)),
    ("vel", np.float32, (3,)),
    ("mass", np.float32),
    ("id", np.float32)
])

N = coordinates.shape[0]
structured_particles = np.empty(N, dtype=particle_dtype)

structured_particles["pos"] = coordinates.astype(np.float32)
structured_particles["vel"] = velocities.astype(np.float32)
structured_particles["mass"] = masses.astype(np.float32)
structured_particles["id"] = ids.astype(np.float32)

### Make particle structure for ROCKSTAR input

ParticleArray = Particle * N
particles = ParticleArray()

# Efficient memory copy from NumPy to ctypes array
ctypes.memmove(
    ctypes.addressof(particles),
    structured_particles.ctypes.data,
    structured_particles.nbytes
)

### import `rockstar_analyze_fof_group`

lib.rockstar_analyze_fof_group.argtypes = [ctypes.POINTER(Particle), ctypes.c_int64, ctypes.c_int]
lib.rockstar_analyze_fof_group.restype = ctypes.c_int

### Run the code

num_particles = coordinates.shape[0]

status = lib.rockstar_analyze_fof_group(particles, num_particles, 1)
print("Rockstar returned:", status)

end = time.time()
print(f"Elapsed time: {(end - start)/60:.2f} minutes")