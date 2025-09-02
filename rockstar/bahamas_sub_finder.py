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

### Cluster ID and DM model 

parser = argparse.ArgumentParser(description="BAHAMAS script")
parser.add_argument("cluster_id", type=str, help="ID of the cluster to process")
parser.add_argument("dm_model", type=str, help="The Dark Matter model to process")
args = parser.parse_args()
cluster_id = args.cluster_id
dm_folder = args.dm_model
print('Processing Cluster ' + cluster_id + ' in ' + dm_folder)

### Import data

data = np.load("/projects/mccleary_group/habjan.e/TNG/Data/BAHAMAS_data/dataGiuliaCerini/" + dm_folder + "/GrNm_0" + cluster_id + ".npz")

coordinates = data['dm_pos'] - data['CoP']
velocities = data['dm_vel']
masses = data['dm_mass']
ids = np.arange(masses.shape[0])

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

lib.rockstar_analyze_fof_group.argtypes = [ctypes.POINTER(Particle), ctypes.c_int64, ctypes.c_int, ctypes.c_double, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_double]
lib.rockstar_analyze_fof_group.restype = ctypes.c_int

### Run the code

num_particles = coordinates.shape[0]

dark_matter_particle_mass = masses[0]

subhalo_fname = f"bahamas_rockstar_subhalos_{dm_folder}_{cluster_id}.list"
member_fname = f"bahamas_rockstar_subhalo_members_{dm_folder}_{cluster_id}.list"

min_particles_in_subhalo = 250
fof_fraction = 0.5

subhalo_fname_b  = subhalo_fname.encode("utf-8")
member_fname_b   = member_fname.encode("utf-8")

status = lib.rockstar_analyze_fof_group(particles, num_particles, 1, dark_matter_particle_mass, subhalo_fname_b, member_fname_b, min_particles_in_subhalo, fof_fraction)
print("Rockstar returned:", status)

end = time.time()
print(f"Elapsed time: {(end - start)/60:.2f} minutes for cluster " + cluster_id + ' in ' + dm_folder)