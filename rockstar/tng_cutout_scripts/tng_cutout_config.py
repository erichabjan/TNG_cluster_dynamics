#!/usr/bin/env python3
"""
What each TNG300 resolution needs, for tng_chunking.py and the 5r200 cutout
scripts. The scripts look everything up here from --sim, so the slurm files
only choose which simulations and particle types to run.

For each (simulation, particle type) the snapshot fields come from one of two
places, decided chunk by chunk:
  1. an EXISTING source already on Explorer, if that chunk file is there and
     holds every required field (TNG300-1 gas: the full snapshot-99 gas chunks
     in /projects/mccleary_group/mcdonough.b/TNG_dust/data/TNG300_gas/snap99);
  2. otherwise the chunk CACHE on /scratch, filled by tng_chunking.py with only
     the fields listed in FIELDS.

DM is not chunked here: the dm_within_5r200_<id>.hdf5 cutouts already exist
for all three resolutions and are complete.
"""
import os

SNAP = 99
API_BASE_URL = "http://www.tng-project.org/api/"

# Downloaded chunks: <CHUNK_CACHE_DIR>/<sim>/snap_<snap>/snapshot_chunks/snapshot-<snap>.<i>.hdf5
CHUNK_CACHE_DIR = "/scratch/habjan.e/_tmp_tng_chunks"
# Finished cutouts: <CUTOUT_ROOT>/5r200_data-<n>/<ptype>_within_5r200_<halo>.hdf5
CUTOUT_ROOT = "/projects/mccleary_group/habjan.e/TNG/Data/TNG_data"
HALO_LIST = "/home/habjan.e/TNG/Sandbox_notebooks/rockstar_sandboxes/halo_ids.txt"

# Halo centre and radius come from these local group catalogs
# (<GROUPCAT_DIR>/<sim>_<field>.hdf5, as written by iapi_TNG) rather than the
# per-halo API info.json, which returns intermittent 503s. They reproduce the
# r200 attr of every existing 5r200 cutout bit for bit. BOXSIZE in ckpc/h.
GROUPCAT_DIR = "/home/habjan.e/TNG/Data/TNG_data"
BOXSIZE = {"TNG300-1": 205000.0, "TNG300-2": 205000.0, "TNG300-3": 205000.0}

# Particle-type names: snapshot group number, and the name the API's field
# filter expects (black holes are 'bhs' there; 'bh' is rejected).
PARTTYPE_NUM = {"gas": 0, "dm": 1, "stars": 4, "bh": 5}
API_TYPE_NAME = {"gas": "gas", "dm": "dm", "stars": "stars", "bh": "bhs"}

# Fields each cutout needs from the snapshot chunks.
#   gas  : added to the existing gas_within_5r200 cutouts (which already hold
#          Coordinates, Velocities, Masses, ParticleIDs). Coordinates select the
#          5 R200 sphere chunk by chunk, ParticleIDs match cells to the cutout,
#          and Density / InternalEnergy / ElectronAbundance are what the X-ray
#          maps need (temperature is derived from u and x_e).
#   stars: GFM_StellarFormationTime separates stars (> 0) from wind particles
#          (<= 0), which are kept in the cutout and dropped downstream.
FIELDS = {
    "gas": ("Coordinates", "ParticleIDs", "Density", "InternalEnergy",
            "ElectronAbundance", "StarFormationRate"),
    "stars": ("Coordinates", "Velocities", "Masses", "ParticleIDs",
              "GFM_StellarFormationTime"),
    "bh": ("Coordinates", "Velocities", "Masses", "ParticleIDs"),
}

# Particle types tng_chunking.py fetches for every simulation below.
CHUNKED_PARTTYPES = ("gas", "stars", "bh")

# n_chunks is the snapshot's NumFilesPerSnapshot. `existing` maps a particle
# type to (directory, filename pattern) of chunk files already on disk.
SIMS = {
    "TNG300-1": {
        "n_chunks": 600,
        "cutout_dir": "5r200_data-1",
        "existing": {
            "gas": ("/projects/mccleary_group/mcdonough.b/TNG_dust/data/TNG300_gas/snap99",
                    "snap_{snap:03d}.{idx}.hdf5"),
        },
    },
    "TNG300-2": {"n_chunks": 100, "cutout_dir": "5r200_data-2", "existing": {}},
    "TNG300-3": {"n_chunks": 16, "cutout_dir": "5r200_data-3", "existing": {}},
}


def sim_info(sim):
    if sim not in SIMS:
        raise KeyError(f"{sim!r} is not configured in tng_cutout_config.SIMS "
                       f"(have {sorted(SIMS)})")
    return SIMS[sim]


def cache_chunk_path(sim, idx, snap=SNAP):
    """Where tng_chunking.py writes, and the cutout scripts read, chunk idx."""
    return os.path.join(CHUNK_CACHE_DIR, sim, f"snap_{snap}", "snapshot_chunks",
                        f"snapshot-{snap}.{idx}.hdf5")


def existing_chunk_path(sim, ptype, idx, snap=SNAP):
    """Path of chunk idx in an existing on-disk source for ptype, or None."""
    src = sim_info(sim)["existing"].get(ptype)
    if src is None:
        return None
    directory, pattern = src
    return os.path.join(directory, pattern.format(snap=snap, idx=idx))


def cutout_path(sim, ptype, halo_id):
    return os.path.join(CUTOUT_ROOT, sim_info(sim)["cutout_dir"],
                        f"{ptype}_within_5r200_{halo_id}.hdf5")
