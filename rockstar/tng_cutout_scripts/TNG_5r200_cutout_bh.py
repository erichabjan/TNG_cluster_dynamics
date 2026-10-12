#!/usr/bin/env python3
"""
Black-hole (PartType5) cutout within rmult * R200 of a TNG halo, read chunk by
chunk from the tng_chunking.py cache, with the fields in
tng_cutout_config.FIELDS['bh']:
    Coordinates, Velocities, Masses, ParticleIDs.

Masses is the dynamical particle mass (what enters the density cube and
lensing), not the subgrid BH_Mass.

Writes <cutout_dir>/bh_within_5r200_<halo>.hdf5 (see tng_cutout_config).
"""
import tng_cutout_common as tc

if __name__ == "__main__":
    tc.cutout_main("bh", __doc__)
