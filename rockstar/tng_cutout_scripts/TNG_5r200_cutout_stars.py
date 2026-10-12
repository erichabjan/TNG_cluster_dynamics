#!/usr/bin/env python3
"""
Star-particle (PartType4) cutout within rmult * R200 of a TNG halo, read chunk
by chunk from the tng_chunking.py cache, with the fields in
tng_cutout_config.FIELDS['stars']:
    Coordinates, Velocities, Masses, ParticleIDs, GFM_StellarFormationTime.

PartType4 also holds wind particles (GFM_StellarFormationTime <= 0), which are
gas in transit, not stars. They are kept here, with their formation time, and
dropped downstream when the dataset is built.

Writes <cutout_dir>/stars_within_5r200_<halo>.hdf5 (see tng_cutout_config).
"""
import tng_cutout_common as tc

if __name__ == "__main__":
    tc.cutout_main("stars", __doc__)
