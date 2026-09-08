"""Gas / dark-matter centroid offsets for the BAHAMAS and TNG300 cluster samples.

For every cluster we compute the mass-weighted centre of mass of the gas and of
the dark matter inside r < APERTURE * R_200 of the centre of potential, and
report the three pairwise separations

    |COM_gas - COM_dm|,   |COM_gas - COP|,   |COM_dm - COP|

both in kpc and normalised by R_200. The normalised numbers are the ones the
power-ratio optimisation consumes: they replace the 3D coherence length as the
per-cluster scalar that selects which clusters enter the stack, because a
centroid offset in units of R_200 is far less resolution-sensitive than a
coherence length.

The centre of potential is `CoP` for BAHAMAS and `GroupPos` (the potential
minimum) for TNG, so `COM_x - COP` is just the centred `COM_x`.

Particle sets, units and wrapping follow the existing coherence scripts
(`DS+/coherence_analysis_3D_bahamas.py`, `DS+/coherence_analysis_3D_tng.py`):

  BAHAMAS  per-cluster .npz, positions in comoving Mpc/h in a 400 cMpc/h box,
           masses already in Msun, h taken per-file (it is 0.7, *not* the 0.6774
           used for TNG).
  TNG      per-cluster hdf5 cutouts of everything within 5 R_200, positions in
           comoving kpc/h in a 205000 ckpc/h box, gas Masses in 1e10 Msun/h.
           PartType1 carries no Masses because the DM particles are equal mass,
           so the DM centroid is an unweighted mean.

All selection and centring is done in code units; only the final offsets are
converted (divided by h, as in the coherence scripts, i.e. comoving kpc).

TNG cutouts run to ~1.8 GB per file, so the particle arrays are streamed in
chunks and accumulated rather than held in memory. Peak usage is a few hundred
MB per worker regardless of cluster size.

Usage
-----
    python offsets.py                       # all 8 sims, SLURM_CPUS_PER_TASK procs
    python offsets.py --sims TNG300-3 --nproc 4
    python offsets.py --aperture 0.5        # re-run at a different aperture
"""

import argparse
import os
import sys
import time
import traceback
import multiprocessing as mp

import numpy as np
import h5py


# ------------------------------------------------------------------ #
# Paths and simulation registry
# ------------------------------------------------------------------ #
BAHAMAS_PATH  = '/projects/mccleary_group/habjan.e/TNG/Data'
TNG_PART_PATH = '/projects/mccleary_group/habjan.e/TNG/Data/TNG_data'
TNG_CAT_PATH  = '/home/habjan.e/TNG/Data/TNG_data'
OUT_PATH      = '/projects/mccleary_group/habjan.e/TNG/Data/offset_data'

BAHAMAS_SIMS = ('CDMb', 'SIDM0.1b', 'SIDM0.3b', 'SIDM1b', 'vdSIDMb')
TNG_SIMS     = ('TNG300-1', 'TNG300-2', 'TNG300-3')

# TNG300-1 first: its cutouts are ~20x larger than TNG300-2's and the BAHAMAS
# .npz files take under a second each, so handing the expensive tasks out first
# keeps the pool from finishing with one straggler.
ALL_SIMS = TNG_SIMS + BAHAMAS_SIMS

CLUSTER_IDS = {sim: [f'{i:03d}' for i in range(1, 101)] for sim in BAHAMAS_SIMS}
CLUSTER_IDS.update({sim: [f'{i:d}' for i in range(0, 100)] for sim in TNG_SIMS})

N_CLUSTER = 100

BAHAMAS_BOX_CMPC = 400.0        # cMpc/h
TNG_BOX_CKPC     = 205000.0     # ckpc/h
H_TNG            = 0.6774

# Equal-mass DM particle masses, Msun/h (divided by h below), as used in
# DS+/coherence_analysis_3D_tng.py. Only needed for the M_dm sanity check --
# the centroid itself is unweighted because every particle has the same mass.
TNG_DM_PART_MASS = {'TNG300-1': 4.0e7, 'TNG300-2': 3.2e8, 'TNG300-3': 2.5e9}


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _wrap(dx, box):
    """Minimum-image separation, matching the convention in the coherence scripts."""
    return (dx + 0.5 * box) % box - 0.5 * box


def _com_from_arrays(pos, weights, radius):
    """Mass-weighted centroid of the particles inside `radius`.

    `pos` must already be centred and wrapped, in the same units as `radius`.
    `weights` may be None for an equal-mass (unweighted) centroid.

    Returns (com, n_inside, weight_sum). `com` is NaN if nothing is inside.
    """
    r2 = np.einsum('ij,ij->i', pos, pos)
    inside = r2 < radius * radius
    n = int(inside.sum())

    if n == 0:
        return np.full(3, np.nan), 0, 0.0

    p = pos[inside]

    if weights is None:
        return p.sum(axis=0, dtype=np.float64) / n, n, float(n)

    w = weights[inside].astype(np.float64)
    w_sum = float(w.sum())

    if w_sum <= 0:
        return np.full(3, np.nan), n, w_sum

    return (p * w[:, None]).sum(axis=0, dtype=np.float64) / w_sum, n, w_sum


def _com_streaming(path, group, mass_key, center, box, radius, chunk):
    """Mass-weighted centroid of an hdf5 particle set, read in chunks.

    `mass_key` is None for equal-mass particles, in which case the centroid is
    an unweighted mean. Everything stays in the file's code units.
    """
    wx_sum = np.zeros(3)
    w_sum = 0.0
    n_inside = 0

    with h5py.File(path, 'r') as f:
        coords = f[f'{group}/Coordinates']
        masses = f[f'{group}/{mass_key}'] if mass_key is not None else None
        n_part = coords.shape[0]

        for start in range(0, n_part, chunk):
            stop = min(start + chunk, n_part)

            pos = coords[start:stop].astype(np.float64)
            pos -= center
            pos = _wrap(pos, box)

            w = masses[start:stop] if masses is not None else None
            com_c, n_c, w_c = _com_from_arrays(pos, w, radius)

            if n_c == 0:
                continue

            wx_sum += com_c * w_c
            w_sum += w_c
            n_inside += n_c

    if w_sum <= 0:
        return np.full(3, np.nan), 0, 0.0

    return wx_sum / w_sum, n_inside, w_sum


# ------------------------------------------------------------------ #
# Per-cluster workers, one per simulation suite
# ------------------------------------------------------------------ #
def _bahamas_cluster(sim, cluster_id, aperture):
    """Centroids for one BAHAMAS cluster. Positions are cMpc/h; masses are Msun."""
    data = np.load(f'{BAHAMAS_PATH}/{sim}/GrNm_{cluster_id}.npz')

    cop = data['CoP']
    r200 = float(data['R200'])          # cMpc/h
    h = float(data['h'])                # 0.7 for BAHAMAS, not the TNG 0.6774
    radius = aperture * r200

    dm_pos = _wrap(data['dm_pos'] - cop, BAHAMAS_BOX_CMPC)
    com_dm, n_dm, m_dm = _com_from_arrays(dm_pos, data['dm_mass'], radius)
    del dm_pos

    gas_pos = _wrap(data['gas_pos'] - cop, BAHAMAS_BOX_CMPC)
    com_gas, n_gas, m_gas = _com_from_arrays(gas_pos, data['gas_mass'], radius)
    del gas_pos

    # cMpc/h -> comoving kpc, the convention the coherence scripts use.
    to_kpc = 1.0e3 / h

    return dict(com_dm=com_dm, com_gas=com_gas, r200=r200, to_kpc=to_kpc,
                n_dm=n_dm, n_gas=n_gas, m_dm=m_dm, m_gas=m_gas,
                m200=float(data['M200']))


_GROUPCAT_CACHE = {}


def _tng_groupcat(sim):
    """(GroupPos, Group_R_Crit200, Group_M_Crit200) for the first N_CLUSTER groups.

    Cached per process, and read straight from the local hdf5 caches so the
    workers never touch the TNG API, which the coherence scripts do call and
    which can 404 or time out. M_200 is converted to Msun for the sanity check.
    """
    if sim not in _GROUPCAT_CACHE:
        with h5py.File(f'{TNG_CAT_PATH}/{sim}_GroupPos.hdf5', 'r') as f:
            pos = f['Group/GroupPos'][:N_CLUSTER].astype(np.float64)
        with h5py.File(f'{TNG_CAT_PATH}/{sim}_Group_R_Crit200.hdf5', 'r') as f:
            r200 = f['Group/Group_R_Crit200'][:N_CLUSTER].astype(np.float64)
        with h5py.File(f'{TNG_CAT_PATH}/{sim}_Group_M_Crit200.hdf5', 'r') as f:
            m200 = f['Group/Group_M_Crit200'][:N_CLUSTER].astype(np.float64)
        _GROUPCAT_CACHE[sim] = (pos, r200, m200 * 1.0e10 / H_TNG)

    return _GROUPCAT_CACHE[sim]


def _tng_cluster(sim, cluster_id, aperture, chunk):
    """Centroids for one TNG300 cluster. Positions are ckpc/h; gas Masses 1e10 Msun/h."""
    sim_int = sim.rsplit('-', 1)[-1]
    group_pos, group_r200, group_m200 = _tng_groupcat(sim)

    ind = int(cluster_id)
    center = group_pos[ind]
    r200 = float(group_r200[ind])       # ckpc/h
    radius = aperture * r200

    dm_file = f'{TNG_PART_PATH}/5r200_data-{sim_int}/dm_within_5r200_{cluster_id}.hdf5'
    gas_file = f'{TNG_PART_PATH}/5r200_data-{sim_int}/gas_within_5r200_{cluster_id}.hdf5'

    # PartType1 has no Masses dataset: equal-mass particles, so unweighted.
    com_dm, n_dm, _ = _com_streaming(dm_file, 'PartType1', None,
                                     center, TNG_BOX_CKPC, radius, chunk)
    com_gas, n_gas, w_gas = _com_streaming(gas_file, 'PartType0', 'Masses',
                                           center, TNG_BOX_CKPC, radius, chunk)

    return dict(com_dm=com_dm, com_gas=com_gas, r200=r200, to_kpc=1.0 / H_TNG,
                n_dm=n_dm, n_gas=n_gas,
                m_dm=n_dm * TNG_DM_PART_MASS[sim] / H_TNG,
                m_gas=w_gas * 1.0e10 / H_TNG,
                m200=float(group_m200[ind]))


def _worker(task):
    """Compute one cluster's offsets. Never raises: failures come back as a record."""
    sim, row, cluster_id, aperture, chunk = task
    t0 = time.time()

    try:
        if sim in BAHAMAS_SIMS:
            res = _bahamas_cluster(sim, cluster_id, aperture)
        else:
            res = _tng_cluster(sim, cluster_id, aperture, chunk)
    except Exception:
        return dict(sim=sim, row=row, cluster_id=cluster_id, ok=False,
                    error=traceback.format_exc(), seconds=time.time() - t0)

    com_dm, com_gas = res['com_dm'], res['com_gas']
    r200, to_kpc = res['r200'], res['to_kpc']

    # Distances in code units first; COP is the origin, so |COM - COP| = |COM|.
    d_gas_dm = float(np.linalg.norm(com_gas - com_dm))
    d_gas_cop = float(np.linalg.norm(com_gas))
    d_dm_cop = float(np.linalg.norm(com_dm))

    res.update(sim=sim, row=row, cluster_id=cluster_id, ok=True, error='',
               seconds=time.time() - t0,
               d_gas_dm=d_gas_dm / r200,
               d_gas_cop=d_gas_cop / r200,
               d_dm_cop=d_dm_cop / r200,
               d_gas_dm_kpc=d_gas_dm * to_kpc,
               d_gas_cop_kpc=d_gas_cop * to_kpc,
               d_dm_cop_kpc=d_dm_cop * to_kpc,
               com_gas_r200=com_gas / r200,
               com_dm_r200=com_dm / r200,
               r200_kpc=r200 * to_kpc)

    return res


# ------------------------------------------------------------------ #
# Driver
# ------------------------------------------------------------------ #
def _empty_arrays():
    scalar_keys = ('d_gas_dm', 'd_gas_cop', 'd_dm_cop',
                   'd_gas_dm_kpc', 'd_gas_cop_kpc', 'd_dm_cop_kpc',
                   'r200_kpc', 'm_dm', 'm_gas', 'm200')

    out = {k: np.full(N_CLUSTER, np.nan) for k in scalar_keys}
    out['com_gas_r200'] = np.full((N_CLUSTER, 3), np.nan)
    out['com_dm_r200'] = np.full((N_CLUSTER, 3), np.nan)
    out['n_dm'] = np.zeros(N_CLUSTER, dtype=np.int64)
    out['n_gas'] = np.zeros(N_CLUSTER, dtype=np.int64)

    return out


def main():
    parser = argparse.ArgumentParser(
        description='Gas / DM centroid offsets for the BAHAMAS and TNG300 clusters')
    parser.add_argument('--sims', nargs='+', default=list(ALL_SIMS),
                        help='simulations to process (default: all eight)')
    parser.add_argument('--aperture', type=float, default=1.0,
                        help='centroid aperture in units of R_200 (default: 1.0)')
    parser.add_argument('--nproc', type=int,
                        default=int(os.environ.get('SLURM_CPUS_PER_TASK',
                                                   os.cpu_count() or 1)),
                        help='worker processes (default: SLURM_CPUS_PER_TASK)')
    parser.add_argument('--chunk', type=int, default=4_000_000,
                        help='particles per hdf5 read for the TNG cutouts')
    parser.add_argument('--outdir', default=OUT_PATH)
    args = parser.parse_args()

    unknown = [s for s in args.sims if s not in ALL_SIMS]
    if unknown:
        parser.error(f'unknown simulation(s): {unknown}')

    os.makedirs(args.outdir, exist_ok=True)

    tasks = [
        (sim, row, cid, args.aperture, args.chunk)
        for sim in args.sims
        for row, cid in enumerate(CLUSTER_IDS[sim])
    ]

    print(f'aperture = {args.aperture} R_200')
    print(f'{len(tasks)} clusters across {len(args.sims)} simulations '
          f'on {args.nproc} processes', flush=True)

    results = {sim: _empty_arrays() for sim in args.sims}
    n_fail = 0
    t_start = time.time()

    # 'fork' keeps the group-catalogue cache logic simple and costs nothing here,
    # since the parent holds no large arrays.
    ctx = mp.get_context('fork')

    with ctx.Pool(processes=args.nproc) as pool:
        for done, res in enumerate(pool.imap_unordered(_worker, tasks, chunksize=1), 1):
            sim, row = res['sim'], res['row']

            if not res['ok']:
                n_fail += 1
                print(f'[{done}/{len(tasks)}] FAILED {sim} {res["cluster_id"]}\n'
                      f'{res["error"]}', flush=True)
                continue

            store = results[sim]
            for key in ('d_gas_dm', 'd_gas_cop', 'd_dm_cop',
                        'd_gas_dm_kpc', 'd_gas_cop_kpc', 'd_dm_cop_kpc',
                        'r200_kpc', 'm_dm', 'm_gas', 'm200',
                        'n_dm', 'n_gas'):
                store[key][row] = res[key]

            store['com_gas_r200'][row] = res['com_gas_r200']
            store['com_dm_r200'][row] = res['com_dm_r200']

            if done % 25 == 0 or done == len(tasks):
                print(f'[{done}/{len(tasks)}] {time.time() - t_start:7.1f}s elapsed '
                      f'(last: {sim} {res["cluster_id"]} in {res["seconds"]:.1f}s, '
                      f'|COM_gas-COM_dm|/R200 = {res["d_gas_dm"]:.4f})', flush=True)

    for sim in args.sims:
        store = results[sim]
        out_file = f'{args.outdir}/{sim}.npz'

        np.savez(
            out_file,
            aperture=np.float64(args.aperture),
            cluster_ids=np.array(CLUSTER_IDS[sim]),
            **store,
        )

        # (M_dm + M_gas) / M_200 inside R_200 should sit a few per cent below 1 --
        # the shortfall is the stars and black holes, which we deliberately leave
        # out of both centroids. Anything far from unity means the centring or a
        # unit conversion is wrong. The deficit grows towards coarser resolution
        # (~0.4% for TNG300-2, ~4% for TNG300-3) because the low-resolution runs
        # overcool and lock more mass into stars.
        finite = np.isfinite(store['d_gas_dm'])
        m_frac = (store['m_dm'] + store['m_gas']) / store['m200']

        print(f'\n{sim}: wrote {out_file}   {finite.sum()}/{N_CLUSTER} clusters')
        print(f'  median |COM_gas-COM_dm|/R200 = {np.nanmedian(store["d_gas_dm"]):.4f}   '
              f'|COM_gas-COP|/R200 = {np.nanmedian(store["d_gas_cop"]):.4f}   '
              f'|COM_dm-COP|/R200 = {np.nanmedian(store["d_dm_cop"]):.4f}')
        print(f'  median M_200 = {np.nanmedian(store["m200"]):.3e} Msun   '
              f'median (M_dm+M_gas)(<{args.aperture} R200) / M_200 = '
              f'{np.nanmedian(m_frac):.3f}')

    print(f'\ntotal {time.time() - t_start:.1f}s, {n_fail} failure(s)', flush=True)

    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
