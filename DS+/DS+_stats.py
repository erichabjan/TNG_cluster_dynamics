#!/usr/bin/env python3
"""
DS+_stats.py

Computes DS+ performance statistics across all 100 TNG clusters and
1,000 random projections per cluster.

Statistics computed:
  1. ARI  – Adjusted Rand Index (global, per cluster × projection)
  2. Fragmentation fraction – per ROCKSTAR subhalo, per projection
  3. Merging fraction – per DS+ subhalo, per projection
  4. Completeness & Purity – per cluster × projection (from virial CSV)

Outputs (saved to SAVE_DIR):
  - ari_stats.csv
  - fragmentation_stats.csv
  - merging_stats.csv
  - completeness_purity_stats.csv
  - cluster_mass_coherence.npy   (100 × 1000 arrays for mass and coherence)
"""

import numpy as np
import pandas as pd
import sys
import os

# ── Setup paths and imports ──────────────────────────────────────────────
dirc_path = '/home/habjan.e/'
sys.path.append(dirc_path + "TNG/TNG_cluster_dynamics")
import TNG_DA

# ── Constants ────────────────────────────────────────────────────────────
N_CLUSTERS = 100
N_PROJ     = 1000

# ── Data paths ───────────────────────────────────────────────────────────
DSP_DATA    = '/projects/mccleary_group/habjan.e/TNG/Data/data_DS+_virial_results/'
COH_DATA    = '/projects/mccleary_group/habjan.e/TNG/Data/coherence_data/TNG/'
COH_3D_DATA = '/projects/mccleary_group/habjan.e/TNG/Data/coherence_data_3D/TNG/'
RS_DATA     = '/projects/mccleary_group/habjan.e/TNG/Data/rockstar_output/tng_rockstar_output/'
SAVE_DIR    = '/projects/mccleary_group/habjan.e/TNG/Data/data_DS+_stats/'
os.makedirs(SAVE_DIR, exist_ok=True)

RS_COLS = [
    "id", "parent_id", "pos_0", "pos_1", "pos_2", "pos_3", "pos_4", "pos_5",
    "num_p", "mass_grav_est", "mgrav_bound", "vrms", "vmax", "rvmax", "rs",
    "kin_to_pot", "Xoff",
]

# ── Load M_200 for all 100 clusters ─────────────────────────────────────
sys.path.append('/home/habjan.e/TNG/Codes/TNG_workshop')
import iapi_TNG as iapi

baseUrl = 'http://www.tng-project.org/api/'
sim = 'TNG300-1'
TNG_data_path = '/home/habjan.e/TNG/Data/'
simdata = iapi.get(baseUrl + sim)
h = simdata['hubble']

tng_m200_raw = iapi.getHaloField(
    field='Group_M_Crit200', simulation=sim, snapshot=99,
    fileName=TNG_data_path + 'TNG_data/' + sim + '_Group_M_Crit200',
    rewriteFile=0,
)
tng_m200 = np.log10(tng_m200_raw[:N_CLUSTERS] * 1e10 / h)   # log10(M_sun)

# ── Pre-allocate fixed-size arrays (100 clusters × 1000 projections) ────
ari_all          = np.full((N_CLUSTERS, N_PROJ), np.nan)
completeness_all = np.full((N_CLUSTERS, N_PROJ), np.nan)
purity_all       = np.full((N_CLUSTERS, N_PROJ), np.nan)
cluster_mass     = np.zeros(N_CLUSTERS)
coh_3d_arr       = np.zeros(N_CLUSTERS)
coh_3d_err_arr   = np.zeros(N_CLUSTERS)
triax_arr        = np.zeros(N_CLUSTERS)
coherence_full     = np.zeros((N_CLUSTERS, N_PROJ))   # per-projection coherence
coherence_err_full = np.zeros((N_CLUSTERS, N_PROJ))   # per-projection 2D error

# ── Variable-size accumulators: fragmentation ────────────────────────────
frag_cl_list      = []
frag_proj_list    = []
frag_rsid_list    = []
frag_val_list     = []
frag_rsmass_list  = []

# ── Variable-size accumulators: merging ──────────────────────────────────
merge_cl_list     = []
merge_proj_list   = []
merge_dsid_list   = []
merge_val_list    = []
merge_com_list    = []

# ═════════════════════════════════════════════════════════════════════════
#  Main loop over clusters
# ═════════════════════════════════════════════════════════════════════════
for cl_id in range(N_CLUSTERS):
    print(f"[{cl_id + 1:3d}/{N_CLUSTERS}] Cluster {cl_id}", flush=True)

    # ── 1. Load pre-computed DS+ group assignments ───────────────────────
    dsp_out_2 = np.load(
        DSP_DATA + f'DS+_array_2_{cl_id}.npy', allow_pickle=True
    )
    dsp_groups = dsp_out_2[:, :, 8]          # shape (1000, N_gal)

    # ── 2. Load TNG cluster properties ───────────────────────────────────
    pos, vel, group, sub_masses, subhalo_type, h, halo_mass = \
        TNG_DA.get_cluster_props(cl_id)
    N_gal = len(group)
    gal_r = np.sqrt(np.sum(pos**2, axis=1))  # 3-D radial distance

    # ── 3. Coherence data ────────────────────────────────────────────────
    coh_len = np.load(COH_DATA + f'coherence_length_{cl_id}.npy')
    coh_err = np.load(COH_DATA + f'coherence_length_err_{cl_id}.npy')
    coherence_full[cl_id]     = coh_len
    coherence_err_full[cl_id] = coh_err
    cluster_mass[cl_id]       = halo_mass

    coh_3d_arr[cl_id]     = np.load(COH_3D_DATA + f'coherence_length_{cl_id}.npy')[0]
    coh_3d_err_arr[cl_id] = np.load(COH_3D_DATA + f'coherence_length_err_{cl_id}.npy')[0]

    # ── 4. Virial CSV → completeness, purity, triaxiality ───────────────
    vdf = pd.read_csv(DSP_DATA + f'DS+_Virial_df_{cl_id}.csv')
    completeness_all[cl_id] = vdf['Completeness'].values
    purity_all[cl_id]       = vdf['Purity'].values
    triax_arr[cl_id]        = vdf['Triaxiality'].values[0]

    # ── 5. ROCKSTAR catalog → mass look-up table ────────────────────────
    rs_df = pd.read_csv(
        RS_DATA + f'rockstar_subhalos_{cl_id}.list',
        sep=r"\s+", comment="#", names=RS_COLS, engine="python",
    )
    rs_ids_cat = np.array(rs_df['id'], dtype=int)
    max_rsid   = rs_ids_cat.max() + 1 if len(rs_ids_cat) > 0 else 1
    rs_mass_lut = np.zeros(max_rsid)
    rs_mass_lut[rs_ids_cat] = np.array(rs_df['mgrav_bound'])

    # ── 6. Evaluate DS+ runs (ARI, fragmentation, merging) ──────────────
    results = TNG_DA.evaluate_dsplus_runs(
        dsp_groups, group,
        ari_include_rs_bg=True,   ari_include_ds_bg=True,
        frag_include_rs_bg=False, frag_include_ds_bg=False,
        merge_include_rs_bg=False, merge_include_ds_bg=False,
    )
    ari_all[cl_id] = results['ari']

    # Pre-compute integer ROCKSTAR labels (needed for merging COM)
    rs_int = TNG_DA._as_int_labels_with_nan_bg(group, bg_label=0)

    # ── 7. Per-projection: collect fragmentation & merging ───────────────
    for r in range(N_PROJ):

        # ── Fragmentation ────────────────────────────────────────────────
        fv   = results['frag'][r]
        fids = results['frag_rs_ids'][r]
        nf   = len(fv)
        if nf > 0:
            frag_cl_list.append(np.full(nf, cl_id, dtype=np.int32))
            frag_proj_list.append(np.full(nf, r, dtype=np.int32))
            frag_rsid_list.append(fids.astype(np.int64))
            frag_val_list.append(fv)
            # Look up ROCKSTAR mass for each subhalo
            safe = np.clip(fids.astype(int), 0, max_rsid - 1)
            frag_rsmass_list.append(
                np.where(fids.astype(int) < max_rsid,
                         rs_mass_lut[safe], np.nan)
            )

        # ── Merging + mass-weighted radial COM ───────────────────────────
        mv   = results['merge'][r]
        mids = results['merge_ds_ids'][r]
        nm   = len(mv)
        if nm > 0:
            merge_cl_list.append(np.full(nm, cl_id, dtype=np.int32))
            merge_proj_list.append(np.full(nm, r, dtype=np.int32))
            merge_dsid_list.append(mids.astype(np.int64))
            merge_val_list.append(mv)

            # Reconstruct DS+ labels with the same background policy used
            # in evaluate_dsplus_runs so COM indices are consistent.
            ds   = TNG_DA._map_ds_background(
                dsp_groups[r], ds_bg_label=-1, unified_bg=0
            )
            keep = (rs_int != 0) & (ds != 0)      # exclude both backgrounds
            kidx = np.where(keep)[0]
            ds_k = ds[keep]

            com = np.full(nm, np.nan)
            for g in range(nm):
                sel = kidx[ds_k == mids[g]]
                if len(sel) > 0:
                    w  = sub_masses[sel]
                    tw = w.sum()
                    com[g] = ((w * gal_r[sel]).sum() / tw
                              if tw > 0 else gal_r[sel].mean())
            merge_com_list.append(com)

# ═════════════════════════════════════════════════════════════════════════
#  Build and save output CSV files
# ═════════════════════════════════════════════════════════════════════════
print("Saving results …", flush=True)

# Shared index columns for the 100 × 1000 tables
ci = np.repeat(np.arange(N_CLUSTERS), N_PROJ)
pi = np.tile(np.arange(N_PROJ), N_CLUSTERS)

# ── ARI stats ────────────────────────────────────────────────────────────
pd.DataFrame({
    'cluster_id':       ci,
    'projection_idx':   pi,
    'ari':              ari_all.ravel(),
    'cluster_mass':     np.repeat(cluster_mass,   N_PROJ),
    'm200':             np.repeat(tng_m200,       N_PROJ),
    'coherence_2d':     coherence_full.ravel(),
    'coherence_2d_err': coherence_err_full.ravel(),
    'coherence_3d':     np.repeat(coh_3d_arr,     N_PROJ),
    'coherence_3d_err': np.repeat(coh_3d_err_arr, N_PROJ),
    'triaxiality':      np.repeat(triax_arr,      N_PROJ),
}).to_csv(SAVE_DIR + 'ari_stats.csv', index=False)
print("  ✓ ari_stats.csv")

# ── Completeness & Purity stats ─────────────────────────────────────────
pd.DataFrame({
    'cluster_id':       ci,
    'projection_idx':   pi,
    'completeness':     completeness_all.ravel(),
    'purity':           purity_all.ravel(),
    'cluster_mass':     np.repeat(cluster_mass,   N_PROJ),
    'm200':             np.repeat(tng_m200,       N_PROJ),
    'coherence_2d':     coherence_full.ravel(),
    'coherence_2d_err': coherence_err_full.ravel(),
    'coherence_3d':     np.repeat(coh_3d_arr,     N_PROJ),
    'coherence_3d_err': np.repeat(coh_3d_err_arr, N_PROJ),
    'triaxiality':      np.repeat(triax_arr,      N_PROJ),
}).to_csv(SAVE_DIR + 'completeness_purity_stats.csv', index=False)
print("  ✓ completeness_purity_stats.csv")

# ── Fragmentation stats ─────────────────────────────────────────────────
if frag_cl_list:
    fc = np.concatenate(frag_cl_list)
    fp = np.concatenate(frag_proj_list)
    pd.DataFrame({
        'cluster_id':          fc,
        'projection_idx':      fp,
        'rockstar_subhalo_id': np.concatenate(frag_rsid_list),
        'fragmentation':       np.concatenate(frag_val_list),
        'rockstar_mass':       np.concatenate(frag_rsmass_list),
        'cluster_mass':        cluster_mass[fc],
        'm200':                tng_m200[fc],
        'coherence_2d':        coherence_full[fc, fp],
        'coherence_2d_err':    coherence_err_full[fc, fp],
        'coherence_3d':        coh_3d_arr[fc],
        'coherence_3d_err':    coh_3d_err_arr[fc],
        'triaxiality':         triax_arr[fc],
    }).to_csv(SAVE_DIR + 'fragmentation_stats.csv', index=False)
    print("  ✓ fragmentation_stats.csv")

# ── Merging stats ────────────────────────────────────────────────────────
if merge_cl_list:
    mc = np.concatenate(merge_cl_list)
    mp = np.concatenate(merge_proj_list)
    pd.DataFrame({
        'cluster_id':        mc,
        'projection_idx':    mp,
        'ds_subhalo_id':     np.concatenate(merge_dsid_list),
        'merging':           np.concatenate(merge_val_list),
        'substructure_com_r': np.concatenate(merge_com_list),
        'cluster_mass':      cluster_mass[mc],
        'm200':              tng_m200[mc],
        'coherence_2d':      coherence_full[mc, mp],
        'coherence_2d_err':  coherence_err_full[mc, mp],
        'coherence_3d':      coh_3d_arr[mc],
        'coherence_3d_err':  coh_3d_err_arr[mc],
        'triaxiality':       triax_arr[mc],
    }).to_csv(SAVE_DIR + 'merging_stats.csv', index=False)
    print("  ✓ merging_stats.csv")

# ── 100 × 1000 companion arrays (cluster mass & per-projection coherence)
cluster_mass_2d = np.repeat(cluster_mass[:, np.newaxis], N_PROJ, axis=1)
np.savez(
    SAVE_DIR + 'cluster_mass_coherence.npz',
    cluster_mass=cluster_mass_2d,        # (100, 1000)
    coherence=coherence_full,            # (100, 1000)
)
print("  ✓ cluster_mass_coherence.npz")

print("All done!")
