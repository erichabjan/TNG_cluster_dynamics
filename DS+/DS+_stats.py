#!/usr/bin/env python3
"""
DS+_stats.py

Computes DS+ performance statistics across all 100 TNG clusters and
1,000 random projections per cluster.

All statistics are evaluated under three substructure-size cut policies
(no_cuts / min3 / min3_maxsqrtN) with backgrounds excluded:
  1. ARI            – Adjusted Rand Index, per cluster × projection
  2. Completeness   – per cluster × projection
  3. Purity         – per cluster × projection
  4. Fragmentation  – per ROCKSTAR subhalo, per projection
  5. Merging        – per DS+ subhalo, per projection

Outputs (saved to SAVE_DIR):
  - dsp_cases_stats.csv          (ARI / C / P for the three cut policies)
  - fragmentation_stats.csv      (per-case fragmentation per RS subhalo, wide)
  - merging_stats.csv            (per-case merging per DS+ subhalo, wide)
  - cluster_mass_coherence.npz   (100 × 1000 arrays for mass and coherence)
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

# Substructure-size cut policies. Each entry: (min_size, max_size).
# max_size is exclusive (group kept if size < max_size). None means
# "no bound on that side".
CASE_PARAMS = {
    'no_cuts':       (None, None),    # keep every detected group
    'min3':          (3,    None),    # require ≥ 3 members
    'min3_maxsqrtN': (3,    'sqrtN'), # require 3 ≤ size < sqrt(N_gal)
}
CASE_NAMES = list(CASE_PARAMS.keys())

# ── Helpers for the size-cut cases ───────────────────────────────────────
def _apply_size_cut_int(labels, *, bg_label=0, min_size=None, max_size=None):
    """Return a copy of `labels` with every non-background group whose size
    falls outside [min_size, max_size) reassigned to `bg_label`."""
    out = labels.copy()
    valid = out != bg_label
    if not valid.any():
        return out
    uniq, inv = np.unique(out[valid], return_inverse=True)
    counts = np.bincount(inv)
    fail = np.zeros(uniq.size, dtype=bool)
    if min_size is not None:
        fail |= counts < min_size
    if max_size is not None:
        fail |= counts >= max_size
    if fail.any():
        out[np.isin(out, uniq[fail])] = bg_label
    return out


def _binary_substructure(labels, *, bg_label=0, min_size=None, max_size=None):
    """Return a 0/1 array marking galaxies that lie in a non-background
    group whose size satisfies the cut."""
    out = np.zeros(labels.shape, dtype=np.int8)
    valid = labels != bg_label
    if not valid.any():
        return out
    uniq, inv = np.unique(labels[valid], return_inverse=True)
    counts = np.bincount(inv)
    keep = np.ones(uniq.size, dtype=bool)
    if min_size is not None:
        keep &= counts >= min_size
    if max_size is not None:
        keep &= counts < max_size
    out[valid] = keep[inv].astype(np.int8)
    return out

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
cluster_mass       = np.zeros(N_CLUSTERS)
coh_3d_arr         = np.zeros(N_CLUSTERS)
coh_3d_err_arr     = np.zeros(N_CLUSTERS)
triax_arr          = np.zeros(N_CLUSTERS)
coherence_full     = np.zeros((N_CLUSTERS, N_PROJ))   # per-projection coherence
coherence_err_full = np.zeros((N_CLUSTERS, N_PROJ))   # per-projection 2D error

# Per-case ARI / completeness / purity (cluster × projection)
ari_cases = {c: np.full((N_CLUSTERS, N_PROJ), np.nan) for c in CASE_NAMES}
cmp_cases = {c: np.full((N_CLUSTERS, N_PROJ), np.nan) for c in CASE_NAMES}
pur_cases = {c: np.full((N_CLUSTERS, N_PROJ), np.nan) for c in CASE_NAMES}

# ── Variable-size wide accumulators ──────────────────────────────────────
# Fragmentation: one row per (cluster, projection, ROCKSTAR subhalo)
frag_cl_list      = []
frag_proj_list    = []
frag_rsid_list    = []
frag_case_lists   = {c: [] for c in CASE_NAMES}
frag_rsmass_list  = []

# Merging: one row per (cluster, projection, DS+ subhalo)
merge_cl_list     = []
merge_proj_list   = []
merge_dsid_list   = []
merge_case_lists  = {c: [] for c in CASE_NAMES}
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

    # ── 4. Virial CSV → triaxiality (scalar per cluster) ─────────────────
    vdf = pd.read_csv(DSP_DATA + f'DS+_Virial_df_{cl_id}.csv')
    triax_arr[cl_id] = vdf['Triaxiality'].values[0]

    # ── 5. ROCKSTAR catalog → mass look-up table ────────────────────────
    rs_df = pd.read_csv(
        RS_DATA + f'rockstar_subhalos_{cl_id}.list',
        sep=r"\s+", comment="#", names=RS_COLS, engine="python",
    )
    rs_ids_cat = np.array(rs_df['id'], dtype=int)
    max_rsid   = rs_ids_cat.max() + 1 if len(rs_ids_cat) > 0 else 1
    rs_mass_lut = np.zeros(max_rsid)
    rs_mass_lut[rs_ids_cat] = np.array(rs_df['mgrav_bound'])

    # ── 6. Pre-compute integer labels (rs fixed; dsp per projection) ─────
    rs_int = TNG_DA._as_int_labels_with_nan_bg(group, bg_label=0)
    dsp_int = np.empty(dsp_groups.shape, dtype=np.int64)
    for r in range(N_PROJ):
        dsp_int[r] = TNG_DA._map_ds_background(
            dsp_groups[r], ds_bg_label=-1, unified_bg=0,
        )

    # ── 7. Per-case ARI / C / P / fragmentation / merging ────────────────
    # All five stats use backgrounds excluded on both sides.
    sqrtN = int(np.sqrt(N_gal))
    case_frag_per_proj  = {c: {} for c in CASE_NAMES}   # case → {r: {rs_id: frag}}
    case_merge_per_proj = {c: {} for c in CASE_NAMES}   # case → {r: {ds_id: merge}}

    for case_name, (min_sz, max_sz_spec) in CASE_PARAMS.items():
        max_sz = sqrtN if max_sz_spec == 'sqrtN' else max_sz_spec

        rs_case = _apply_size_cut_int(
            rs_int, bg_label=0, min_size=min_sz, max_size=max_sz,
        )
        tng_bin = _binary_substructure(
            rs_int, bg_label=0, min_size=min_sz, max_size=max_sz,
        )
        n_real = int(tng_bin.sum())

        for r in range(N_PROJ):
            dsp_case_row = _apply_size_cut_int(
                dsp_int[r], bg_label=0, min_size=min_sz, max_size=max_sz,
            )
            dsp_bin = _binary_substructure(
                dsp_int[r], bg_label=0, min_size=min_sz, max_size=max_sz,
            )

            ari_cases[case_name][cl_id, r] = TNG_DA.adjusted_rand_index(
                rs_case, dsp_case_row,
                include_rs_bg=False, include_ds_bg=False, bg_label=0,
            )

            n_dsp  = int(dsp_bin.sum())
            n_both = int((tng_bin & dsp_bin).sum())
            cmp_cases[case_name][cl_id, r] = (
                n_both / n_real if n_real > 0 else np.nan
            )
            pur_cases[case_name][cl_id, r] = (
                n_both / n_dsp if n_dsp > 0 else np.nan
            )

            f_frag, frag_rs_ids = TNG_DA.fragmentation_per_rockstar_subhalo(
                rs_case, dsp_case_row,
                include_rs_bg=False, include_ds_bg=False, bg_label=0,
            )
            case_frag_per_proj[case_name][r] = dict(
                zip(frag_rs_ids.tolist(), f_frag.tolist())
            )

            f_merge, merge_ds_ids = TNG_DA.merging_per_ds_group(
                rs_case, dsp_case_row,
                include_rs_bg=False, include_ds_bg=False, bg_label=0,
            )
            case_merge_per_proj[case_name][r] = dict(
                zip(merge_ds_ids.tolist(), f_merge.tolist())
            )

    # ── 8. Assemble wide-format frag/merge rows per projection ───────────
    # The no_cuts case is the superset (cuts are nested), so its subhalo
    # ids drive the row set; stricter cases produce NaN where they drop ids.
    for r in range(N_PROJ):
        # ── Fragmentation ────────────────────────────────────────────────
        no_cuts_frag = case_frag_per_proj['no_cuts'].get(r, {})
        rs_ids_sorted = sorted(no_cuts_frag.keys())
        if rs_ids_sorted:
            rs_ids_arr = np.array(rs_ids_sorted, dtype=np.int64)
            nf = rs_ids_arr.size
            frag_cl_list.append(np.full(nf, cl_id, dtype=np.int32))
            frag_proj_list.append(np.full(nf, r, dtype=np.int32))
            frag_rsid_list.append(rs_ids_arr)
            for case_name in CASE_NAMES:
                d = case_frag_per_proj[case_name].get(r, {})
                vals = np.array(
                    [d.get(int(rid), np.nan) for rid in rs_ids_arr],
                    dtype=float,
                )
                frag_case_lists[case_name].append(vals)
            safe = np.clip(rs_ids_arr.astype(int), 0, max_rsid - 1)
            frag_rsmass_list.append(
                np.where(rs_ids_arr.astype(int) < max_rsid,
                         rs_mass_lut[safe], np.nan)
            )

        # ── Merging ──────────────────────────────────────────────────────
        no_cuts_merge = case_merge_per_proj['no_cuts'].get(r, {})
        ds_ids_sorted = sorted(no_cuts_merge.keys())
        if ds_ids_sorted:
            ds_ids_arr = np.array(ds_ids_sorted, dtype=np.int64)
            nm = ds_ids_arr.size
            merge_cl_list.append(np.full(nm, cl_id, dtype=np.int32))
            merge_proj_list.append(np.full(nm, r, dtype=np.int32))
            merge_dsid_list.append(ds_ids_arr)
            for case_name in CASE_NAMES:
                d = case_merge_per_proj[case_name].get(r, {})
                vals = np.array(
                    [d.get(int(did), np.nan) for did in ds_ids_arr],
                    dtype=float,
                )
                merge_case_lists[case_name].append(vals)

            # Mass-weighted radial COM (case-independent: derived from raw
            # labels with both backgrounds excluded — same mask used to
            # define the no_cuts ds_id set).
            ds = TNG_DA._map_ds_background(
                dsp_groups[r], ds_bg_label=-1, unified_bg=0
            )
            keep = (rs_int != 0) & (ds != 0)
            kidx = np.where(keep)[0]
            ds_k = ds[keep]
            com = np.full(nm, np.nan)
            for g in range(nm):
                sel = kidx[ds_k == int(ds_ids_arr[g])]
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

# ── ARI / completeness / purity per cut-policy case ─────────────────────
case_cols = {
    'cluster_id':       ci,
    'projection_idx':   pi,
}
for c in CASE_NAMES:
    case_cols[f'ari_{c}']          = ari_cases[c].ravel()
    case_cols[f'completeness_{c}'] = cmp_cases[c].ravel()
    case_cols[f'purity_{c}']       = pur_cases[c].ravel()
case_cols.update({
    'cluster_mass':     np.repeat(cluster_mass,   N_PROJ),
    'm200':             np.repeat(tng_m200,       N_PROJ),
    'coherence_2d':     coherence_full.ravel(),
    'coherence_2d_err': coherence_err_full.ravel(),
    'coherence_3d':     np.repeat(coh_3d_arr,     N_PROJ),
    'coherence_3d_err': np.repeat(coh_3d_err_arr, N_PROJ),
    'triaxiality':      np.repeat(triax_arr,      N_PROJ),
})
pd.DataFrame(case_cols).to_csv(SAVE_DIR + 'dsp_cases_stats.csv', index=False)
print("  ✓ dsp_cases_stats.csv")

# ── Fragmentation stats (wide: one column per cut-policy case) ──────────
if frag_cl_list:
    fc = np.concatenate(frag_cl_list)
    fp = np.concatenate(frag_proj_list)
    frag_df = {
        'cluster_id':          fc,
        'projection_idx':      fp,
        'rockstar_subhalo_id': np.concatenate(frag_rsid_list),
    }
    for c in CASE_NAMES:
        frag_df[f'fragmentation_{c}'] = np.concatenate(frag_case_lists[c])
    frag_df.update({
        'rockstar_mass':    np.concatenate(frag_rsmass_list),
        'cluster_mass':     cluster_mass[fc],
        'm200':             tng_m200[fc],
        'coherence_2d':     coherence_full[fc, fp],
        'coherence_2d_err': coherence_err_full[fc, fp],
        'coherence_3d':     coh_3d_arr[fc],
        'coherence_3d_err': coh_3d_err_arr[fc],
        'triaxiality':      triax_arr[fc],
    })
    pd.DataFrame(frag_df).to_csv(
        SAVE_DIR + 'fragmentation_stats.csv', index=False
    )
    print("  ✓ fragmentation_stats.csv")

# ── Merging stats (wide: one column per cut-policy case) ────────────────
if merge_cl_list:
    mc = np.concatenate(merge_cl_list)
    mp = np.concatenate(merge_proj_list)
    merge_df = {
        'cluster_id':     mc,
        'projection_idx': mp,
        'ds_subhalo_id':  np.concatenate(merge_dsid_list),
    }
    for c in CASE_NAMES:
        merge_df[f'merging_{c}'] = np.concatenate(merge_case_lists[c])
    merge_df.update({
        'substructure_com_r': np.concatenate(merge_com_list),
        'cluster_mass':       cluster_mass[mc],
        'm200':               tng_m200[mc],
        'coherence_2d':       coherence_full[mc, mp],
        'coherence_2d_err':   coherence_err_full[mc, mp],
        'coherence_3d':       coh_3d_arr[mc],
        'coherence_3d_err':   coh_3d_err_arr[mc],
        'triaxiality':        triax_arr[mc],
    })
    pd.DataFrame(merge_df).to_csv(
        SAVE_DIR + 'merging_stats.csv', index=False
    )
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
