#!/usr/bin/env python3
"""
Helpers shared by the TNG300 5r200 cutout scripts (gas, stars, bh): halo
centre and radius from the API, periodic distances, and which file supplies
each snapshot chunk for a particle type (tng_cutout_config's existing source
first, then the tng_chunking.py cache).
"""
import os
import json
import time
from typing import List, Optional

import h5py
import numpy as np
import requests

import tng_cutout_config as tcc

API_KEY = os.environ.get("TNG_API_KEY", "")
HEADERS = {"api-key": API_KEY}
STAMP_ATTR = "requested_fields"          # written by tng_chunking.py


def get_json(url: str, max_retries: int = 6, backoff: float = 2.0):
    """GET with retries: the TNG API returns intermittent 503/504s."""
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=(10, 120))
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == max_retries:
                raise
            print(f"[warn] {url}: {e}; retry {attempt}/{max_retries}", flush=True)
            time.sleep(backoff ** attempt)


def _local_group_field(sim: str, field: str, halo_id: int):
    path = os.path.join(tcc.GROUPCAT_DIR, f"{sim}_{field}.hdf5")
    if not os.path.exists(path):
        return None
    with h5py.File(path, "r") as f:
        return np.asarray(f["Group"][field][halo_id])


def halo_geometry(sim: str, snap: int, halo_id: int, rmult: float,
                  rdef: str = "Group_R_Crit200", center_field: str = "GroupPos"):
    """
    (center [ckpc/h], r200, rcut, boxsize) as the DM/gas cutouts used: from the
    local group catalogs when present (snapshot 99 only), else the API.
    """
    center = r200 = None
    if snap == tcc.SNAP and sim in tcc.BOXSIZE:
        center = _local_group_field(sim, center_field, halo_id)
        r200 = _local_group_field(sim, rdef, halo_id)
    if center is not None and r200 is not None:
        center = center.astype(np.float64)
        r200 = float(r200)
        boxsize = tcc.BOXSIZE[sim]
    else:
        if len(API_KEY) != 32:
            raise RuntimeError("Set TNG_API_KEY env var (32-char string) before running.")
        halo = get_json(f"{tcc.API_BASE_URL}{sim}/snapshots/{snap}/halos/{halo_id}/info.json")
        center = np.asarray(halo[center_field], dtype=np.float64)
        r200 = float(halo[rdef])
        boxsize = float(get_json(f"{tcc.API_BASE_URL}{sim}/")["boxsize"])
    return center, r200, float(rmult) * r200, boxsize


def periodic_radius(coords, center, boxsize):
    d = np.asarray(coords, dtype=np.float64) - center
    d = (d + 0.5 * boxsize) % boxsize - 0.5 * boxsize
    return np.sqrt(np.einsum("ij,ij->i", d, d))


def _provides(path: str, ptype: str, fields, require_stamp: bool) -> bool:
    """Same test as tng_chunking.has_fields: the fields are there, or the
    chunk has no particles of the type (and, for cache files, it was asked for)."""
    try:
        with h5py.File(path, "r") as f:
            if require_stamp:
                stamp = json.loads(f.attrs.get(STAMP_ATTR, "{}"))
                if not set(fields) <= set(stamp.get(ptype, ())):
                    return False
            grp = f"PartType{tcc.PARTTYPE_NUM[ptype]}"
            if grp in f:
                return all(k in f[grp] for k in fields)
            return int(f["Header"].attrs["NumPart_ThisFile"][tcc.PARTTYPE_NUM[ptype]]) == 0
    except Exception:
        return False


def chunk_sources(sim: str, ptype: str, fields, snap: int = tcc.SNAP,
                  max_chunks: Optional[int] = None) -> List[str]:
    """
    The file to read chunk i from, for every chunk of `sim`, for `ptype`.
    Raises, listing the gaps, if any chunk is in neither the existing source
    nor the cache -- i.e. tng_chunking.py has not finished -- so a cutout is
    never written from a partial snapshot.

    max_chunks limits the scan to the first N chunks, for smoke tests only:
    the resulting cutout is incomplete.
    """
    n = tcc.sim_info(sim)["n_chunks"]
    if max_chunks:
        n = min(n, int(max_chunks))
    paths, missing = [], []
    for idx in range(n):
        ex = tcc.existing_chunk_path(sim, ptype, idx, snap)
        if ex is not None and _provides(ex, ptype, fields, require_stamp=False):
            paths.append(ex)
            continue
        cache = tcc.cache_chunk_path(sim, idx, snap)
        if _provides(cache, ptype, fields, require_stamp=True):
            paths.append(cache)
            continue
        missing.append(idx)
    if missing:
        head = ", ".join(map(str, missing[:20])) + (" ..." if len(missing) > 20 else "")
        raise FileNotFoundError(
            f"{sim} {ptype}: {len(missing)} of {n} chunks have no source with fields "
            f"{list(fields)} (chunks {head}). Run / finish tng_chunking.slurm first.")
    return paths


def write_parttype_cutout(ptype: str, halo_id: int, sim: str, snap: int, rmult: float,
                          rdef: str, center_field: str, outname: str,
                          compression: Optional[str] = "gzip", overwrite: bool = False,
                          max_chunks: Optional[int] = None) -> str:
    """
    Cutout of `ptype` ('stars' or 'bh') within rmult * R200 of the halo centre,
    read chunk by chunk like the DM and gas cutouts, with the fields in
    tng_cutout_config.FIELDS[ptype]. Coordinates/Velocities/Masses are float32
    (as in the DM and gas cutouts), ParticleIDs uint64, other fields keep the
    snapshot dtype. Written to outname + '.part' and renamed into place, so a
    killed task never leaves a file that looks finished; an existing complete
    file is skipped unless overwrite.
    """
    fields = tcc.FIELDS[ptype]
    grp_name = f"PartType{tcc.PARTTYPE_NUM[ptype]}"

    if os.path.exists(outname):
        if overwrite:
            os.remove(outname)
        else:
            print(f"[skip] {outname} exists (use --overwrite to replace)")
            return outname

    center, r200, rcut, boxsize = halo_geometry(sim, snap, halo_id, rmult, rdef, center_field)
    sources = chunk_sources(sim, ptype, fields, snap, max_chunks=max_chunks)
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    tmp = outname + ".part"

    cast = {"Coordinates": "f4", "Velocities": "f4", "Masses": "f4", "ParticleIDs": "u8"}
    n_written = n_chunks_without = 0
    with h5py.File(tmp, "w") as fout:
        g = fout.create_group(grp_name)
        dsets = {}
        for fp in sources:
            with h5py.File(fp, "r") as f:
                if grp_name not in f:
                    n_chunks_without += 1
                    continue
                pt = f[grp_name]
                sel = periodic_radius(pt["Coordinates"][:], center, boxsize) <= rcut
                if not np.any(sel):
                    continue
                for k in fields:
                    v = pt[k][:][sel]
                    v = v.astype(cast.get(k, v.dtype), copy=False)
                    if k not in dsets:
                        dsets[k] = g.create_dataset(k, shape=(0,) + v.shape[1:],
                                                    maxshape=(None,) + v.shape[1:], dtype=v.dtype,
                                                    chunks=True, compression=compression)
                    append_rows(dsets[k], v)
                n_written += int(np.count_nonzero(sel))
        for k in fields:              # a halo with none of this type still gets the layout
            if k not in dsets:
                shape = (0, 3) if k in ("Coordinates", "Velocities") else (0,)
                g.create_dataset(k, shape=shape, maxshape=(None,) + shape[1:],
                                 dtype=cast.get(k, "f4"), chunks=True, compression=compression)

        fout.attrs["sim"] = sim
        fout.attrs["snap"] = snap
        fout.attrs["halo_id"] = halo_id
        fout.attrs["center_field"] = center_field
        fout.attrs["center"] = center
        fout.attrs["rdef"] = rdef
        fout.attrs["r200"] = r200
        fout.attrs["rmult"] = rmult
        fout.attrs["rcut"] = rcut
        fout.attrs["boxsize"] = boxsize
        fout.attrs[f"N_{ptype}_selected"] = n_written
        fout.attrs["n_chunks_read"] = len(sources)
        fout.attrs[f"n_chunks_missing_{grp_name}"] = n_chunks_without
        if max_chunks:
            fout.attrs["INCOMPLETE_max_chunks"] = int(max_chunks)
    os.replace(tmp, outname)
    print(f"[done] wrote {outname}  N_{ptype}_selected={n_written} from {len(sources)} chunks")
    return outname


def cutout_main(ptype: str, doc: str):
    """Command line shared by TNG_5r200_cutout_stars.py and TNG_5r200_cutout_bh.py."""
    import argparse
    ap = argparse.ArgumentParser(description=doc,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--halo-id", type=int, required=True)
    ap.add_argument("--sim", type=str, default="TNG300-1")
    ap.add_argument("--snap", type=int, default=tcc.SNAP)
    ap.add_argument("--rmult", type=float, default=5.0)
    ap.add_argument("--rdef", type=str, default="Group_R_Crit200")
    ap.add_argument("--center-field", type=str, default="GroupPos")
    ap.add_argument("--out", type=str, default=None,
                    help="Cutout path (default: tng_cutout_config.cutout_path)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--compression", type=str, default="gzip")
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="Smoke tests only: read the first N chunks (the cutout will be incomplete).")
    args = ap.parse_args()
    write_parttype_cutout(
        ptype, args.halo_id, args.sim, args.snap, args.rmult, args.rdef, args.center_field,
        args.out or tcc.cutout_path(args.sim, ptype, args.halo_id),
        compression=args.compression, overwrite=args.overwrite, max_chunks=args.max_chunks)


def append_rows(dset, arr):
    n0 = dset.shape[0]
    n1 = n0 + arr.shape[0]
    if arr.ndim == 1:
        dset.resize((n1,))
        dset[n0:n1] = arr
    else:
        dset.resize((n1, arr.shape[1]))
        dset[n0:n1, :] = arr
