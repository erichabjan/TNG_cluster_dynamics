#!/usr/bin/env python3
"""
Gas cutouts within rmult * R200 of a TNG halo.

Default mode (TNG300-1/2/3, configured in tng_cutout_config.py): ADD the
X-ray fields -- Density, InternalEnergy, ElectronAbundance, StarFormationRate
-- to the existing gas_within_5r200_<halo>.hdf5, whose Coordinates,
Velocities, Masses and ParticleIDs are left untouched. Cells are matched by
ParticleID, so the cutout's particle set (the same one the DM cutout and every
earlier analysis used) does not change. Each chunk comes from the existing
TNG300-1 gas chunks when they hold the fields, else from the tng_chunking.py
cache; the cutout is only written once every one of its cells has been found.

--fresh: the original behaviour -- build a new Coordinates / Velocities /
Masses / ParticleIDs cutout from a local chunk directory (--chunks-dir, --out),
e.g. for TNG-Cluster.
"""
import os
import re
import sys
import time
import h5py
import numpy as np
import requests
from urllib.parse import urljoin
import argparse
from typing import Tuple, Optional

import tng_cutout_config as tcc
import tng_cutout_common as tc

BASEURL = "http://www.tng-project.org/api/"
# Prefer env var so you don't hardcode keys in scripts
API_KEY = os.environ.get("TNG_API_KEY", "")
HEADERS = {"api-key": API_KEY}

def get_json(url: str):
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json()

def _periodic_delta(x, x0, boxsize):
    dx = x - x0
    dx = (dx + 0.5 * boxsize) % boxsize - 0.5 * boxsize
    return dx

def _list_local_chunks(chunks_dir: str, snap: int) -> list[str]:
    """
    Return sorted list of local files: snapshot-<snap>.<i>.hdf5
    """
    pat = re.compile(rf"snapshot-{snap}\.(\d+)\.hdf5$")
    out = []
    for fn in os.listdir(chunks_dir):
        m = pat.match(fn)
        if m:
            out.append((int(m.group(1)), os.path.join(chunks_dir, fn)))
    out.sort(key=lambda x: x[0])
    return [p for _, p in out]

def _append_rows(dset, arr):
    n0 = dset.shape[0]
    n1 = n0 + arr.shape[0]
    if arr.ndim == 1:
        dset.resize((n1,))
        dset[n0:n1] = arr
    else:
        dset.resize((n1, arr.shape[1]))
        dset[n0:n1, :] = arr

def write_dm_cutout_within_rmult(
    halo_id: int,
    sim: str,
    snap: int,
    rmult: float,
    rdef: str,
    center_field: str,
    chunks_dir: str,
    outname: str,
    compression: Optional[str] = "gzip",
    overwrite: bool = False,
    dtype_pos: str = "f4",
    dtype_vel: str = "f4",
    dtype_id: str  = "u8",
    dtype_mass: str = "f4",
) -> str:
    """
    Build a DM cutout within rmult*r200 around halo center using local snapshot chunk files.
    Writes PartType0/{Coordinates,Velocities,ParticleIDs} to outname.
    """

    if len(API_KEY) != 32:
        raise RuntimeError("Set TNG_API_KEY env var (32-char string) before running.")

    if os.path.exists(outname):
        if overwrite:
            os.remove(outname)
        else:
            raise FileExistsError(f"{outname} exists (set overwrite=True to replace).")

    # Halo info (center + r200)
    halo_info_url = f"{BASEURL}{sim}/snapshots/{snap}/halos/{halo_id}/info.json"
    halo = get_json(halo_info_url)
    center = np.asarray(halo[center_field], dtype=np.float64)  # (3,)
    r200   = float(halo[rdef])
    rcut   = float(rmult) * r200

    # Boxsize for periodic distances
    sim_meta = get_json(f"{BASEURL}{sim}/")
    boxsize = float(sim_meta["boxsize"])

    # Local chunk files
    chunk_files = _list_local_chunks(chunks_dir, snap)
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {chunks_dir} for snap={snap}")

    os.makedirs(os.path.dirname(outname), exist_ok=True)

    with h5py.File(outname, "w") as fout:
        g = fout.create_group("PartType0")
        dset_pos = g.create_dataset(
            "Coordinates",
            shape=(0, 3),
            maxshape=(None, 3),
            dtype=dtype_pos,
            chunks=True,
            compression=compression
        )
        dset_vel = g.create_dataset(
            "Velocities",
            shape=(0, 3),
            maxshape=(None, 3),
            dtype=dtype_vel,
            chunks=True,
            compression=compression
        )
        dset_ids = g.create_dataset(
            "ParticleIDs",
            shape=(0,),
            maxshape=(None,),
            dtype=dtype_id,
            chunks=True,
            compression=compression
        )
        dset_mass = g.create_dataset(
            "Masses",
            shape=(0,),
            maxshape=(None,),
            dtype=dtype_mass,
            chunks=True,
            compression=compression
        )

        n_written = 0
        n_chunks_read = 0
        n_chunks_missing_pt1 = 0

        for fp in chunk_files:
            n_chunks_read += 1
            try:
                with h5py.File(fp, "r") as f:
                    if "PartType0" not in f:
                        n_chunks_missing_pt1 += 1
                        continue
                    pt = f["PartType0"]
                    coords = pt["Coordinates"][:]
                    vels   = pt["Velocities"][:]
                    pids   = pt["ParticleIDs"][:]
                    masses = pt["Masses"][:]
            except OSError as e:
                print(f"[warn] could not read {fp}: {e}")
                continue

            # periodic radius
            dx = _periodic_delta(coords[:, 0], center[0], boxsize)
            dy = _periodic_delta(coords[:, 1], center[1], boxsize)
            dz = _periodic_delta(coords[:, 2], center[2], boxsize)
            rr = np.sqrt(dx*dx + dy*dy + dz*dz)

            m = rr <= rcut
            if not np.any(m):
                continue

            sel_coords = coords[m].astype(dtype_pos, copy=False)
            sel_vels   = vels[m].astype(dtype_vel, copy=False)
            sel_pids   = pids[m].astype(dtype_id,  copy=False)
            sel_masses = masses[m].astype(dtype_mass, copy=False)

            _append_rows(dset_pos, sel_coords)
            _append_rows(dset_vel, sel_vels)
            _append_rows(dset_ids, sel_pids)
            _append_rows(dset_mass, sel_masses)
            n_written += sel_coords.shape[0]

        fout.attrs["sim"] = sim
        fout.attrs["snap"] = snap
        fout.attrs["halo_id"] = halo_id
        fout.attrs["center_field"] = center_field
        fout.attrs["rdef"] = rdef
        fout.attrs["r200"] = r200
        fout.attrs["rmult"] = rmult
        fout.attrs["rcut"] = rcut
        fout.attrs["boxsize"] = boxsize
        fout.attrs["N_gas_selected"] = n_written
        fout.attrs["n_chunks_read"] = n_chunks_read
        fout.attrs["n_chunks_missing_PartType0"] = n_chunks_missing_pt1

    print(f"[done] wrote {outname}  N_gas_selected={n_written}")
    return outname


# Fields add_xray_fields writes; Coordinates and ParticleIDs in FIELDS['gas']
# are only read, for the radius cut and the match.
XRAY_FIELDS = tuple(k for k in tcc.FIELDS["gas"] if k not in ("Coordinates", "ParticleIDs"))
DONE_ATTR = "xray_fields_added"


def add_xray_fields(
    halo_id: int,
    sim: str,
    snap: int,
    rmult: float,
    rdef: str,
    center_field: str,
    cutout: str,
    compression: Optional[str] = "gzip",
    overwrite: bool = False,
    max_chunks: Optional[int] = None,
) -> str:
    """
    Add XRAY_FIELDS to an existing gas cutout, in place, by ParticleID.

    Every chunk is cut to the same rmult * R200 sphere the cutout was made
    with (a 1e-6 relative margin absorbs float rounding at the edge; cells
    selected that are not in the cutout are counted and ignored), and its
    cells are matched to the cutout by ParticleID. Nothing is written until
    every cutout cell has a value, so the file is never left half-updated
    with plausible-looking data; on success the attr xray_fields_added marks
    it done, and a rerun skips it unless overwrite.
    """
    if not os.path.exists(cutout):
        raise FileNotFoundError(
            f"{cutout} does not exist; this mode adds fields to an existing gas "
            f"cutout (use --fresh to build one from chunks)")

    with h5py.File(cutout, "r") as f:
        g = f["PartType0"]
        if (not overwrite and f.attrs.get(DONE_ATTR) is not None
                and all(k in g for k in XRAY_FIELDS)):
            print(f"[skip] {cutout} already has {list(XRAY_FIELDS)}")
            return cutout
        ids = g["ParticleIDs"][:]
        old_rcut = float(f.attrs["rcut"])

    center, r200, rcut, boxsize = tc.halo_geometry(sim, snap, halo_id, rmult, rdef, center_field)
    if abs(rcut - old_rcut) > 1e-6 * old_rcut:
        raise ValueError(f"{cutout}: rcut {old_rcut} in the file, {rcut} from the API now")

    sources = tc.chunk_sources(sim, "gas", tcc.FIELDS["gas"], snap, max_chunks=max_chunks)
    ex0 = tcc.existing_chunk_path(sim, "gas", 0, snap)
    ex_dir = os.path.dirname(ex0) if ex0 else None
    n_existing = sum(1 for p in sources if ex_dir and os.path.dirname(p) == ex_dir)
    print(f"[info] {sim} halo {halo_id}: {ids.size:,} cutout cells, rcut={rcut:.2f} ckpc/h, "
          f"{len(sources)} chunks ({n_existing} existing, {len(sources) - n_existing} cache)", flush=True)

    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    if np.any(sorted_ids[1:] == sorted_ids[:-1]):
        raise ValueError(f"{cutout}: duplicate ParticleIDs; cannot match by ID")

    values = {}
    filled = np.zeros(ids.size, dtype=bool)
    n_outside_cutout = 0
    r_sel = rcut * (1.0 + 1e-6)
    t0 = time.time()

    for i, fp in enumerate(sources):
        with h5py.File(fp, "r") as f:
            g = f["PartType0"] if "PartType0" in f else None
            sel = (tc.periodic_radius(g["Coordinates"][:], center, boxsize) <= r_sel
                   if g is not None else None)
            if sel is not None and np.any(sel):
                pid = g["ParticleIDs"][:][sel]
                pos = np.searchsorted(sorted_ids, pid)
                pos_c = np.minimum(pos, sorted_ids.size - 1)
                hit = sorted_ids[pos_c] == pid
                n_outside_cutout += int(np.count_nonzero(~hit))
                target = order[pos_c[hit]]
                if np.any(filled[target]):
                    raise ValueError(f"{fp}: a cutout cell was matched twice")
                for k in XRAY_FIELDS:
                    v = g[k][:][sel][hit]
                    if k not in values:
                        values[k] = np.full(ids.size, np.nan, dtype=v.dtype)
                    values[k][target] = v
                filled[target] = True
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(sources)} chunks, {int(filled.sum()):,} cells matched, "
                  f"{time.time() - t0:.0f} s", flush=True)

    n_missing = int(ids.size - filled.sum())
    if n_missing:
        raise RuntimeError(
            f"{cutout}: {n_missing:,} of {ids.size:,} cells were not found in the chunks "
            f"(max_chunks={max_chunks}); nothing written")

    with h5py.File(cutout, "a") as f:
        g = f["PartType0"]
        for k in XRAY_FIELDS:
            if k in g:
                del g[k]                  # a partial earlier write, or --overwrite
            g.create_dataset(k, data=values[k], chunks=True, compression=compression)
        f.attrs["xray_fields_source"] = (
            f"{n_existing} chunks from the existing source, "
            f"{len(sources) - n_existing} from the tng_chunking cache")
        f.attrs["xray_fields_n_selected_not_in_cutout"] = n_outside_cutout
        f.attrs[DONE_ATTR] = ",".join(XRAY_FIELDS)    # last: marks the file complete

    print(f"[done] {cutout}: added {list(XRAY_FIELDS)} for {ids.size:,} cells "
          f"({n_outside_cutout} edge cells outside the cutout ignored), {time.time() - t0:.0f} s")
    return cutout


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--halo-id", type=int, required=True)
    ap.add_argument("--sim", type=str, default="TNG300-1")
    ap.add_argument("--snap", type=int, default=99)
    ap.add_argument("--rmult", type=float, default=5.0)
    ap.add_argument("--rdef", type=str, default="Group_R_Crit200")
    ap.add_argument("--center-field", type=str, default="GroupPos")
    ap.add_argument("--fresh", action="store_true",
                    help="Build a new pos/vel/mass/ID cutout from --chunks-dir (original behaviour).")
    ap.add_argument("--chunks-dir", type=str, default=None,
                    help="--fresh only: directory containing snapshot-<snap>.<i>.hdf5")
    ap.add_argument("--out", type=str, default=None,
                    help="Cutout path (default: tng_cutout_config.cutout_path; required with --fresh)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--compression", type=str, default="gzip")
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="Smoke tests only: scan the first N chunks (the match will be incomplete).")
    args = ap.parse_args()

    if args.fresh:
        if not (args.chunks_dir and args.out):
            ap.error("--fresh needs --chunks-dir and --out")
        write_dm_cutout_within_rmult(
            halo_id=args.halo_id,
            sim=args.sim,
            snap=args.snap,
            rmult=args.rmult,
            rdef=args.rdef,
            center_field=args.center_field,
            chunks_dir=args.chunks_dir,
            outname=args.out,
            compression=args.compression,
            overwrite=args.overwrite,
        )
        return

    add_xray_fields(
        halo_id=args.halo_id,
        sim=args.sim,
        snap=args.snap,
        rmult=args.rmult,
        rdef=args.rdef,
        center_field=args.center_field,
        cutout=args.out or tcc.cutout_path(args.sim, "gas", args.halo_id),
        compression=args.compression,
        overwrite=args.overwrite,
        max_chunks=args.max_chunks,
    )

if __name__ == "__main__":
    main()
