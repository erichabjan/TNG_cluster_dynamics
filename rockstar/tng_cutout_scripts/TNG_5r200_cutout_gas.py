#!/usr/bin/env python3
import os
import re
import h5py
import numpy as np
import requests
from urllib.parse import urljoin
import argparse
from typing import Tuple, Optional

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
                    coords = pt["Coordinates"][:]  # typically float32
                    vels   = pt["Velocities"][:]
                    pids   = pt["ParticleIDs"][:]
            except OSError as e:
                # corrupted local file (shouldn't happen if downloads validated)
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

            _append_rows(dset_pos, sel_coords)
            _append_rows(dset_vel, sel_vels)
            _append_rows(dset_ids, sel_pids)
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
        fout.attrs["N_dm_selected"] = n_written
        fout.attrs["n_chunks_read"] = n_chunks_read
        fout.attrs["n_chunks_missing_PartType0"] = n_chunks_missing_pt1

    print(f"[done] wrote {outname}  N_dm_selected={n_written}")
    return outname

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--halo-id", type=int, required=True)
    ap.add_argument("--sim", type=str, default="TNG300-1")
    ap.add_argument("--snap", type=int, default=99)
    ap.add_argument("--rmult", type=float, default=5.0)
    ap.add_argument("--rdef", type=str, default="Group_R_Crit200")
    ap.add_argument("--center-field", type=str, default="GroupPos")
    ap.add_argument("--chunks-dir", type=str, required=True,
                    help="Directory containing local snapshot chunk files snapshot-<snap>.<i>.hdf5")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--compression", type=str, default="gzip")
    args = ap.parse_args()

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

if __name__ == "__main__":
    main()
