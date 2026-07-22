#!/usr/bin/env python3
"""Create a mass-aware TNG-Cluster dark-matter cutout.

The public TNG-Cluster snapshot is a stitched "virtual box" containing 352
independent zoom simulations.  Each original zoom has its own low-resolution
background, so a spatial selection over every snapshot chunk would double
count overlapping backgrounds.

This script:

1. obtains the target halo's GroupOrigHaloID;
2. identifies snapshot chunks belonging to that original zoom;
3. selects PartType1 (high-resolution) and PartType2 (coarse) particles within
   rmult * R200, with periodic wrapping;
4. writes one combined DarkMatter table containing a mass and source particle
   type for every selected particle.

Output layout
-------------
DarkMatter/Coordinates       float[N,3], ckpc/h
DarkMatter/Velocities        float[N,3], km sqrt(a) / s
DarkMatter/ParticleIDs       uint64[N]
DarkMatter/Masses            float[N], 1e10 Msun/h
DarkMatter/SourcePartType    uint8[N], 1=high-resolution, 2=coarse

Root attributes include separate PartType1/PartType2 counts and mass sums,
coarse-particle number and mass fractions, and the aperture metadata.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Any, Optional

import h5py
import numpy as np
import requests


BASEURL = "https://www.tng-project.org/api/"
API_KEY = os.environ.get("TNG_API_KEY", "")
HEADERS = {"api-key": API_KEY}
ID_BLOCK_SIZE = 1_000_000_000
PARTICLE_TYPES = (1, 2)


def get_json(url: str) -> dict[str, Any]:
    response = requests.get(url, headers=HEADERS, timeout=(10, 120))
    response.raise_for_status()
    return response.json()


def _periodic_delta(x: np.ndarray, x0: np.ndarray, boxsize: float) -> np.ndarray:
    """Minimum-image displacement for positions in a periodic box."""
    dx = np.asarray(x, dtype=np.float64) - np.asarray(x0, dtype=np.float64)
    dx -= boxsize * np.rint(dx / boxsize)
    return dx


def _list_local_chunks(chunks_dir: str, snap: int) -> list[tuple[int, str]]:
    """Return ``(chunk_index, path)`` pairs sorted by chunk index."""
    pattern = re.compile(rf"snapshot-{snap}\.(\d+)\.hdf5$")
    chunks: list[tuple[int, str]] = []

    for filename in os.listdir(chunks_dir):
        match = pattern.fullmatch(filename)
        if match:
            chunks.append((int(match.group(1)), os.path.join(chunks_dir, filename)))

    chunks.sort(key=lambda item: item[0])
    return chunks


def _validate_chunk_inventory(
    chunks: list[tuple[int, str]],
    allow_missing_chunks: bool,
) -> None:
    if not chunks:
        raise FileNotFoundError("No matching snapshot chunk files were found.")

    with h5py.File(chunks[0][1], "r") as first:
        expected = int(first["Header"].attrs.get("NumFilesPerSnapshot", len(chunks)))

    found_indices = {index for index, _ in chunks}
    missing = sorted(set(range(expected)) - found_indices)
    unexpected = sorted(index for index in found_indices if index >= expected)

    if unexpected:
        raise RuntimeError(
            f"Found chunk indices outside the Header expectation 0..{expected - 1}: "
            f"{unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}"
        )

    if missing and not allow_missing_chunks:
        preview = missing[:20]
        raise RuntimeError(
            f"Snapshot is incomplete: found {len(chunks)} of {expected} chunks; "
            f"missing {preview}{' ...' if len(missing) > 20 else ''}. "
            "Use --allow-missing-chunks only if you intentionally accept an "
            "incomplete search."
        )


def _first_particle_id(handle: h5py.File) -> Optional[int]:
    """Return one ParticleID from a chunk, independent of particle type."""
    for part_type in (1, 2, 0, 3, 4, 5):
        group_name = f"PartType{part_type}"
        if group_name not in handle:
            continue
        group = handle[group_name]
        if "ParticleIDs" in group and group["ParticleIDs"].shape[0] > 0:
            return int(group["ParticleIDs"][0])
    return None


def _chunk_orig_halo_id(path: str) -> Optional[int]:
    """Infer the original zoom provenance from a chunk's shifted IDs."""
    with h5py.File(path, "r") as handle:
        particle_id = _first_particle_id(handle)
    if particle_id is None:
        return None
    return particle_id // ID_BLOCK_SIZE


def _find_original_zoom_chunks(
    chunks: list[tuple[int, str]],
    orig_halo_id: int,
) -> list[tuple[int, str]]:
    """Find chunks whose particles/cells belong to one original zoom."""
    selected: list[tuple[int, str]] = []
    empty_chunks: list[int] = []

    for chunk_index, path in chunks:
        chunk_orig_id = _chunk_orig_halo_id(path)
        if chunk_orig_id is None:
            empty_chunks.append(chunk_index)
        elif chunk_orig_id == orig_halo_id:
            selected.append((chunk_index, path))

    if empty_chunks:
        print(
            f"[warn] {len(empty_chunks)} chunks contained no ParticleIDs and could "
            "not be assigned provenance."
        )

    if not selected:
        raise RuntimeError(
            f"No local chunks belong to GroupOrigHaloID={orig_halo_id}. "
            "Check the simulation, snapshot, and chunk inventory."
        )

    return selected


def _halo_orig_id(halo: dict[str, Any], override: Optional[int]) -> int:
    if override is not None:
        return int(override)

    aliases = (
        "GroupOrigHaloID",
        "group_orig_halo_id",
        "orig_halo_id",
        "origID",
    )
    for key in aliases:
        if key in halo:
            return int(halo[key])

    raise KeyError(
        "The halo API response did not contain GroupOrigHaloID. Supply it with "
        "--orig-halo-id. Available API keys include: "
        + ", ".join(sorted(halo.keys()))
    )


def _append_rows(dataset: h5py.Dataset, values: np.ndarray) -> None:
    if values.shape[0] == 0:
        return

    old_size = dataset.shape[0]
    new_size = old_size + values.shape[0]
    dataset.resize((new_size,) + dataset.shape[1:])
    dataset[old_size:new_size, ...] = values


def _particle_masses(
    handle: h5py.File,
    group: h5py.Group,
    part_type: int,
    indices: Optional[np.ndarray],
    count: int,
) -> np.ndarray:
    """Read individual masses or construct them from the Header MassTable."""
    if "Masses" in group:
        if indices is None:
            return np.asarray(group["Masses"][:], dtype=np.float64)
        return np.asarray(group["Masses"][indices], dtype=np.float64)

    mass_table = np.asarray(handle["Header"].attrs["MassTable"], dtype=np.float64)
    constant_mass = float(mass_table[part_type])
    if constant_mass <= 0.0:
        raise RuntimeError(
            f"PartType{part_type} has neither a Masses dataset nor a positive "
            "Header/MassTable value."
        )
    return np.full(count, constant_mass, dtype=np.float64)


def _create_output_datasets(
    output: h5py.File,
    compression: Optional[str],
    dtype_pos: str,
    dtype_vel: str,
    dtype_id: str,
    dtype_mass: str,
) -> dict[str, h5py.Dataset]:
    group = output.create_group("DarkMatter")
    datasets = {
        "Coordinates": group.create_dataset(
            "Coordinates",
            shape=(0, 3),
            maxshape=(None, 3),
            dtype=dtype_pos,
            chunks=True,
            compression=compression,
        ),
        "Velocities": group.create_dataset(
            "Velocities",
            shape=(0, 3),
            maxshape=(None, 3),
            dtype=dtype_vel,
            chunks=True,
            compression=compression,
        ),
        "ParticleIDs": group.create_dataset(
            "ParticleIDs",
            shape=(0,),
            maxshape=(None,),
            dtype=dtype_id,
            chunks=True,
            compression=compression,
        ),
        "Masses": group.create_dataset(
            "Masses",
            shape=(0,),
            maxshape=(None,),
            dtype=dtype_mass,
            chunks=True,
            compression=compression,
        ),
        "SourcePartType": group.create_dataset(
            "SourcePartType",
            shape=(0,),
            maxshape=(None,),
            dtype="u1",
            chunks=True,
            compression=compression,
        ),
    }

    datasets["Coordinates"].attrs["units"] = "ckpc/h"
    datasets["Velocities"].attrs["units"] = "km sqrt(a) / s"
    datasets["Masses"].attrs["units"] = "1e10 Msun/h"
    datasets["SourcePartType"].attrs["meaning"] = (
        "1=high-resolution dark matter; 2=coarse/low-resolution dark matter"
    )
    return datasets


def write_dm_cutout_within_rmult(
    halo_id: int,
    sim: str,
    snap: int,
    rmult: float,
    rdef: str,
    center_field: str,
    chunks_dir: str,
    outname: str,
    orig_halo_id: Optional[int] = None,
    compression: Optional[str] = "gzip",
    overwrite: bool = False,
    allow_missing_chunks: bool = False,
    dtype_pos: str = "f4",
    dtype_vel: str = "f4",
    dtype_id: str = "u8",
    dtype_mass: str = "f8",
) -> str:
    """Write a provenance-safe, mass-aware TNG-Cluster DM cutout."""
    if len(API_KEY) != 32:
        raise RuntimeError(
            "Set the TNG_API_KEY environment variable to your 32-character API key."
        )
    if rmult <= 0.0:
        raise ValueError("rmult must be positive.")

    outname = os.path.abspath(outname)
    temporary_outname = outname + ".part"
    if os.path.exists(outname) and not overwrite:
        raise FileExistsError(f"{outname} exists; pass --overwrite to replace it.")
    if os.path.exists(temporary_outname):
        os.remove(temporary_outname)

    halo_url = f"{BASEURL}{sim}/snapshots/{snap}/halos/{halo_id}/info.json"
    halo = get_json(halo_url)
    center = np.asarray(halo[center_field], dtype=np.float64)
    if center.shape != (3,):
        raise ValueError(f"{center_field} must contain three coordinates, got {center!r}.")
    r200 = float(halo[rdef])
    rcut = float(rmult) * r200
    target_orig_id = _halo_orig_id(halo, orig_halo_id)

    sim_meta = get_json(f"{BASEURL}{sim}/")
    boxsize = float(sim_meta["boxsize"])

    chunks = _list_local_chunks(chunks_dir, snap)
    _validate_chunk_inventory(chunks, allow_missing_chunks=allow_missing_chunks)
    zoom_chunks = _find_original_zoom_chunks(chunks, target_orig_id)
    print(
        f"[info] halo_id={halo_id} GroupOrigHaloID={target_orig_id} "
        f"zoom_chunks={[index for index, _ in zoom_chunks]}"
    )

    os.makedirs(os.path.dirname(outname), exist_ok=True)
    id_low = np.uint64(target_orig_id * ID_BLOCK_SIZE)
    id_high = np.uint64((target_orig_id + 1) * ID_BLOCK_SIZE)
    rcut_squared = rcut * rcut

    counts = {1: 0, 2: 0}
    mass_sums = {1: 0.0, 2: 0.0}
    seen_groups = {1: 0, 2: 0}
    missing_groups = {1: 0, 2: 0}

    try:
        with h5py.File(temporary_outname, "w") as output:
            datasets = _create_output_datasets(
                output,
                compression=compression,
                dtype_pos=dtype_pos,
                dtype_vel=dtype_vel,
                dtype_id=dtype_id,
                dtype_mass=dtype_mass,
            )

            for chunk_index, path in zoom_chunks:
                with h5py.File(path, "r") as handle:
                    for part_type in PARTICLE_TYPES:
                        group_name = f"PartType{part_type}"
                        if group_name not in handle:
                            missing_groups[part_type] += 1
                            continue

                        group = handle[group_name]
                        seen_groups[part_type] += 1
                        required = ("Coordinates", "Velocities", "ParticleIDs")
                        absent = [field for field in required if field not in group]
                        if absent:
                            raise KeyError(
                                f"{path}:{group_name} is missing required fields {absent}."
                            )

                        particle_ids_all = np.asarray(group["ParticleIDs"][:], dtype=np.uint64)
                        same_zoom = (particle_ids_all > id_low) & (particle_ids_all < id_high)
                        if not np.any(same_zoom):
                            continue

                        # Normally every particle in each selected file has the same
                        # provenance. Keep the explicit filter as a consistency guard.
                        if np.all(same_zoom):
                            provenance_indices = None
                            coordinates = np.asarray(group["Coordinates"][:])
                            velocities = np.asarray(group["Velocities"][:])
                            particle_ids = particle_ids_all
                        else:
                            provenance_indices = np.flatnonzero(same_zoom)
                            coordinates = np.asarray(group["Coordinates"][provenance_indices])
                            velocities = np.asarray(group["Velocities"][provenance_indices])
                            particle_ids = particle_ids_all[provenance_indices]

                        masses = _particle_masses(
                            handle,
                            group,
                            part_type=part_type,
                            indices=provenance_indices,
                            count=particle_ids.shape[0],
                        )

                        delta = _periodic_delta(coordinates, center, boxsize)
                        radius_squared = np.einsum("ij,ij->i", delta, delta)
                        inside = radius_squared <= rcut_squared
                        if not np.any(inside):
                            continue

                        selected_coordinates = coordinates[inside].astype(dtype_pos, copy=False)
                        selected_velocities = velocities[inside].astype(dtype_vel, copy=False)
                        selected_ids = particle_ids[inside].astype(dtype_id, copy=False)
                        selected_masses = masses[inside].astype(dtype_mass, copy=False)
                        selected_types = np.full(
                            selected_ids.shape[0], part_type, dtype=np.uint8
                        )

                        _append_rows(datasets["Coordinates"], selected_coordinates)
                        _append_rows(datasets["Velocities"], selected_velocities)
                        _append_rows(datasets["ParticleIDs"], selected_ids)
                        _append_rows(datasets["Masses"], selected_masses)
                        _append_rows(datasets["SourcePartType"], selected_types)

                        counts[part_type] += int(selected_ids.shape[0])
                        mass_sums[part_type] += float(np.sum(selected_masses, dtype=np.float64))

                print(
                    f"[info] processed original-zoom chunk {chunk_index}: "
                    f"N1={counts[1]} N2={counts[2]}"
                )

            if seen_groups[1] == 0:
                raise RuntimeError(
                    "PartType1 was absent from every chunk in the original zoom. "
                    "The local files are incomplete or were downloaded with an "
                    "incompatible field filter."
                )
            if seen_groups[2] == 0:
                raise RuntimeError(
                    "PartType2 was absent from every chunk in the original zoom. "
                    "TNG-Cluster's coarse background should be present; verify that "
                    "the chunk downloads were not restricted to the API 'dm' fields."
                )

            total_count = counts[1] + counts[2]
            total_mass = mass_sums[1] + mass_sums[2]
            if total_count == 0:
                raise RuntimeError(
                    "No dark-matter particles were selected. Check halo metadata, "
                    "GroupOrigHaloID, units, and chunk provenance."
                )

            output.attrs["sim"] = sim
            output.attrs["snap"] = snap
            output.attrs["halo_id"] = halo_id
            output.attrs["GroupOrigHaloID"] = target_orig_id
            output.attrs["center_field"] = center_field
            output.attrs["center"] = center
            output.attrs["rdef"] = rdef
            output.attrs["r200"] = r200
            output.attrs["rmult"] = rmult
            output.attrs["rcut"] = rcut
            output.attrs["boxsize"] = boxsize
            output.attrs["coordinate_units"] = "ckpc/h"
            output.attrs["mass_units"] = "1e10 Msun/h"
            output.attrs["N_dm_selected"] = total_count
            output.attrs["N_PartType1"] = counts[1]
            output.attrs["N_PartType2"] = counts[2]
            output.attrs["Mass_PartType1"] = mass_sums[1]
            output.attrs["Mass_PartType2"] = mass_sums[2]
            output.attrs["Mass_dm_selected"] = total_mass
            output.attrs["coarse_number_fraction"] = counts[2] / total_count
            output.attrs["coarse_mass_fraction"] = (
                mass_sums[2] / total_mass if total_mass > 0.0 else np.nan
            )
            output.attrs["strictly_high_resolution"] = np.uint8(counts[2] == 0)
            output.attrs["n_snapshot_chunks_found"] = len(chunks)
            output.attrs["n_original_zoom_chunks_read"] = len(zoom_chunks)
            output.attrs["original_zoom_chunk_indices"] = np.asarray(
                [index for index, _ in zoom_chunks], dtype=np.int32
            )
            output.attrs["n_zoom_chunks_missing_PartType1"] = missing_groups[1]
            output.attrs["n_zoom_chunks_missing_PartType2"] = missing_groups[2]

        os.replace(temporary_outname, outname)
    except Exception:
        if os.path.exists(temporary_outname):
            os.remove(temporary_outname)
        raise

    print(
        f"[done] wrote {outname} N={counts[1] + counts[2]} "
        f"(PartType1={counts[1]}, PartType2={counts[2]}) "
        f"coarse_mass_fraction="
        f"{mass_sums[2] / (mass_sums[1] + mass_sums[2]) if total_mass > 0 else np.nan:.6g}"
    )
    return outname


def _compression_arg(value: str) -> Optional[str]:
    if value.lower() in ("none", "off", "false"):
        return None
    if value not in ("gzip", "lzf"):
        raise argparse.ArgumentTypeError("compression must be gzip, lzf, or none")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a single mass-aware dark-matter cutout from one original "
            "TNG-Cluster zoom."
        )
    )
    parser.add_argument("--halo-id", type=int, required=True)
    parser.add_argument("--sim", default="TNG-Cluster")
    parser.add_argument("--snap", type=int, default=99)
    parser.add_argument("--rmult", type=float, default=5.0)
    parser.add_argument("--rdef", default="Group_R_Crit200")
    parser.add_argument("--center-field", default="GroupPos")
    parser.add_argument(
        "--orig-halo-id",
        type=int,
        default=None,
        help=(
            "Override GroupOrigHaloID. Normally read from the halo API response; "
            "use this only if that response omits the field."
        ),
    )
    parser.add_argument(
        "--chunks-dir",
        required=True,
        help="Directory containing snapshot-<snap>.<index>.hdf5 files.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compression", type=_compression_arg, default="gzip")
    parser.add_argument(
        "--allow-missing-chunks",
        action="store_true",
        help="Allow an incomplete snapshot inventory (not recommended).",
    )
    args = parser.parse_args()

    write_dm_cutout_within_rmult(
        halo_id=args.halo_id,
        sim=args.sim,
        snap=args.snap,
        rmult=args.rmult,
        rdef=args.rdef,
        center_field=args.center_field,
        chunks_dir=args.chunks_dir,
        outname=args.out,
        orig_halo_id=args.orig_halo_id,
        compression=args.compression,
        overwrite=args.overwrite,
        allow_missing_chunks=args.allow_missing_chunks,
    )


if __name__ == "__main__":
    main()