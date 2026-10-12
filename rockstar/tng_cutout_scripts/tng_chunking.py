#!/usr/bin/env python3
"""
Download and cache TNG snapshot chunk files to scratch.

For a simulation configured in tng_cutout_config.SIMS, the particle types and
fields to fetch come from tng_cutout_config.FIELDS, and the chunk count from
SIMS[sim]['n_chunks'], so `--sim` alone is enough. Each chunk is checked per
particle type:
  - satisfied by an EXISTING on-disk source (e.g. the TNG300-1 gas chunks under
    /projects/.../mcdonough.b) -> nothing is downloaded for that type;
  - satisfied by the scratch CACHE (a previous download that requested those
    fields) -> skipped;
  - otherwise only the missing types are requested from the API, with field
    filtering, and the requested fields are stamped into the file
    (attr 'requested_fields'). The stamp is what proves a type was asked for:
    a filtered download reports NumPart_ThisFile = 0 for every type it did not
    request, which looks the same as a chunk with no particles of that type.

Unconfigured simulations (e.g. TNG-Cluster) keep the original behaviour: the
whole chunk, or DM only with --dm-only.
"""
import os
import sys
import json
import argparse
import time
import requests
import h5py
from typing import Dict, List, Optional, Tuple

import tng_cutout_config as tcc

BASE_URL = tcc.API_BASE_URL
STAMP_ATTR = "requested_fields"


def parse_indices(s: str) -> List[int]:
    # "1,2,5-9,20"
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_fields(s: str) -> Dict[str, Tuple[str, ...]]:
    """'gas:Coordinates,Density;stars:Coordinates,Masses' -> {ptype: fields}."""
    out = {}
    for block in s.split(";"):
        block = block.strip()
        if not block:
            continue
        ptype, _, fields = block.partition(":")
        ptype = ptype.strip()
        if ptype not in tcc.PARTTYPE_NUM:
            raise ValueError(f"unknown particle type {ptype!r} in --fields "
                             f"(expected one of {sorted(tcc.PARTTYPE_NUM)})")
        out[ptype] = tuple(f.strip() for f in fields.split(",") if f.strip())
    return out


def is_valid_hdf5(path: str) -> bool:
    try:
        with h5py.File(path, "r"):
            return True
    except Exception:
        return False


def has_fields(path: str, ptype: str, fields, require_stamp: bool) -> bool:
    """
    True if `path` provides `fields` for `ptype`: the PartType group holds them
    all, or the chunk has no particles of that type. For cached downloads
    (require_stamp) the type must also have been requested, since an
    unrequested type reads as zero particles too.
    """
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


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def get_json(url: str, api_key: str):
    r = requests.get(url, headers={"api-key": api_key})
    r.raise_for_status()
    return r.json()


def download_file(url: str, api_key: str, out_path: str, params: Optional[dict] = None,
                  timeout=(10, 300), max_retries: int = 8, backoff: float = 2.0):
    """
    Stream download to out_path+'.part', then atomic rename to out_path.
    Retries on transient errors. 8 attempts (sleeps 2..128 s, ~4 min in all):
    the TNG data servers return 503s and the odd 403 for longer than the
    ~30 s that 5 attempts covered, which cost 3 of 1240 chunks in the first run.
    """
    tmp_path = out_path + ".part"
    headers = {"api-key": api_key}

    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, params=params, headers=headers, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):  # 8MB
                        if chunk:
                            f.write(chunk)
            os.replace(tmp_path, out_path)  # atomic on same filesystem
            return
        except Exception as e:
            # Clean up partial
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            if attempt == max_retries:
                raise
            sleep_s = backoff ** attempt
            print(f"[warn] download failed (attempt {attempt}/{max_retries}) for {url}: {e}. sleeping {sleep_s:.1f}s",
                  flush=True)
            time.sleep(sleep_s)


def snapshot_chunk_url(sim: str, snap: int, idx: int) -> str:
    # matches what your listing shows
    return f"{BASE_URL}{sim}/files/snapshot-{snap}.{idx}.hdf5"


def chunk_local_path(cache_dir: str, sim: str, snap: int, idx: int) -> str:
    # stable: matches API naming, and tng_cutout_config.cache_chunk_path
    chunk_dir = os.path.join(cache_dir, sim, f"snap_{snap}", "snapshot_chunks")
    safe_mkdir(chunk_dir)
    return os.path.join(chunk_dir, f"snapshot-{snap}.{idx}.hdf5")


def resolve_indices(args) -> List[int]:
    indices = []

    if args.indices:
        indices.extend(parse_indices(args.indices))

    if args.range is not None:
        a, b = args.range
        indices.extend(range(a, b + 1))

    if args.index_file:
        with open(args.index_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                indices.append(int(line))

    # Stride partition over tasks. --task-id/--n-tasks take precedence, so one
    # slurm array can serve several simulations (the slurm file maps its array
    # index to (simulation, task within that simulation)); --use-slurm-array
    # reads the array environment directly, for one simulation per array.
    task_id, n_tasks = None, None
    if args.task_id is not None:
        task_id, n_tasks = args.task_id, args.n_tasks
    elif args.use_slurm_array:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        n_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", str(args.array_ntasks)))
        if n_tasks <= 0:
            n_tasks = args.array_ntasks

    if task_id is not None:
        # default full range if user didn't pass any indices/range/file
        if not indices:
            indices = list(range(args.default_min, args.default_max + 1))
        indices = [i for i in sorted(set(indices)) if (i % n_tasks) == task_id]

    return sorted(set(indices))


def main():
    ap = argparse.ArgumentParser(description="Download and cache TNG snapshot chunk files to scratch.")
    ap.add_argument("--api-key", type=str, default=None,
                    help="TNG API key. If omitted, uses env TNG_API_KEY.")
    ap.add_argument("--sim", type=str, default="TNG300-1")
    ap.add_argument("--snap", type=int, default=tcc.SNAP)
    ap.add_argument("--cache-dir", type=str, default=tcc.CHUNK_CACHE_DIR,
                    help="Base cache directory on scratch (default: tng_cutout_config.CHUNK_CACHE_DIR).")
    ap.add_argument("--fields", type=str, default=None,
                    help="Per-type fields to fetch, e.g. 'gas:Coordinates,Density;stars:Masses;bh:Masses'. "
                         "Default: tng_cutout_config.FIELDS for a configured --sim, else the whole chunk.")
    ap.add_argument("--dm-only", action="store_true",
                    help="Append dm field params to reduce content (optional).")
    ap.add_argument("--dm-fields", type=str, default="Coordinates,Velocities,ParticleIDs",
                    help="DM fields to request if --dm-only is set.")
    ap.add_argument("--indices", type=str, default=None,
                    help="Comma list with optional ranges, e.g. '0-50,77,90-100'")
    ap.add_argument("--range", type=int, nargs=2, default=None,
                    metavar=("MIN", "MAX"),
                    help="Download inclusive range MIN MAX.")
    ap.add_argument("--index-file", type=str, default=None,
                    help="Text file with one index per line.")
    ap.add_argument("--skip-valid", action="store_true",
                    help="If present and valid HDF5 exists, skip. (Default: true behavior anyway.)")
    ap.add_argument("--force-redownload", action="store_true",
                    help="Always redownload even if valid file exists.")
    ap.add_argument("--use-slurm-array", action="store_true",
                    help="Use SLURM_ARRAY_TASK_ID to stride over indices.")
    ap.add_argument("--array-ntasks", type=int, default=28,
                    help="Fallback task count if SLURM_ARRAY_TASK_COUNT not set.")
    ap.add_argument("--task-id", type=int, default=None,
                    help="This task's index within --n-tasks (overrides the slurm array environment).")
    ap.add_argument("--n-tasks", type=int, default=None,
                    help="Number of tasks the chunk range is strided over, with --task-id.")
    ap.add_argument("--default-min", type=int, default=0)
    ap.add_argument("--default-max", type=int, default=None,
                    help="Last chunk index (default: n_chunks - 1 for a configured --sim, else 599).")

    args = ap.parse_args()

    configured = args.sim in tcc.SIMS
    if args.default_max is None:
        args.default_max = tcc.SIMS[args.sim]["n_chunks"] - 1 if configured else 599
    if (args.task_id is None) != (args.n_tasks is None):
        ap.error("--task-id and --n-tasks go together")

    api_key = args.api_key or os.environ.get("TNG_API_KEY", "")
    if len(api_key) != 32:
        print("ERROR: API key missing/invalid. Provide --api-key or set env TNG_API_KEY", file=sys.stderr)
        sys.exit(2)

    indices = resolve_indices(args)
    if not indices:
        # A task can legitimately get nothing (more tasks than chunks, as for
        # TNG300-3's 16 chunks); only a non-strided call with no indices is an error.
        if args.task_id is not None or args.use_slurm_array:
            print(f"[done] sim={args.sim}: no chunks assigned to this task", flush=True)
            return
        print("ERROR: no indices resolved. Provide --indices/--range/--index-file or --use-slurm-array.",
              file=sys.stderr)
        sys.exit(2)

    # What to fetch: per-type fields (configured sims or --fields), else the
    # legacy whole-chunk / --dm-only download.
    if args.fields is not None:
        wanted = parse_fields(args.fields)
    elif configured and not args.dm_only:
        wanted = {p: tcc.FIELDS[p] for p in tcc.CHUNKED_PARTTYPES}
    else:
        wanted = None

    if wanted is None:
        legacy_params = {"dm": args.dm_fields} if args.dm_only else None
        print(f"[info] sim={args.sim} snap={args.snap} n_indices={len(indices)} cache_dir={args.cache_dir} "
              f"fields={'dm:' + args.dm_fields if args.dm_only else 'whole chunk'}", flush=True)
    else:
        sources = {p: ("existing" if configured and p in tcc.SIMS[args.sim]["existing"] else "download")
                   for p in wanted}
        print(f"[info] sim={args.sim} snap={args.snap} n_indices={len(indices)} cache_dir={args.cache_dir}",
              flush=True)
        for p, fl in wanted.items():
            print(f"[info]   {p:5s} ({sources[p]} source first): {','.join(fl)}", flush=True)

    n_ok = n_skip = n_redo = n_existing = n_fail = 0
    for idx in indices:
        url = snapshot_chunk_url(args.sim, args.snap, idx)
        out_path = chunk_local_path(args.cache_dir, args.sim, args.snap, idx)

        if wanted is None:
            # --- legacy: whole chunk or DM only, validity = opens as HDF5 ---
            params = legacy_params
            if os.path.exists(out_path):
                if args.force_redownload:
                    os.remove(out_path)
                elif is_valid_hdf5(out_path):
                    n_skip += 1
                    continue
                else:
                    # truncated/corrupt: delete and redo
                    try:
                        os.remove(out_path)
                    except Exception:
                        pass
                    n_redo += 1
            stamp = None
        else:
            # --- per-type: existing source, then cache, then download ---
            to_fetch = {}
            for p, fl in wanted.items():
                ex = tcc.existing_chunk_path(args.sim, p, idx, args.snap) if configured else None
                if ex is not None and has_fields(ex, p, fl, require_stamp=False):
                    continue
                to_fetch[p] = fl
            if not to_fetch:
                n_existing += 1
                continue
            if (not args.force_redownload and os.path.exists(out_path)
                    and all(has_fields(out_path, p, fl, require_stamp=True) for p, fl in to_fetch.items())):
                n_skip += 1
                continue
            if os.path.exists(out_path):
                # corrupt, or cached with a different field set: replace it
                try:
                    os.remove(out_path)
                except Exception:
                    pass
                n_redo += 1
            params = {tcc.API_TYPE_NAME[p]: ",".join(fl) for p, fl in to_fetch.items()}
            stamp = to_fetch

        try:
            download_file(url, api_key, out_path, params=params)
            # Validate after download
            if not is_valid_hdf5(out_path):
                raise RuntimeError("Downloaded file is not a valid HDF5 (likely truncated).")
            if stamp is not None:
                with h5py.File(out_path, "a") as f:
                    f.attrs[STAMP_ATTR] = json.dumps({p: list(fl) for p, fl in stamp.items()})
                bad = [p for p, fl in stamp.items() if not has_fields(out_path, p, fl, require_stamp=True)]
                if bad:
                    raise RuntimeError(f"downloaded chunk lacks requested fields for {bad}")
            n_ok += 1
            print(f"[ok] {idx} -> {out_path}" + (f" ({','.join(stamp)})" if stamp else ""), flush=True)
        except Exception as e:
            n_fail += 1
            print(f"[fail] idx={idx} url={url} err={e}", file=sys.stderr, flush=True)

    print(f"[done] sim={args.sim} ok={n_ok} skipped_cached={n_skip} satisfied_by_existing={n_existing} "
          f"redownloaded={n_redo} failed={n_fail}", flush=True)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
