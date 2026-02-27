#!/usr/bin/env python3
import os
import sys
import argparse
import time
import requests
import h5py
from typing import List, Optional

BASE_URL = "http://www.tng-project.org/api/"

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

def is_valid_hdf5(path: str) -> bool:
    try:
        with h5py.File(path, "r"):
            return True
    except Exception:
        return False

def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def get_json(url: str, api_key: str):
    r = requests.get(url, headers={"api-key": api_key})
    r.raise_for_status()
    return r.json()

def download_file(url: str, api_key: str, out_path: str, params: Optional[dict] = None,
                  timeout=(10, 300), max_retries: int = 5, backoff: float = 2.0):
    """
    Stream download to out_path+'.part', then atomic rename to out_path.
    Retries on transient errors.
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
    # stable: matches API naming
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

    # SLURM array mode: each task downloads a strided subset
    if args.use_slurm_array:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        n_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", str(args.array_ntasks)))
        if n_tasks <= 0:
            n_tasks = args.array_ntasks
        # default full range if user didn’t pass any indices/range/file
        if not indices:
            indices = list(range(args.default_min, args.default_max + 1))
        # stride partition
        indices = [i for i in sorted(set(indices)) if (i % n_tasks) == task_id]

    indices = sorted(set(indices))
    return indices

def main():
    ap = argparse.ArgumentParser(description="Download and cache TNG snapshot chunk files to scratch.")
    ap.add_argument("--api-key", type=str, default=None,
                    help="TNG API key. If omitted, uses env TNG_API_KEY.")
    ap.add_argument("--sim", type=str, default="TNG300-1")
    ap.add_argument("--snap", type=int, default=99)
    ap.add_argument("--cache-dir", type=str, required=True,
                    help="Base cache directory on scratch.")
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
    ap.add_argument("--default-min", type=int, default=0)
    ap.add_argument("--default-max", type=int, default=599)

    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("TNG_API_KEY", "")
    if len(api_key) != 32:
        print("ERROR: API key missing/invalid. Provide --api-key or set env TNG_API_KEY", file=sys.stderr)
        sys.exit(2)

    indices = resolve_indices(args)
    if not indices:
        print("ERROR: no indices resolved. Provide --indices/--range/--index-file or --use-slurm-array.",
              file=sys.stderr)
        sys.exit(2)

    params = None
    if args.dm_only:
        params = {"dm": args.dm_fields}

    print(f"[info] sim={args.sim} snap={args.snap} n_indices={len(indices)} cache_dir={args.cache_dir}", flush=True)

    n_ok = 0
    n_skip = 0
    n_redo = 0
    for idx in indices:
        url = snapshot_chunk_url(args.sim, args.snap, idx)
        out_path = chunk_local_path(args.cache_dir, args.sim, args.snap, idx)

        if os.path.exists(out_path):
            if args.force_redownload:
                os.remove(out_path)
            else:
                # validate file
                if is_valid_hdf5(out_path):
                    n_skip += 1
                    continue
                else:
                    # truncated/corrupt: delete and redo
                    try:
                        os.remove(out_path)
                    except Exception:
                        pass
                    n_redo += 1

        try:
            download_file(url, api_key, out_path, params=params)
            # Validate after download
            if not is_valid_hdf5(out_path):
                raise RuntimeError("Downloaded file is not a valid HDF5 (likely truncated).")
            n_ok += 1
            print(f"[ok] {idx} -> {out_path}", flush=True)
        except Exception as e:
            print(f"[fail] idx={idx} url={url} err={e}", file=sys.stderr, flush=True)

    print(f"[done] ok={n_ok} skipped={n_skip} redownloaded_corrupt={n_redo}", flush=True)

if __name__ == "__main__":
    main()