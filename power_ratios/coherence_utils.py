"""
Catalog access, rotation and CIC deposit for the coherence analyses, with no
network access: the TNG servers are unreliable, so every TNG quantity comes
from the group catalogs already on disk, and the rotation and CIC deposit are
copied from TNG_DA.py (whose import calls the TNG API).
"""
import os

import h5py
import numpy as np


# TNG300 snapshot 99 (z = 0, a = 1): Planck 2015 h, comoving box 205 cMpc/h.
TNG300_H = 0.6774
TNG300_BOXSIZE_CKPC_H = 205000.0
TNG_CATALOG_DIRS = ("/home/habjan.e/TNG/Data/TNG_data",
                    "/projects/mccleary_group/habjan.e/TNG/Data/TNG_data")


def tng_group_field(sim, field):
    """Group/<field> for every group of `sim` at snapshot 99, from <sim>_<field>.hdf5 on disk."""
    for d in TNG_CATALOG_DIRS:
        path = os.path.join(d, f"{sim}_{field}.hdf5")
        if os.path.exists(path):
            with h5py.File(path, "r") as f:
                return f["Group"][field][:]
    raise FileNotFoundError(f"no {sim}_{field}.hdf5 in {TNG_CATALOG_DIRS}")


def tng_center_on_group(coordinates, sim, cluster_id):
    """ckpc/h snapshot coordinates -> about GroupPos of `cluster_id`, wrapped across the periodic box (ckpc/h)."""
    center = tng_group_field(sim, "GroupPos")[cluster_id].astype(np.float64)
    difpos = np.subtract(coordinates, center)
    return (difpos + 0.5 * TNG300_BOXSIZE_CKPC_H) % TNG300_BOXSIZE_CKPC_H - 0.5 * TNG300_BOXSIZE_CKPC_H


# ============================================================
# Rotation (copied from TNG_DA.rotate_to_viewing_frame)
# ============================================================

def rotate_to_viewing_frame(positions: np.ndarray,
                            velocities: np.ndarray,
                            viewing_direction: np.ndarray):
    """
    Rotate 3-D positions and velocities into a frame whose +z-axis is
    the given viewing_direction.

    Parameters
    ----------
    positions : (N, 3) array_like
        Original positions.
    velocities : (N, 3) array_like
        Original velocities.
    viewing_direction : (3,) array_like
        Non-zero vector that defines the new coordinate system.

    Returns
    -------
    pos_rot : (N, 3) ndarray
        Positions in the new frame.
    vel_rot : (N, 3) ndarray
        Velocities in the new frame.
    """

    # -- 1. Normalise viewing_direction (w) -----------------------------------
    w = np.asarray(viewing_direction, dtype=np.float64)
    if w.shape != (3,) or not np.isfinite(w).all():
        raise ValueError("viewing_direction must be a finite 3-vector.")
    w /= np.linalg.norm(w)            # unit vector along new z-axis

    # -- 2. Build an orthonormal basis (u, v, w) ------------------------------
    # Choose a helper vector 'a' that is guaranteed not to be parallel to w:
    a = np.zeros(3)
    a[np.argmin(np.abs(w))] = 1.0     # axis with smallest |component|
    u = np.cross(w, a)
    u /= np.linalg.norm(u)            # first axis in the sky-plane
    v = np.cross(w, u)                # second axis, automatically unit

    # Rotation matrix: columns are the basis vectors expressed in the old frame
    R = np.column_stack((u, v, w))    # shape (3, 3), orthonormal

    # -- 3. Rotate positions and velocities -----------------------------------
    pos_rot = positions  @ R
    vel_rot = velocities @ R

    return pos_rot, vel_rot


# ============================================================
# CIC deposit (copied from TNG_DA.deposit_cic_scalar)
# ============================================================

def _prep_grid_shape(ngrid):
    if isinstance(ngrid, int):
        Nx = Ny = Nz = int(ngrid)
    else:
        Nx, Ny, Nz = map(int, ngrid)
    if Nx <= 0 or Ny <= 0 or Nz <= 0:
        raise ValueError("ngrid must be positive")
    return Nx, Ny, Nz

def _filter_inbounds(positions, L):
    x, y, z = positions[:,0], positions[:,1], positions[:,2]
    m = (x >= -L) & (x < L) & (y >= -L) & (y < L) & (z >= -L) & (z < L)
    return m

def _cic_indices_weights_nonperiodic(p_chunk, L, N):
    Nx, Ny, Nz = N
    dx, dy, dz = (2.0*L)/Nx, (2.0*L)/Ny, (2.0*L)/Nz
    gx = (p_chunk[:,0] + L) / dx
    gy = (p_chunk[:,1] + L) / dy
    gz = (p_chunk[:,2] + L) / dz

    ix0 = np.floor(gx).astype(np.int64)
    iy0 = np.floor(gy).astype(np.int64)
    iz0 = np.floor(gz).astype(np.int64)

    fx = gx - ix0
    fy = gy - iy0
    fz = gz - iz0

    ix1 = ix0 + 1
    iy1 = iy0 + 1
    iz1 = iz0 + 1

    return (iz0, iz1), (iy0, iy1), (ix0, ix1), (fz, fy, fx), (Nz, Ny, Nx)

def _masked_add(flat, idx, w, Ntot):
    m = (idx >= 0) & (idx < Ntot)
    if np.any(m):
        np.add.at(flat, idx[m], w[m])

def _accumulate_cic_scalar_nonperiodic(grid, idxz, idxy, idxx, fzfyfx, w):
    Nz, Ny, Nx = grid.shape
    iz0, iz1 = idxz; iy0, iy1 = idxy; ix0, ix1 = idxx
    fz, fy, fx = fzfyfx

    wz0 = 1.0 - fz; wy0 = 1.0 - fy; wx0 = 1.0 - fx
    wz1 = fz;       wy1 = fy;       wx1 = fx

    w000 = wz0 * wy0 * wx0 * w
    w001 = wz0 * wy0 * wx1 * w
    w010 = wz0 * wy1 * wx0 * w
    w011 = wz0 * wy1 * wx1 * w
    w100 = wz1 * wy0 * wx0 * w
    w101 = wz1 * wy0 * wx1 * w
    w110 = wz1 * wy1 * wx0 * w
    w111 = wz1 * wy1 * wx1 * w

    flat = grid.ravel()
    stride_y = Nx
    stride_z = Ny * Nx
    Ntot = flat.size

    idx000 = iz0 * stride_z + iy0 * stride_y + ix0
    idx001 = iz0 * stride_z + iy0 * stride_y + ix1
    idx010 = iz0 * stride_z + iy1 * stride_y + ix0
    idx011 = iz0 * stride_z + iy1 * stride_y + ix1
    idx100 = iz1 * stride_z + iy0 * stride_y + ix0
    idx101 = iz1 * stride_z + iy0 * stride_y + ix1
    idx110 = iz1 * stride_z + iy1 * stride_y + ix0
    idx111 = iz1 * stride_z + iy1 * stride_y + ix1

    mz0 = (iz0 >= 0) & (iz0 < Nz); mz1 = (iz1 >= 0) & (iz1 < Nz)
    my0 = (iy0 >= 0) & (iy0 < Ny); my1 = (iy1 >= 0) & (iy1 < Ny)
    mx0 = (ix0 >= 0) & (ix0 < Nx); mx1 = (ix1 >= 0) & (ix1 < Nx)

    m000 = mz0 & my0 & mx0
    m001 = mz0 & my0 & mx1
    m010 = mz0 & my1 & mx0
    m011 = mz0 & my1 & mx1
    m100 = mz1 & my0 & mx0
    m101 = mz1 & my0 & mx1
    m110 = mz1 & my1 & mx0
    m111 = mz1 & my1 & mx1

    _masked_add(flat, idx000[m000], w000[m000], Ntot)
    _masked_add(flat, idx001[m001], w001[m001], Ntot)
    _masked_add(flat, idx010[m010], w010[m010], Ntot)
    _masked_add(flat, idx011[m011], w011[m011], Ntot)
    _masked_add(flat, idx100[m100], w100[m100], Ntot)
    _masked_add(flat, idx101[m101], w101[m101], Ntot)
    _masked_add(flat, idx110[m110], w110[m110], Ntot)
    _masked_add(flat, idx111[m111], w111[m111], Ntot)

def deposit_cic_scalar(positions, L, ngrid, weights=None, chunksize=1_000_000, dtype=np.float64):
    Nx, Ny, Nz = _prep_grid_shape(ngrid)
    grid = np.zeros((Nz, Ny, Nx), dtype=dtype)

    m = _filter_inbounds(positions, L)
    p = positions[m]
    if weights is None:
        w = np.ones(p.shape[0], dtype=dtype)
    else:
        w_full = np.asarray(weights, dtype=dtype)
        if w_full.shape[0] != positions.shape[0]:
            raise ValueError("weights must match positions length")
        w = w_full[m]

    N = p.shape[0]
    start = 0
    while start < N:
        end = min(start + chunksize, N)
        pc = p[start:end]
        wc = w[start:end]
        idxz, idxy, idxx, fzfyfx, _ = _cic_indices_weights_nonperiodic(pc, L, (Nx, Ny, Nz))
        _accumulate_cic_scalar_nonperiodic(grid, idxz, idxy, idxx, fzfyfx, wc)
        start = end
    return grid
