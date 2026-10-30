"""
Adaptive-SPH mass cubes and projected maps for the coherence analyses.

Replaces the CIC deposit (TNG_DA.deposit_cic_scalar) of the coherence_* scripts.
Adapted from cluster_deprojection/dataset/density_utils.py, with one difference:
every field here is a single species, so h_j is the distance from particle j to
its n_ngb-th nearest neighbour of the SAME species (dark matter or gas), not of
all species together. The field is

    rho(x) = sum_j m_j W(|x - x_j|, h_j),

with W the Gadget cubic spline of compact support h_j, floored at the
spline-equivalent of the gravitational softening (2.8 x Plummer softening).

Units are those of the inputs: the scripts pass kpc and Msun, so h is kpc, and
the deposits return Msun per cell (voxel or pixel) -- the same convention as
deposit_cic_scalar, not a density.

  sph_mass_cube : the voxel integral of rho on an N^3 grid over [-L, L]^3, zyx.
                  Each kernel is sampled on a sub-voxel grid aligned to the
                  voxel edges (>= sub samples across its support) and the
                  samples are renormalised to the particle's mass, so mass is
                  conserved particle by particle.
  sph_mass_map  : the same field integrated over |z| < L and over each pixel of
                  an N^2 grid (yx), i.e. sph_mass_cube(...).sum(axis=0), but
                  without building the cube: the kernel's line-of-sight
                  integral between the cube's z faces is read from a table, so
                  a projection costs ~(h/dx)^2 per particle instead of (h/dx)^3.

h_j depends only on distances between particles, which a rotation leaves
unchanged, so it is computed once per (simulation, cluster, species) and cached
in CACHE_ROOT; the 3D script builds the cache and the 2D script reuses it.
"""
import math
import os

import numba as nb
import numpy as np
from scipy.spatial import cKDTree


N_NGB = 64                         # h_j = distance to the N_NGB-th same-species neighbour
SUB = 6.0                          # sub-samples across a kernel's support
CACHE_ROOT = "/scratch/habjan.e/fourier_analysis_sph"

# Physical Plummer-equivalent softening at z = 0, kpc (Pillepich+ 2018 for
# TNG300; 4 kpc/h for BAHAMAS, Robertson+ 2019), as in cluster_deprojection.
TNG300_SIMULATIONS = ("TNG300-1", "TNG300-2", "TNG300-3")
BAHAMAS_MODELS = ("CDMb", "SIDM0.1b", "SIDM0.3b", "SIDM1b", "vdSIDMb")
SOFTENING_KPC = {"TNG300-1": 1.48, "TNG300-2": 2.95, "TNG300-3": 5.90,
                 **{m: 5.7 for m in BAHAMAS_MODELS}}


def h_floor_kpc(sim):
    """Floor of the smoothing lengths for `sim` (a TNG300 run or BAHAMAS model): 2.8 x softening."""
    return 2.8 * SOFTENING_KPC[sim]


def n_threads():
    """Cores allotted to this job (1 outside SLURM)."""
    return int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))


# ============================================================
# Smoothing lengths
# ============================================================

def smoothing_lengths(pos, n_ngb, h_floor, workers=1):
    """
    Per-particle SPH support h_j: distance to the n_ngb-th nearest neighbour
    among the particles in `pos`, floored at h_floor (units of pos).

    pos must hold every particle of the species in the cutout, not just those
    inside the cube, or particles near the cube's faces would see only half
    their neighbours and get inflated kernels.
    """
    tree = cKDTree(pos)
    # k = n_ngb + 1: the nearest 'neighbour' at distance 0 is the particle itself.
    dist, _ = tree.query(pos, k=[n_ngb + 1], workers=workers)
    return np.maximum(dist[:, 0], h_floor)


def cache_path(sim, species, cluster_id):
    """<CACHE_ROOT>/<TNG300|BAHAMAS>/<sim>/<species>_<cluster_id>_ngb<N_NGB>.npz"""
    suite = "TNG300" if sim in TNG300_SIMULATIONS else "BAHAMAS"
    return os.path.join(CACHE_ROOT, suite, sim, f"{species}_{cluster_id}_ngb{N_NGB}.npz")


def species_h(pos, sim, species, cluster_id, workers=None):
    """
    smoothing_lengths() of one species of one cluster (pos in kpc), read from
    the cache if a matching file is there and written to it otherwise.

    The particle order of `pos` must be the same on every call (the scripts
    read the cutouts in file order). A cached file whose particle count, n_ngb
    or floor disagrees is recomputed rather than trusted. Written to a
    temporary name and renamed, so a killed job never leaves a truncated cache.
    """
    workers = n_threads() if workers is None else workers
    path = cache_path(sim, species, cluster_id)
    floor = h_floor_kpc(sim)
    if os.path.exists(path):
        try:
            with np.load(path) as c:
                if (int(c["n_particles"]) == pos.shape[0]
                        and int(c["n_ngb"]) == N_NGB
                        and float(c["h_floor_kpc"]) == float(floor)):
                    print(f"SPH h: read {path}")
                    return c["h"].astype(np.float64)
        except (OSError, KeyError, ValueError):
            pass                                  # unreadable: recompute

    print(f"SPH h: computing {species} ({pos.shape[0]} particles, {workers} workers)")
    h = smoothing_lengths(pos, N_NGB, floor, workers=workers)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}.npz"
    np.savez(tmp, h=h.astype(np.float32), n_particles=np.int64(pos.shape[0]),
             n_ngb=np.int64(N_NGB), h_floor_kpc=np.float64(floor), species=species)
    os.replace(tmp, path)
    print(f"SPH h: wrote {path}")
    # float32 on disk; return what a cache hit would, so the two paths agree.
    return h.astype(np.float32).astype(np.float64)


# ============================================================
# Kernel and its line-of-sight integrals
# ============================================================

@nb.njit(inline="always", cache=True)
def _spline(q):
    """Gadget cubic spline in q = r / h, up to normalisation (renormalised per particle)."""
    if q < 0.5:
        return 1.0 - 6.0 * q * q + 6.0 * q * q * q
    if q < 1.0:
        t = 1.0 - q
        return 2.0 * t * t * t
    return 0.0


def _spline_np(q):
    return np.where(q < 0.5, 1.0 - 6.0 * q**2 + 6.0 * q**3,
                    np.where(q < 1.0, 2.0 * (1.0 - q)**3, 0.0))


def _column_table(n_r=513, n_t=513, refine=8):
    """
    C[i, k] = int_0^{t_k} W(sqrt(R_i^2 + u^2)) du, R_i, t_k on [0, 1] (units of h):
    the kernel's line-of-sight integral at projected radius R from the
    midplane to height t.
    """
    R = np.linspace(0.0, 1.0, n_r)[:, None]
    u = np.linspace(0.0, 1.0, (n_t - 1) * refine + 1)[None, :]
    w = _spline_np(np.sqrt(R**2 + u**2))
    du = u[0, 1] - u[0, 0]
    cum = np.concatenate([np.zeros((n_r, 1)),
                          np.cumsum(0.5 * (w[:, 1:] + w[:, :-1]) * du, axis=1)], axis=1)
    return np.ascontiguousarray(cum[:, ::refine])


def _marginal_table(n_t=1025, refine=8):
    """
    M[k] = fraction of a kernel's mass at height u < t_k, t_k on [-1, 1]. The
    marginal density in u is 2 pi int_|u|^1 W(r) r dr.
    """
    r = np.linspace(0.0, 1.0, (n_t - 1) * refine + 1)
    f = _spline_np(r) * r
    dr = r[1] - r[0]
    # G(a) = int_a^1 W(r) r dr
    G = np.concatenate([np.cumsum((0.5 * (f[1:] + f[:-1]) * dr)[::-1])[::-1], [0.0]])
    u = np.linspace(-1.0, 1.0, 2 * (n_t - 1) * refine + 1)
    marg = np.interp(np.abs(u), r, G)
    du = u[1] - u[0]
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (marg[1:] + marg[:-1]) * du)])
    cum /= cum[-1]
    return np.ascontiguousarray(np.interp(np.linspace(-1.0, 1.0, n_t), u, cum))


_COLUMN = _column_table()
_MARGINAL = _marginal_table()


@nb.njit(inline="always", cache=True)
def _column(ctab, qR, t):
    """Signed int_0^t W(sqrt(qR^2 + u^2)) du for qR in [0, 1], t in [-1, 1], bilinear in the table."""
    a = abs(t)
    nr, nt = ctab.shape
    fr = qR * (nr - 1)
    i = min(int(fr), nr - 2)
    wr = fr - i
    ft = a * (nt - 1)
    k = min(int(ft), nt - 2)
    wt = ft - k
    c = ((1.0 - wr) * ((1.0 - wt) * ctab[i, k] + wt * ctab[i, k + 1])
         + wr * ((1.0 - wt) * ctab[i + 1, k] + wt * ctab[i + 1, k + 1]))
    return c if t >= 0.0 else -c


@nb.njit(inline="always", cache=True)
def _marginal(mtab, t):
    """Fraction of a kernel's mass below height t in [-1, 1] (units of h)."""
    n = mtab.shape[0]
    f = (t + 1.0) * 0.5 * (n - 1)
    k = min(max(int(f), 0), n - 2)
    w = f - k
    return (1.0 - w) * mtab[k] + w * mtab[k + 1]


# ============================================================
# Deposits
# ============================================================

@nb.njit(parallel=True, cache=True)
def _deposit_spline(pos, m, h, N, L, sub, nth):
    """Mass per voxel (zyx) and the mass falling outside [-L, L]^3."""
    dx = 2.0 * L / N
    grids = np.zeros((nth, N, N, N))
    outside = np.zeros(nth)
    n = pos.shape[0]
    for t in nb.prange(nth):
        g = grids[t]
        for j in range(t, n, nth):
            x, y, z, hj = pos[j, 0], pos[j, 1], pos[j, 2], h[j]
            if (x + hj < -L or x - hj > L or y + hj < -L or y - hj > L
                    or z + hj < -L or z - hj > L):
                outside[t] += m[j]
                continue
            # Support entirely inside one voxel: the voxel integral is the
            # whole mass, exactly what the sub-sampling below would give.
            vx0 = int(math.floor((x - hj + L) / dx)); vy0 = int(math.floor((y - hj + L) / dx))
            vz0 = int(math.floor((z - hj + L) / dx))
            if (vx0 == int(math.floor((x + hj + L) / dx))
                    and vy0 == int(math.floor((y + hj + L) / dx))
                    and vz0 == int(math.floor((z + hj + L) / dx))):
                if 0 <= vx0 < N and 0 <= vy0 < N and 0 <= vz0 < N:
                    g[vz0, vy0, vx0] += m[j]
                else:
                    outside[t] += m[j]
                continue
            s = max(1, int(math.ceil(sub * dx / hj)))
            d = dx / s
            i0 = int(math.floor((x - hj + L) / d)); i1 = int(math.floor((x + hj + L) / d))
            j0 = int(math.floor((y - hj + L) / d)); j1 = int(math.floor((y + hj + L) / d))
            k0 = int(math.floor((z - hj + L) / d)); k1 = int(math.floor((z + hj + L) / d))
            ih2 = 1.0 / (hj * hj)
            wsum = 0.0
            for a in range(i0, i1 + 1):
                ddx = -L + (a + 0.5) * d - x
                for b in range(j0, j1 + 1):
                    ddy = -L + (b + 0.5) * d - y
                    for c in range(k0, k1 + 1):
                        ddz = -L + (c + 0.5) * d - z
                        q2 = (ddx * ddx + ddy * ddy + ddz * ddz) * ih2
                        if q2 < 1.0:
                            wsum += _spline(math.sqrt(q2))
            if wsum <= 0.0:                       # unresolved: whole mass at the centre
                vx = int(math.floor((x + L) / dx)); vy = int(math.floor((y + L) / dx))
                vz = int(math.floor((z + L) / dx))
                if 0 <= vx < N and 0 <= vy < N and 0 <= vz < N:
                    g[vz, vy, vx] += m[j]
                else:
                    outside[t] += m[j]
                continue
            norm = m[j] / wsum
            for a in range(i0, i1 + 1):
                ddx = -L + (a + 0.5) * d - x
                vx = a // s
                for b in range(j0, j1 + 1):
                    ddy = -L + (b + 0.5) * d - y
                    vy = b // s
                    for c in range(k0, k1 + 1):
                        ddz = -L + (c + 0.5) * d - z
                        q2 = (ddx * ddx + ddy * ddy + ddz * ddz) * ih2
                        if q2 < 1.0:
                            w = norm * _spline(math.sqrt(q2))
                            vz = c // s
                            if 0 <= vx < N and 0 <= vy < N and 0 <= vz < N:
                                g[vz, vy, vx] += w
                            else:
                                outside[t] += w
    return grids.sum(axis=0), outside.sum()


@nb.njit(parallel=True, cache=True)
def _deposit_spline_2d(pos, m, h, N, L, sub, nth, ctab, mtab):
    """Mass per pixel (yx) of the field within |z| < L, and the mass outside it."""
    dx = 2.0 * L / N
    grids = np.zeros((nth, N, N))
    outside = np.zeros(nth)
    n = pos.shape[0]
    for t in nb.prange(nth):
        g = grids[t]
        for j in range(t, n, nth):
            x, y, z, hj = pos[j, 0], pos[j, 1], pos[j, 2], h[j]
            if (x + hj < -L or x - hj > L or y + hj < -L or y - hj > L
                    or z + hj < -L or z - hj > L):
                outside[t] += m[j]
                continue
            # The kernel's extent between the cube's z faces, in units of h_j.
            ih = 1.0 / hj
            t1 = max(-1.0, (-L - z) * ih)
            t2 = min(1.0, (L - z) * ih)
            # Support inside one pixel (or unresolved below): the pixel gets
            # the kernel's mass between the z faces, from the marginal table.
            px0 = int(math.floor((x - hj + L) / dx)); py0 = int(math.floor((y - hj + L) / dx))
            single = (px0 == int(math.floor((x + hj + L) / dx))
                      and py0 == int(math.floor((y + hj + L) / dx)))
            s = max(1, int(math.ceil(sub * dx / hj)))
            d = dx / s
            i0 = int(math.floor((x - hj + L) / d)); i1 = int(math.floor((x + hj + L) / d))
            j0 = int(math.floor((y - hj + L) / d)); j1 = int(math.floor((y + hj + L) / d))
            wsum = 0.0
            if not single:
                for a in range(i0, i1 + 1):
                    ddx = (-L + (a + 0.5) * d - x) * ih
                    for b in range(j0, j1 + 1):
                        ddy = (-L + (b + 0.5) * d - y) * ih
                        q2 = ddx * ddx + ddy * ddy
                        if q2 < 1.0:
                            wsum += 2.0 * _column(ctab, math.sqrt(q2), 1.0)
            if single or wsum <= 0.0:
                px = int(math.floor((x + L) / dx)); py = int(math.floor((y + L) / dx))
                f = _marginal(mtab, t2) - _marginal(mtab, t1)
                if 0 <= px < N and 0 <= py < N:
                    g[py, px] += m[j] * f
                    outside[t] += m[j] * (1.0 - f)
                else:
                    outside[t] += m[j]
                continue
            # Renormalised to the full column (both z faces open), so the
            # mass beyond the faces is tallied as outside, as in the 3D deposit.
            norm = m[j] / wsum
            dep = 0.0
            for a in range(i0, i1 + 1):
                ddx = (-L + (a + 0.5) * d - x) * ih
                px = a // s
                for b in range(j0, j1 + 1):
                    ddy = (-L + (b + 0.5) * d - y) * ih
                    q2 = ddx * ddx + ddy * ddy
                    if q2 < 1.0:
                        py = b // s
                        if 0 <= px < N and 0 <= py < N:
                            qR = math.sqrt(q2)
                            w = norm * (_column(ctab, qR, t2) - _column(ctab, qR, t1))
                            g[py, px] += w
                            dep += w
            outside[t] += m[j] - dep
    return grids.sum(axis=0), outside.sum()


def sph_mass_cube(pos, mass, h, L, N, sub=SUB, threads=None):
    """
    SPH mass per voxel on an N^3 grid over [-L, L]^3, zyx order (the layout of
    deposit_cic_scalar). pos (n,3) in the viewing frame, mass (n,), h (n,) from
    species_h, all in the units of L. Returns (cube, mass outside the cube).
    """
    threads = n_threads() if threads is None else threads
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    mass = np.ascontiguousarray(mass, dtype=np.float64)
    h = np.ascontiguousarray(h, dtype=np.float64)
    cube, outside = _deposit_spline(pos, mass, h, int(N), float(L), float(sub), int(threads))
    return cube, float(outside)


def sph_mass_map(pos, mass, h, L, N, sub=SUB, threads=None):
    """
    SPH mass per pixel of the field within |z| < L, projected along z onto an
    N^2 grid over [-L, L]^2, yx order: sph_mass_cube(...).sum(axis=0) without
    the cube. Returns (map, mass outside the cube).
    """
    threads = n_threads() if threads is None else threads
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    mass = np.ascontiguousarray(mass, dtype=np.float64)
    h = np.ascontiguousarray(h, dtype=np.float64)
    image, outside = _deposit_spline_2d(pos, mass, h, int(N), float(L), float(sub),
                                        int(threads), _COLUMN, _MARGINAL)
    return image, float(outside)
