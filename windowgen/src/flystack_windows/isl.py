"""Inter-satellite link (ISL) windows from line of sight and range.

brahe's access search is satellite-to-ground only, so ISL visibility is scanned here from brahe
ECI states.
"""

from __future__ import annotations

import brahe as bh
import numpy as np
import pandas as pd

from .constellation import Satellite, ensure_eop, to_epoch
from .spec import ConstellationSpec, IslSpec

ISL_COLUMNS = ["plane_1", "sat_1", "plane_2", "sat_2", "start_s", "end_s"]

# Pair-samples evaluated per chunk; bounds peak memory at a few hundred MB.
_CHUNK_PAIR_SAMPLES = 2_000_000


def link_margin_m(r1: np.ndarray, r2: np.ndarray, isl: IslSpec) -> np.ndarray:
    """
        link_margin_m(r1, r2, isl) -> ndarray

    Signed link margin between two sets of ECI positions; the link is up where it is positive.

      - `r1`, `r2` — ECI positions of the two ends, same shape (..., 3) (m)
      - `isl` — link model (grazing altitude, optional max range)

    Returns the smaller of the occlusion margin and (`max_range - |r2 - r1|`), shape (...) (m).
    The occlusion margin is the closest approach of the r1-r2 segment to Earth's centre minus
    the occluding radius. For "wgs84" the ellipsoid inflated by the grazing altitude is mapped to
    a sphere by stretching z by (a + h) / (b + h), so the margin is measured in that stretched
    frame; it is exact in sign, which is all the window search uses.
    """
    h = isl.grazing_altitude_km * 1e3
    if isl.earth_model == "wgs84":
        a = bh.WGS84_A + h
        stretch = np.array([1.0, 1.0, a / (bh.WGS84_A * (1.0 - bh.WGS84_F) + h)])
        p1, p2 = r1 * stretch, r2 * stretch
    else:
        a = bh.R_EARTH + h
        p1, p2 = r1, r2
    d = p2 - p1
    dd = np.einsum("...k,...k->...", d, d)
    s = np.clip(-np.einsum("...k,...k->...", p1, d) / np.where(dd > 0, dd, 1.0), 0.0, 1.0)
    margin = np.linalg.norm(p1 + s[..., None] * d, axis=-1) - a
    if isl.max_range_km is not None:
        rng = np.linalg.norm(r2 - r1, axis=-1)
        margin = np.minimum(margin, isl.max_range_km * 1e3 - rng)
    return margin


def _positions(sats: list[Satellite], t0: bh.Epoch, times_s: np.ndarray) -> np.ndarray:
    """ECI positions of every satellite at `times_s` (s after `t0`), shape (n_sats, n_times, 3) (m)."""
    epochs = [t0 + float(t) for t in times_s]
    return np.stack([np.asarray(s.propagator.states_eci(epochs))[:, :3] for s in sats])


def _crossing(ta, tb, ma, mb):
    """Time at which the margin interpolated linearly from (ta, ma) to (tb, mb) crosses zero."""
    frac = np.clip(ma / (ma - mb), 0.0, 1.0)
    return ta + frac * (tb - ta)


def isl_windows(spec: ConstellationSpec, sats: list[Satellite], isl: IslSpec) -> pd.DataFrame:
    """
        isl_windows(spec, sats, isl) -> DataFrame

    Scan every satellite pair for intervals where `link_margin_m` is positive.

      - `spec` — constellation settings (scenario start and duration)
      - `sats` — satellites from `build_constellation(spec)`
      - `isl` — link model and scan step

    Returns one row per link window with columns `plane_1`, `sat_1`, `plane_2`, `sat_2`
    (1-indexed) and `start_s`, `end_s` (s since scenario start), sorted by `start_s`. A link up
    for the whole scenario has `start_s == 0` and `end_s == spec.duration_s`.
    """
    ensure_eop()
    t0 = to_epoch(spec)
    T = float(spec.duration_s)
    times = np.arange(0.0, T, isl.grid_step_s)
    times = np.append(times, T)
    ii, jj = np.triu_indices(len(sats), k=1)
    n_pairs = len(ii)
    chunk = max(16, _CHUNK_PAIR_SAMPLES // max(n_pairs, 1))

    open_at = np.full(n_pairs, np.nan)
    prev_t, prev_m = None, None
    starts, ends, pair_idx = [], [], []

    for c0 in range(0, len(times), chunk):
        tc = times[c0:c0 + chunk]
        R = _positions(sats, t0, tc)
        m = link_margin_m(R[ii], R[jj], isl)                     # (n_pairs, len(tc))
        if prev_m is None:
            up0 = m[:, 0] > 0
            open_at[up0] = 0.0
            t_ext, m_ext = tc, m
        else:
            t_ext = np.concatenate([[prev_t], tc])
            m_ext = np.concatenate([prev_m[:, None], m], axis=1)
        up = m_ext > 0
        change = up[:, 1:] != up[:, :-1]
        p, k = np.nonzero(change)                                # k indexes the left sample
        order = np.lexsort((k, p))
        p, k = p[order], k[order]
        t_cross = _crossing(t_ext[k], t_ext[k + 1], m_ext[p, k], m_ext[p, k + 1])
        rising = up[p, k + 1]
        for pi, tx, r in zip(p.tolist(), t_cross.tolist(), rising.tolist()):
            if r:
                open_at[pi] = tx
            else:
                starts.append(open_at[pi]); ends.append(tx); pair_idx.append(pi)
                open_at[pi] = np.nan
        prev_t, prev_m = tc[-1], m[:, -1]

    still_open = np.nonzero(~np.isnan(open_at))[0]
    for pi in still_open.tolist():
        starts.append(open_at[pi]); ends.append(T); pair_idx.append(pi)

    pair_idx = np.asarray(pair_idx, dtype=int)
    planes = np.array([s.plane for s in sats]); nums = np.array([s.sat for s in sats])
    a, b = ii[pair_idx], jj[pair_idx]
    df = pd.DataFrame({
        "plane_1": planes[a], "sat_1": nums[a], "plane_2": planes[b], "sat_2": nums[b],
        "start_s": np.asarray(starts, dtype=float), "end_s": np.asarray(ends, dtype=float),
    })
    if isl.both_directions:
        swapped = df.rename(columns={"plane_1": "plane_2", "sat_1": "sat_2",
                                     "plane_2": "plane_1", "sat_2": "sat_1"})[ISL_COLUMNS]
        df = pd.concat([df, swapped], ignore_index=True)
    return df.sort_values(["start_s", "plane_1", "sat_1", "plane_2", "sat_2"],
                          kind="stable").reset_index(drop=True)
