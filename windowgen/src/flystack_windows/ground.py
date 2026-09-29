"""Satellite-to-ground-station contact windows."""

from __future__ import annotations

import brahe as bh
import brahe.datasets as bhd
import pandas as pd

from .constellation import Satellite, ensure_eop, parse_name, to_epoch
from .spec import ConstellationSpec, GroundSpec

GROUND_COLUMNS = ["plane", "sat", "station", "start_s", "end_s"]


def load_stations(ground: GroundSpec) -> list[bh.PointLocation]:
    """
        load_stations(ground) -> list[PointLocation]

    Collect ground stations from the GeoJSON file and brahe providers named in `ground`.

      - `ground` — ground-segment settings

    Returns the stations, filtered to `ground.include` when that is set.

    NOTE: station names must be unique; they become the `ground_station` column.
    """
    stations = []
    if ground.stations_file is not None:
        stations += bhd.groundstations_load_from_file(str(ground.stations_file))
    for provider in ground.providers:
        stations += bhd.groundstations_load(provider)
    if ground.include is not None:
        wanted = set(ground.include)
        stations = [s for s in stations if s.get_name() in wanted]
        missing = wanted - {s.get_name() for s in stations}
        if missing:
            raise ValueError(f"stations not found: {sorted(missing)}")
    names = [s.get_name() for s in stations]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"duplicate station names: {sorted(dupes)}")
    if not stations:
        raise ValueError("no ground stations configured")
    return stations


def ground_windows(spec: ConstellationSpec, sats: list[Satellite], ground: GroundSpec) -> pd.DataFrame:
    """
        ground_windows(spec, sats, ground) -> DataFrame

    Compute every satellite-to-station contact over the scenario with brahe `location_accesses`.

      - `spec` — constellation settings (supplies the scenario start and duration)
      - `sats` — satellites from `build_constellation(spec)`
      - `ground` — stations and minimum elevation

    Returns one row per pass with columns `plane`, `sat` (1-indexed), `station`, and
    `start_s`, `end_s` (s since scenario start), sorted by `start_s`.
    """
    ensure_eop()
    t0 = to_epoch(spec)
    stations = load_stations(ground)
    windows = bh.location_accesses(
        stations, [s.propagator for s in sats], t0, t0 + spec.duration_s,
        bh.ElevationConstraint(min_elevation_deg=ground.min_elevation_deg))
    rows = []
    for w in windows:
        plane, sat = parse_name(w.satellite_name)
        rows.append((plane, sat, w.location_name, w.window_open - t0, w.window_close - t0))
    df = pd.DataFrame(rows, columns=GROUND_COLUMNS)
    return df.sort_values(["start_s", "plane", "sat", "station"], kind="stable").reset_index(drop=True)
