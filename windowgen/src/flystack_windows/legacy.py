"""Write windows in the STK-export CSV layout read by `project/fed/strategies`."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .spec import ConstellationSpec

GROUND_LEGACY_COLUMNS = [
    "index", "Duration (sec)", "Start Time Seconds Cumulative", "End Time Seconds Cumulative",
    "Start Time Seconds datetime", "End Time Seconds datetime", "ground_station", "cluster_num", "sat_num",
]
ISL_LEGACY_COLUMNS = [
    "index", "Duration (sec)", "Start Time Seconds Cumulative", "End Time Seconds Cumulative",
    "Start Time Seconds datetime", "End Time Seconds datetime",
    "cluster_num_1", "sat_num_1", "cluster_num_2", "sat_num_2",
]


def pseudo_seconds(dt: datetime) -> float:
    """
        pseudo_seconds(dt) -> float

    The strategies' "cumulative seconds" clock for a UTC datetime: 365-day years since 1970 plus
    day-of-year, the same arithmetic as `time_to_int` in `project/fed/strategies/utils.py`.

      - `dt` — UTC datetime

    Returns pseudo-seconds (s). 2024-04-14 16:00 UTC maps to 1711987200, the start time
    hard-coded in the AutoFLSat strategies.

    NOTE: this is not Unix time; it ignores leap days, so it only advances monotonically within
    a calendar year of a leap year. `to_legacy_*` therefore offsets from the scenario start
    instead of converting each timestamp.
    """
    return (dt.microsecond * 1e-6 + dt.second + dt.minute * 60 + dt.hour * 3600
            + (dt.timetuple().tm_yday - 1) * 86400 + (dt.year - 1970) * 86400 * 365)


def _time_columns(spec: ConstellationSpec, start_s: np.ndarray, end_s: np.ndarray) -> dict:
    """Legacy time columns for windows given in seconds since scenario start."""
    base = pseudo_seconds(spec.epoch_utc)
    start_s = np.round(np.asarray(start_s, dtype=float), 3)
    end_s = np.round(np.asarray(end_s, dtype=float), 3)
    epoch = spec.epoch_utc.replace(tzinfo=None)
    fmt = lambda s: (epoch + timedelta(seconds=float(s))).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return {
        "Duration (sec)": np.round(end_s - start_s, 3),
        "Start Time Seconds Cumulative": base + start_s,
        "End Time Seconds Cumulative": base + end_s,
        "Start Time Seconds datetime": [fmt(s) for s in start_s],
        "End Time Seconds datetime": [fmt(s) for s in end_s],
    }


def to_legacy_ground(spec: ConstellationSpec, windows: pd.DataFrame) -> pd.DataFrame:
    """
        to_legacy_ground(spec, windows) -> DataFrame

    Convert `ground_windows` output to the STK ground-station CSV layout.

      - `spec` — constellation settings (scenario start)
      - `windows` — DataFrame with `plane`, `sat`, `station`, `start_s`, `end_s`

    Returns a DataFrame with `GROUND_LEGACY_COLUMNS`, sorted by start time.
    """
    w = windows.sort_values("start_s", kind="stable").reset_index(drop=True)
    df = pd.DataFrame({"index": np.arange(len(w)), **_time_columns(spec, w.start_s, w.end_s),
                       "ground_station": w.station.values,
                       "cluster_num": w.plane.values, "sat_num": w.sat.values})
    return df[GROUND_LEGACY_COLUMNS]


def to_legacy_isl(spec: ConstellationSpec, windows: pd.DataFrame) -> pd.DataFrame:
    """
        to_legacy_isl(spec, windows) -> DataFrame

    Convert `isl_windows` output to the STK `_inter` CSV layout.

      - `spec` — constellation settings (scenario start)
      - `windows` — DataFrame with `plane_1`, `sat_1`, `plane_2`, `sat_2`, `start_s`, `end_s`

    Returns a DataFrame with `ISL_LEGACY_COLUMNS`, sorted by start time.

    NOTE: a link up for the whole scenario gets `Duration (sec)` equal to the scenario length.
    `choose_sat_csv_auto` drops rows whose duration is exactly 7862400 s (91 days), so with a
    91-day scenario those always-on links are filtered as they were for STK.
    """
    w = windows.sort_values("start_s", kind="stable").reset_index(drop=True)
    df = pd.DataFrame({"index": np.arange(len(w)), **_time_columns(spec, w.start_s, w.end_s),
                       "cluster_num_1": w.plane_1.values, "sat_num_1": w.sat_1.values,
                       "cluster_num_2": w.plane_2.values, "sat_num_2": w.sat_2.values})
    return df[ISL_LEGACY_COLUMNS]


def legacy_filename(spec: ConstellationSpec, tag: str, isl: bool) -> str:
    """
        legacy_filename(spec, tag, isl) -> str

    File name `{S}s_{P}c_{tag}[_inter].csv`; `FedSatGen` parses S and P from the first two fields.

      - `spec` — constellation settings
      - `tag` — free-form label for the scenario, must not contain the leading `{S}s_{P}c_` fields
      - `isl` — True for the inter-satellite file
    """
    return f"{spec.sats_per_plane}s_{spec.n_planes}c_{tag}{'_inter' if isl else ''}.csv"


def write_legacy_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write a legacy-layout DataFrame with the leading unnamed index column the STK files have."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)
    return path
