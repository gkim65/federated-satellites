"""Scenario description: constellation, ground segment and inter-satellite link model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

PATTERNS = ("star", "delta")
PROPAGATORS = ("sgp4", "keplerian")
EARTH_MODELS = ("wgs84", "sphere")


@dataclass(frozen=True)
class ConstellationSpec:
    """
        ConstellationSpec(n_planes, sats_per_plane, phasing, altitude_km, inclination_deg, ...)

    Circular Walker constellation plus the simulation span it is propagated over.

      - `n_planes` — number of orbital planes P (these are the FL clusters)
      - `sats_per_plane` — satellites per plane S
      - `phasing` — Walker phasing factor f, in 0..P-1
      - `altitude_km` — orbit altitude above the equatorial radius `brahe.R_EARTH` (km)
      - `inclination_deg` — orbit inclination (deg)
      - `pattern` — "star" (planes spread over 180 deg of RAAN) or "delta" (360 deg)
      - `raan0_deg` — RAAN of plane 0 (deg)
      - `mean_anomaly0_deg` — mean anomaly of satellite 0 in plane 0 (deg)
      - `epoch_utc` — scenario start, timezone-aware UTC datetime
      - `duration_s` — scenario length (s)
      - `propagator` — "sgp4" (J2 secular drift, bstar = 0) or "keplerian" (two-body)
      - `step_s` — propagator output step (s)
    """

    n_planes: int
    sats_per_plane: int
    phasing: int
    altitude_km: float
    inclination_deg: float
    pattern: str
    epoch_utc: datetime
    duration_s: float
    raan0_deg: float = 0.0
    mean_anomaly0_deg: float = 0.0
    propagator: str = "sgp4"
    step_s: float = 60.0

    def __post_init__(self):
        if self.pattern not in PATTERNS:
            raise ValueError(f"pattern must be one of {PATTERNS}, got {self.pattern!r}")
        if self.propagator not in PROPAGATORS:
            raise ValueError(f"propagator must be one of {PROPAGATORS}, got {self.propagator!r}")
        if not 0 <= self.phasing < self.n_planes:
            raise ValueError(f"phasing must be in 0..{self.n_planes - 1}, got {self.phasing}")
        if self.epoch_utc.tzinfo is None:
            raise ValueError("epoch_utc must be timezone-aware (UTC)")

    @property
    def n_sats(self) -> int:
        """Total number of satellites, P * S."""
        return self.n_planes * self.sats_per_plane


@dataclass(frozen=True)
class GroundSpec:
    """
        GroundSpec(min_elevation_deg, stations_file=None, providers=(), include=None)

    Ground segment used for satellite-to-ground contact windows.

      - `min_elevation_deg` — minimum elevation above the local horizon for a contact (deg)
      - `stations_file` — GeoJSON FeatureCollection of Point stations, each with a `name` property
      - `providers` — brahe ground-station providers to add, e.g. ("ksat", "aws")
      - `include` — if given, keep only stations with these names
    """

    min_elevation_deg: float
    stations_file: Path | None = None
    providers: tuple[str, ...] = ()
    include: tuple[str, ...] | None = None


@dataclass(frozen=True)
class IslSpec:
    """
        IslSpec(grazing_altitude_km, max_range_km=None, earth_model="wgs84", grid_step_s=10.0,
                both_directions=True)

    Inter-satellite link model: a link exists while the line of sight clears the Earth and the
    range limit holds.

      - `grazing_altitude_km` — the line of sight must pass at least this far above the Earth
        surface (km)
      - `max_range_km` — maximum link range, or None for no limit (km)
      - `earth_model` — occluding body: "wgs84" (oblate ellipsoid) or "sphere" (radius
        `brahe.R_EARTH`)
      - `grid_step_s` — sampling step of the visibility scan (s); crossings are interpolated
        linearly between samples
      - `both_directions` — also emit each window as (b, a), as the STK exports do

    NOTE: a window shorter than `grid_step_s` that opens and closes between two samples is missed.
    """

    grazing_altitude_km: float
    max_range_km: float | None = None
    earth_model: str = "wgs84"
    grid_step_s: float = 10.0
    both_directions: bool = True

    def __post_init__(self):
        if self.earth_model not in EARTH_MODELS:
            raise ValueError(f"earth_model must be one of {EARTH_MODELS}, got {self.earth_model!r}")


@dataclass(frozen=True)
class Scenario:
    """A named constellation with its optional ground segment and ISL model."""

    name: str
    constellation: ConstellationSpec
    ground: GroundSpec | None = None
    isl: IslSpec | None = None
    extra: dict = field(default_factory=dict)


def _parse_epoch(value) -> datetime:
    """Parse an ISO-8601 string (or datetime) into a timezone-aware UTC datetime."""
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def load_scenario(path: str | Path) -> Scenario:
    """
        load_scenario(path) -> Scenario

    Read a scenario YAML file (see `configs/` for the schema).

      - `path` — path to the YAML file; a relative `ground.stations_file` resolves against its folder

    Returns the parsed `Scenario`.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    c = raw["constellation"]
    duration_s = float(c.get("duration_s", 0.0)) + 86400.0 * float(c.get("duration_days", 0.0))
    constellation = ConstellationSpec(
        n_planes=int(c["n_planes"]),
        sats_per_plane=int(c["sats_per_plane"]),
        phasing=int(c["phasing"]),
        altitude_km=float(c["altitude_km"]),
        inclination_deg=float(c["inclination_deg"]),
        pattern=str(c["pattern"]),
        epoch_utc=_parse_epoch(c["epoch_utc"]),
        duration_s=duration_s,
        raan0_deg=float(c.get("raan0_deg", 0.0)),
        mean_anomaly0_deg=float(c.get("mean_anomaly0_deg", 0.0)),
        propagator=str(c.get("propagator", "sgp4")),
        step_s=float(c.get("step_s", 60.0)),
    )
    ground = None
    if raw.get("ground"):
        g = raw["ground"]
        sf = g.get("stations_file")
        ground = GroundSpec(
            min_elevation_deg=float(g["min_elevation_deg"]),
            stations_file=(path.parent / sf) if sf else None,
            providers=tuple(g.get("providers", ())),
            include=tuple(g["include"]) if g.get("include") else None,
        )
    isl = None
    if raw.get("isl"):
        i = raw["isl"]
        isl = IslSpec(
            grazing_altitude_km=float(i["grazing_altitude_km"]),
            max_range_km=float(i["max_range_km"]) if i.get("max_range_km") is not None else None,
            earth_model=str(i.get("earth_model", "wgs84")),
            grid_step_s=float(i.get("grid_step_s", 10.0)),
            both_directions=bool(i.get("both_directions", True)),
        )
    return Scenario(name=str(raw.get("name", path.stem)), constellation=constellation,
                    ground=ground, isl=isl, extra=raw.get("extra", {}) or {})
