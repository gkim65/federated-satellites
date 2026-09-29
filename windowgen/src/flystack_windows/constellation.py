"""Walker constellation propagators built with brahe."""

from __future__ import annotations

import re
from dataclasses import dataclass

import brahe as bh

from .spec import ConstellationSpec

BASE_NAME = "SAT"
_NAME_RE = re.compile(rf"^{BASE_NAME}-P(\d+)-S(\d+)$")

_eop_ready = False


def ensure_eop() -> None:
    """Load brahe's Earth-orientation data once per process (needed for ECI/ECEF transforms)."""
    global _eop_ready
    if not _eop_ready:
        bh.initialize_eop()
        _eop_ready = True


def to_epoch(spec: ConstellationSpec) -> bh.Epoch:
    """Scenario start of `spec` as a brahe UTC `Epoch`."""
    t = spec.epoch_utc
    return bh.Epoch.from_datetime(t.year, t.month, t.day, t.hour, t.minute,
                                  float(t.second), float(t.microsecond) * 1e3, bh.TimeSystem.UTC)


@dataclass(frozen=True)
class Satellite:
    """
        Satellite(plane, sat, propagator)

    One constellation member.

      - `plane` — orbital plane / FL cluster number, 1-indexed (the STK `cluster_num`)
      - `sat` — satellite number within its plane, 1-indexed (the STK `sat_num`)
      - `propagator` — brahe propagator named `SAT-P{plane-1}-S{sat-1}`
    """

    plane: int
    sat: int
    propagator: object


def parse_name(name: str) -> tuple[int, int]:
    """
        parse_name(name) -> (plane, sat)

    Map a brahe Walker satellite name `SAT-P{p}-S{s}` (0-indexed) to 1-indexed (plane, sat).
    """
    m = _NAME_RE.match(name)
    if m is None:
        raise ValueError(f"not a Walker satellite name: {name!r}")
    return int(m.group(1)) + 1, int(m.group(2)) + 1


def build_constellation(spec: ConstellationSpec) -> list[Satellite]:
    """
        build_constellation(spec) -> list[Satellite]

    Generate the Walker constellation described by `spec` and one propagator per satellite.

      - `spec` — constellation and propagator settings

    Returns the satellites in plane-major order (plane 1 sat 1, plane 1 sat 2, ...).
    """
    ensure_eop()
    pattern = bh.WalkerPattern.STAR if spec.pattern == "star" else bh.WalkerPattern.DELTA
    gen = (
        bh.WalkerConstellationGenerator.builder(
            spec.n_sats, spec.n_planes, spec.phasing,
            bh.R_EARTH + spec.altitude_km * 1e3, spec.inclination_deg, to_epoch(spec))
        .angle_format(bh.AngleFormat.DEGREES)
        .pattern(pattern)
        .reference_raan(spec.raan0_deg)
        .reference_mean_anomaly(spec.mean_anomaly0_deg)
        .base_name(BASE_NAME)
        .build()
    )
    if spec.propagator == "sgp4":
        props = gen.as_sgp_propagators(spec.step_s, 0.0, 0.0, 0.0)
    else:
        props = gen.as_keplerian_propagators(spec.step_s)
    sats = [Satellite(*parse_name(p.get_name()), p) for p in props]
    return sorted(sats, key=lambda s: (s.plane, s.sat))
