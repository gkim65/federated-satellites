"""Satellite contact-window generation with brahe for federated-learning simulation."""

from .cli import main
from .constellation import build_constellation
from .ground import ground_windows, load_stations
from .isl import isl_windows, link_margin_m
from .legacy import legacy_filename, pseudo_seconds, to_legacy_ground, to_legacy_isl
from .spec import ConstellationSpec, GroundSpec, IslSpec, Scenario, load_scenario

__all__ = [
    "main", "build_constellation", "ground_windows", "load_stations", "isl_windows",
    "link_margin_m", "legacy_filename", "pseudo_seconds", "to_legacy_ground", "to_legacy_isl",
    "ConstellationSpec", "GroundSpec", "IslSpec", "Scenario", "load_scenario",
]
