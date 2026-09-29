"""Command line: generate contact windows for a scenario YAML."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .constellation import build_constellation
from .ground import ground_windows
from .isl import isl_windows
from .legacy import legacy_filename, to_legacy_ground, to_legacy_isl, write_legacy_csv
from .spec import load_scenario


def main(argv: list[str] | None = None) -> None:
    """
        main(argv=None)

    Generate ground and/or ISL windows for a scenario and write them as Parquet (canonical,
    seconds since scenario start) and as STK-layout CSVs for the existing strategies.

      - `argv` — command-line arguments; see `flystack-windows --help`
    """
    ap = argparse.ArgumentParser(prog="flystack-windows", description=main.__doc__)
    ap.add_argument("scenario", type=Path, help="scenario YAML")
    ap.add_argument("--out", type=Path, default=Path("windows_out"), help="output folder")
    ap.add_argument("--only", choices=("ground", "isl"), help="generate one kind only")
    args = ap.parse_args(argv)

    sc = load_scenario(args.scenario)
    spec = sc.constellation
    tag = sc.extra.get("tag", f"brahe_{spec.pattern}")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    sats = build_constellation(spec)
    print(f"{sc.name}: {spec.n_planes} planes x {spec.sats_per_plane} sats, "
          f"{spec.duration_s / 86400:.2f} days, {spec.propagator}")

    if sc.ground is not None and args.only in (None, "ground"):
        t = time.time()
        gw = ground_windows(spec, sats, sc.ground)
        gw.to_parquet(out / f"{sc.name}_ground.parquet")
        p = write_legacy_csv(to_legacy_ground(spec, gw), out / legacy_filename(spec, tag, isl=False))
        print(f"  ground: {len(gw)} windows in {time.time() - t:.1f} s -> {p}")

    if sc.isl is not None and args.only in (None, "isl"):
        t = time.time()
        iw = isl_windows(spec, sats, sc.isl)
        iw.to_parquet(out / f"{sc.name}_isl.parquet")
        p = write_legacy_csv(to_legacy_isl(spec, iw), out / legacy_filename(spec, tag, isl=True))
        print(f"  isl: {len(iw)} windows in {time.time() - t:.1f} s -> {p}")
