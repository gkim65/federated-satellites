# flystack-windows

Generates satellite contact windows with [brahe](https://docs.brahe.space/latest/) for the
federated-learning simulation in `project/`. It replaces the STK-exported access CSVs. You
describe a Walker constellation in YAML, and the tool writes its windows in the STK CSV layout
that the strategies in `project/fed/strategies` already read.

It produces two kinds of window:

- **Satellite-to-ground:** computed with brahe's `location_accesses` and an elevation mask. It
  can use stations from a GeoJSON file or from any brahe provider (`ksat`, `aws`, `atlas`, ...).
- **Inter-satellite links (ISL):** a link is up while the line of sight clears the WGS84
  ellipsoid (or a sphere) by a grazing height, with an optional maximum range. brahe's access
  search only covers satellite-to-ground, so these are scanned on a time grid from brahe ECI
  states. Each crossing is then interpolated linearly between samples.

## Setup and use

This is its own `uv` project. brahe 1.7 needs `rich>=14` and flwr 1.19 pins `rich<14`, so the two
cannot share one environment.

```
cd windowgen
uv sync
uv run flystack-windows configs/stk_fit_10s_4c.yaml --out ../datasets/landsat
uv run pytest
```

Each run writes the following to `--out`:

- `{S}s_{P}c_{tag}.csv` for ground windows and `{S}s_{P}c_{tag}_inter.csv` for ISL windows.
  These are drop-in replacements for the STK files, selected with `stk.sim_fname` in
  `project/config/config.yaml`.
- `{name}_ground.parquet` and `{name}_isl.parquet`, holding the same windows with times in
  seconds since the scenario start.

The CSV times use the strategies' clock (`time_to_int` in `project/fed/strategies/utils.py`).
That clock counts 365-day years since 1970, so 2024-04-14 16:00 UTC is 1711987200.

## Scenario YAML

See `configs/`.

| Section | Keys |
|---|---|
| `constellation` | Walker `n_planes`, `sats_per_plane`, `phasing`, `pattern` (star/delta), `altitude_km`, `inclination_deg`, `raan0_deg`, `mean_anomaly0_deg`, `epoch_utc`, `duration_days`, `propagator` (sgp4/keplerian) |
| `ground` | `min_elevation_deg`, and `stations_file` and/or `providers`; optionally `include` to keep only some station names |
| `isl` | `grazing_altitude_km`, `max_range_km`, `earth_model` (wgs84/sphere), `grid_step_s`, `both_directions` |

Planes and satellites are numbered from 1 in the output (`cluster_num`, `sat_num`). Planes are
in RAAN order, so in a Walker star, plane 1 and plane P are the counter-rotating pair across the
seam.

## Agreement with the STK exports

`configs/stk_fit_10s_4c.yaml` and `configs/stk_fit_10s_3c.yaml` use one set of parameters:

- Walker star, f = 1
- two-body orbit, 90 deg inclination, 506.404 km altitude
- WGS84 occlusion with a 6 km grazing height

These values were fitted to the STK ISL files. They were not read from the STK scenario. Checked
with `scripts/compare_stk_isl.py`:

| File | Rows (brahe / STK) | Per-plane-pair counts | Max \|start diff\| | Max \|duration diff\| |
|---|---|---|---|---|
| `10s_4c_s_landsat_star_inter.csv` | 1,438,604 / 1,438,604 | identical | 0.53 s | 0.91 s |
| `10s_3c_s_landsat_star_inter.csv` | 663,976 / 663,976 | identical | 0.53 s | 0.65 s |

The ACM waterfall scheduler (`scheduleAdjacentISL`, 50 cycles, no dropout) picks the same
windows from both sources to within 0.3 s, for P = 3 and P = 4.

The ground-station files (`10s_10c`, `12s_12c`) have not been compared yet. That needs the STK
facility coordinates and elevation mask.

## Limits

- **Constellation shape:** only circular Walker constellations are built today. The propagator
  is two-body (`keplerian`), or `sgp4` with B* = 0, which adds J2 secular drift but no drag.
- **Short ISL windows:** a window shorter than `grid_step_s` that opens and closes between two
  samples is missed.
- **Dataset partitions:** the non-IID splits in `project/utils` are fixed at 100 partitions,
  which caps a constellation at 100 satellites for CIFAR10/MNIST/EuroSAT.
