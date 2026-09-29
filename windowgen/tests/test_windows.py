from datetime import datetime, timedelta, timezone

import brahe as bh
import numpy as np
import pytest

from flystack_windows import (ConstellationSpec, GroundSpec, IslSpec, build_constellation,
                              ground_windows, isl_windows, legacy_filename, link_margin_m,
                              pseudo_seconds, to_legacy_ground, to_legacy_isl)

EPOCH = datetime(2024, 4, 14, 16, 0, 0, tzinfo=timezone.utc)


def reference_time_to_int(dateobj):
    """Verbatim copy of `time_to_int` in project/fed/strategies/utils.py."""
    total = int(dateobj.strftime('%f')) * .000001
    total += int(dateobj.strftime('%S'))
    total += int(dateobj.strftime('%M')) * 60
    total += int(dateobj.strftime('%H')) * 60 * 60
    total += (int(dateobj.strftime('%j')) - 1) * 60 * 60 * 24
    total += (int(dateobj.strftime('%Y')) - 1970) * 60 * 60 * 24 * 365
    return total


def spec(**kw):
    base = dict(n_planes=1, sats_per_plane=10, phasing=0, altitude_km=705.0, inclination_deg=98.2,
                pattern="star", epoch_utc=EPOCH, duration_s=86400.0, propagator="keplerian")
    base.update(kw)
    return ConstellationSpec(**base)


def test_pseudo_seconds_matches_strategy_clock():
    assert pseudo_seconds(EPOCH) == 1711987200
    for dt in [EPOCH, EPOCH + timedelta(days=37, seconds=12.25), datetime(2023, 12, 31, 23, 59, 59)]:
        assert pseudo_seconds(dt) == pytest.approx(reference_time_to_int(dt), abs=1e-6)


def test_link_margin_geometry():
    r = bh.R_EARTH + 705e3
    isl = IslSpec(grazing_altitude_km=0.0, earth_model="sphere")
    a = np.array([r, 0.0, 0.0])
    assert link_margin_m(a, -a, isl) < 0                      # through the Earth
    th = np.deg2rad(36.0)
    b = np.array([r * np.cos(th), r * np.sin(th), 0.0])
    assert link_margin_m(a, b, isl) == pytest.approx(r * np.cos(th / 2) - bh.R_EARTH, rel=1e-9)
    capped = IslSpec(grazing_altitude_km=0.0, max_range_km=1000.0, earth_model="sphere")
    assert link_margin_m(a, b, capped) == pytest.approx(1000e3 - np.linalg.norm(b - a), rel=1e-9)


def test_wgs84_occludes_less_over_the_poles():
    # Chord whose midpoint sits 10 km above the pole: blocked by the sphere (R_EARTH is about
    # 21 km above the polar radius), clear of the WGS84 ellipsoid.
    b = bh.WGS84_A * (1.0 - bh.WGS84_F)
    z = b + 10e3
    r1, r2 = np.array([-3000e3, 0.0, z]), np.array([3000e3, 0.0, z])
    assert link_margin_m(r1, r2, IslSpec(grazing_altitude_km=0.0, earth_model="sphere")) < 0
    assert link_margin_m(r1, r2, IslSpec(grazing_altitude_km=0.0, earth_model="wgs84")) > 0
    # The same chord over the equator is judged the same way by both models.
    q1, q2 = np.array([bh.WGS84_A + 10e3, -3000e3, 0.0]), np.array([bh.WGS84_A + 10e3, 3000e3, 0.0])
    assert link_margin_m(q1, q2, IslSpec(grazing_altitude_km=0.0, earth_model="wgs84")) == pytest.approx(10e3, abs=1.0)


def test_in_plane_links_follow_line_of_sight():
    # 10 per plane: neighbours 36 deg apart clear the Earth (r cos 18 deg > R_E), next-nearest
    # 72 deg apart do not (r cos 36 deg < R_E), so only neighbour links exist, up all day.
    s = spec()
    w = isl_windows(s, build_constellation(s), IslSpec(grazing_altitude_km=0.0, earth_model="sphere", grid_step_s=60.0))
    assert len(w) == 20                                       # 10 neighbour pairs, both directions
    assert ((w.sat_1 - w.sat_2) % 10).isin([1, 9]).all()
    assert (w.start_s == 0).all() and (w.end_s == s.duration_s).all()
    # 4 per plane: neighbours 90 deg apart never see each other.
    s4 = spec(sats_per_plane=4)
    assert len(isl_windows(s4, build_constellation(s4), IslSpec(grazing_altitude_km=0.0, earth_model="sphere", grid_step_s=60.0))) == 0


def test_isl_crossings_converge_with_step():
    s = spec(n_planes=3, sats_per_plane=6, phasing=1, duration_s=3 * 3600.0)
    sats = build_constellation(s)
    coarse = isl_windows(s, sats, IslSpec(grazing_altitude_km=100.0, grid_step_s=20.0))
    fine = isl_windows(s, sats, IslSpec(grazing_altitude_km=100.0, grid_step_s=1.0))
    key = ["plane_1", "sat_1", "plane_2", "sat_2"]
    part = lambda w: (w[(w.plane_1 != w.plane_2) & (w.start_s > 0) & (w.end_s < s.duration_s)]
                      .sort_values(key + ["start_s"]).reset_index(drop=True))
    c, f = part(coarse), part(fine)
    assert len(c) == len(f) > 0
    assert (c[key].values == f[key].values).all()
    assert np.abs(c.start_s - f.start_s).max() < 1.0
    assert np.abs(c.end_s - f.end_s).max() < 1.0


def test_windows_are_well_formed_and_legacy_layout():
    s = spec(n_planes=2, sats_per_plane=3, phasing=1, duration_s=2 * 86400.0, propagator="sgp4")
    sats = build_constellation(s)
    assert [(x.plane, x.sat) for x in sats] == [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    gw = ground_windows(s, sats, GroundSpec(min_elevation_deg=5.0, providers=("ksat",)))
    assert len(gw) > 0 and (gw.end_s > gw.start_s).all() and gw.start_s.is_monotonic_increasing
    lg = to_legacy_ground(s, gw)
    assert lg["Start Time Seconds Cumulative"].iloc[0] >= 1711987200
    assert lg["Duration (sec)"].equals((lg["End Time Seconds Cumulative"]
                                        - lg["Start Time Seconds Cumulative"]).round(3))
    assert set(lg.cluster_num) == {1, 2} and set(lg.sat_num) == {1, 2, 3}
    iw = isl_windows(s, sats, IslSpec(grazing_altitude_km=100.0))
    li = to_legacy_isl(s, iw)
    assert li["Start Time Seconds Cumulative"].is_monotonic_increasing
    assert legacy_filename(s, "brahe_star", isl=True) == "3s_2c_brahe_star_inter.csv"


def test_full_span_duration_matches_stk_always_on_marker():
    s = spec(duration_s=91 * 86400.0, propagator="keplerian")
    iw = isl_windows(s, build_constellation(s), IslSpec(grazing_altitude_km=0.0, grid_step_s=600.0))
    assert (to_legacy_isl(s, iw)["Duration (sec)"] == 7862400.000).all()
