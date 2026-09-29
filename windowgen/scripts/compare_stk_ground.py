"""Compare a generated ground-station CSV against an STK ground export.

Usage: uv run python scripts/compare_stk_ground.py GENERATED.csv STK.csv [DAYS]

Only stations present in both files are compared. With DAYS, both files are cut to windows
that open after the scenario start and close before DAYS days, so partial passes at the edges
do not count. Prints per-station pass counts, duration quantiles, and, for every STK pass, the
start and duration difference to the generated pass of the same satellite and station with the
nearest start time.
"""

import sys

import numpy as np
import pandas as pd

T0 = 1711987200.0
KEY = ["cluster_num", "sat_num", "ground_station"]
T = "Start Time Seconds Cumulative"
DUR = "Duration (sec)"


def load(path, stations=None, days=None):
    df = pd.read_csv(path)
    if stations is not None:
        df = df[df.ground_station.isin(stations)]
    if days is not None:
        df = df[(df[T] > T0) & (df["End Time Seconds Cumulative"] < T0 + days * 86400.0)]
    return df


def main(gen_path, stk_path, days=None):
    days = float(days) if days is not None else None
    gen = load(gen_path, days=days)
    stk = load(stk_path, stations=set(gen.ground_station), days=days)
    print(f"passes: generated {len(gen)}, STK {len(stk)} ({len(stk) / len(gen) - 1:+.2%} STK vs generated)")
    counts = pd.DataFrame({"generated": gen.groupby("ground_station").size(),
                           "stk": stk.groupby("ground_station").size()})
    print(counts.assign(ratio=lambda d: (d.generated / d.stk).round(3)))
    q = [0.1, 0.5, 0.9, 1.0]
    print(pd.DataFrame({"generated": gen[DUR].quantile(q), "stk": stk[DUR].quantile(q)})
          .round(1).rename_axis("duration quantile"))

    d_start, d_dur = [], []
    gen_groups = {k: g.sort_values(T) for k, g in gen.groupby(KEY)}
    for k, s in stk.groupby(KEY):
        g = gen_groups.get(k)
        if g is None or len(g) < 2:
            continue
        gt, st = g[T].to_numpy(), s[T].to_numpy()
        idx = np.clip(np.searchsorted(gt, st), 1, len(gt) - 1)
        idx = np.where(np.abs(gt[idx - 1] - st) < np.abs(gt[idx] - st), idx - 1, idx)
        d_start.append(gt[idx] - st)
        d_dur.append(g[DUR].to_numpy()[idx] - s[DUR].to_numpy())
    d_start, d_dur = np.abs(np.concatenate(d_start)), np.abs(np.concatenate(d_dur))
    qs = [0.5, 0.9, 0.99, 1.0]
    print(f"STK passes matched: {len(d_start)}")
    print("|start diff| s quantiles (50/90/99/100%):", np.round(np.quantile(d_start, qs), 1))
    print("|duration diff| s quantiles (50/90/99/100%):", np.round(np.quantile(d_dur, qs), 1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
