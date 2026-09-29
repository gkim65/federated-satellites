"""Compare a generated `_inter` CSV against an STK `_inter` export, pair by pair.

Usage: uv run python scripts/compare_stk_isl.py GENERATED.csv STK.csv

Prints per-plane-pair window counts and duration quantiles for both files, then matches every STK
window to the generated window of the same satellite pair with the nearest start time and reports
the start and duration differences.
"""

import sys

import numpy as np
import pandas as pd

ALWAYS_ON_S = 7862400.0
KEY = ["cluster_num_1", "sat_num_1", "cluster_num_2", "sat_num_2"]
T = "Start Time Seconds Cumulative"
DUR = "Duration (sec)"


def load(path):
    df = pd.read_csv(path)
    return df[df[DUR] != ALWAYS_ON_S], int((df[DUR] == ALWAYS_ON_S).sum())


def summary(df):
    one = df[df.cluster_num_1 < df.cluster_num_2]
    return one.groupby(["cluster_num_1", "cluster_num_2"]).size(), one[DUR].quantile([0.1, 0.5, 0.9, 1.0])


def main(gen_path, stk_path):
    gen, gen_on = load(gen_path)
    stk, stk_on = load(stk_path)
    print(f"rows: generated {len(gen)} (+{gen_on} always-on), STK {len(stk)} (+{stk_on} always-on)")
    (gc, gq), (sc, sq) = summary(gen), summary(stk)
    print(pd.DataFrame({"generated": gc, "stk": sc}).assign(diff=lambda d: d.generated - d.stk))
    print(pd.DataFrame({"generated": gq, "stk": sq}).round(1).rename_axis("duration quantile"))

    d_start, d_dur, unmatched = [], [], 0
    gen_groups = {k: g.sort_values(T) for k, g in gen.groupby(KEY)}
    for k, s in stk.groupby(KEY):
        g = gen_groups.get(k)
        if g is None:
            unmatched += len(s)
            continue
        gt, st = g[T].to_numpy(), s[T].to_numpy()
        idx = np.clip(np.searchsorted(gt, st), 1, len(gt) - 1)
        idx = np.where(np.abs(gt[idx - 1] - st) < np.abs(gt[idx] - st), idx - 1, idx)
        d_start.append(gt[idx] - st)
        d_dur.append(g[DUR].to_numpy()[idx] - s[DUR].to_numpy())
    d_start, d_dur = np.concatenate(d_start), np.concatenate(d_dur)
    q = [0.5, 0.9, 0.99, 1.0]
    print(f"STK windows matched: {len(d_start)}, pairs missing from generated: {unmatched} windows")
    print("|start diff| s quantiles (50/90/99/100%):", np.round(np.quantile(np.abs(d_start), q), 2))
    print("|duration diff| s quantiles (50/90/99/100%):", np.round(np.quantile(np.abs(d_dur), q), 2))


if __name__ == "__main__":
    main(*sys.argv[1:3])
