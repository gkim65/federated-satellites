"""Table 3: wall-clock time to convergence (WCTC) of the AutoFLSat waterfall under ISL dropout.

Usage (repo root): uv run python -m figures.table3_dropout [--seeds 30] [--planes 2 3 4]

Simulates the waterfall schedule only (no training) over `--cycles` FL cycles on an `_inter`
window CSV, dropping each candidate ISL window with probability equal to the dropout rate, and
prints WCTC per rate. The 0% row is deterministic. With `--seeds N > 1` each rate is run with
seeds 0..N-1 and min / median / max are printed instead.
"""

import argparse
import contextlib
import io

import numpy as np
import pandas as pd

from figures.style import STK_ISL_4C
from project.fed.strategies.AutoFLSatWaterFall import scheduleAdjacentISL

DROPOUT_RATES = [0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]


def build_waterfall_sequence(cluster_n):
    """(phase, left_pair, right_pair) steps of one FL cycle, as used for the paper's table."""
    sequence = []
    mid = cluster_n // 2
    for step in range(mid - 1 if cluster_n % 2 == 0 else mid):
        left_pair = (step + 1, step + 2)
        right_pair = (cluster_n - step, cluster_n - step - 1)
        if left_pair == right_pair[::-1]:
            right_pair = None
        sequence.append(('scatter', left_pair, right_pair))
    if cluster_n % 2 == 0:
        sequence.append(('middle_exchange', (cluster_n // 2, cluster_n // 2 + 1), None))
    allgather_start = cluster_n // 2
    for step in range(allgather_start, 0, -1):
        left_pair = (step + 1, step)
        right_pair = (cluster_n - step, cluster_n - step + 1)
        if left_pair[1] == right_pair[1] or cluster_n == 2:
            right_pair = None
        sequence.append(('allgather', left_pair, right_pair))
    return sequence


def simulate_scheduling(sat_df, cluster_n, config_epochs, local_train_s, n_fl_cycles,
                        dropout_rate, factor_c=1):
    """
        simulate_scheduling(sat_df, cluster_n, config_epochs, local_train_s, n_fl_cycles,
                            dropout_rate, factor_c=1) -> float

    Run `n_fl_cycles` of local training plus waterfall steps and return the WCTC (hours).

      - `sat_df` — `_inter` windows restricted to planes 1..cluster_n
      - `cluster_n` — number of orbital planes P
      - `config_epochs` — the `epochs` setting; the scheduler waits `config_epochs + 120` s
      - `local_train_s` — simulated local training time per cycle (s)
      - `n_fl_cycles` — number of full two-tier cycles
      - `dropout_rate` — probability that a candidate ISL window is unavailable
    """
    counter = 0
    t0 = sat_df['Start Time Seconds Cumulative'].iloc[0]
    start_time_og = t0
    seq = build_waterfall_sequence(cluster_n)
    for _ in range(n_fl_cycles):
        start_time_og += local_train_s
        for _, left_pair, right_pair in seq:
            start_time, _, counter, _, _, _ = scheduleAdjacentISL(
                sat_df, counter, factor_c, start_time_og, int(config_epochs) + 120,
                left_pair, right_pair, dropout_rate, wandbUse=False, verbose=False)
            start_time_og = start_time
    return (start_time_og - t0) / 3600


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", default=STK_ISL_4C, help="10s_4c `_inter` window CSV")
    ap.add_argument("--planes", type=int, nargs="+", default=[4], help="P values (paper: 4)")
    ap.add_argument("--cycles", type=int, default=50, help="FL cycles (paper: 50)")
    ap.add_argument("--config-epochs", type=int, default=5)
    ap.add_argument("--local-train-s", type=float, default=120.0,
                    help="local training time per cycle in s; the paper's 0%% row (23.81 h) "
                         "holds for any value in 90-215")
    ap.add_argument("--seed", type=int, default=0, help="seed for a single run")
    ap.add_argument("--seeds", type=int, default=1, help="run seeds 0..N-1 and summarise")
    args = ap.parse_args()

    df = pd.read_csv(args.windows)
    for P in args.planes:
        planes = list(range(1, P + 1))
        d = df[df.cluster_num_1.isin(planes) & df.cluster_num_2.isin(planes)].reset_index(drop=True)
        seeds = range(args.seeds) if args.seeds > 1 else [args.seed]
        wctc = {r: [] for r in DROPOUT_RATES}
        for seed in seeds:
            np.random.seed(seed)
            for r in DROPOUT_RATES:
                with contextlib.redirect_stdout(io.StringIO()):
                    wctc[r].append(simulate_scheduling(d, P, args.config_epochs,
                                                       args.local_train_s, args.cycles, r))
        base = wctc[0.0][0]
        print(f"\nP = {P}, {args.cycles} FL cycles, local training {args.local_train_s:g} s, "
              f"{'seed ' + str(args.seed) if args.seeds == 1 else f'{args.seeds} seeds'}")
        if args.seeds == 1:
            print(f"{'ISL dropout':>12} {'WCTC (h)':>9} {'increase':>9}")
            for r, v in wctc.items():
                inc = "-" if r == 0 else f"{(v[0] - base) / base:.1%}"
                print(f"{r:>12.0%} {v[0]:>9.2f} {inc:>9}")
        else:
            print(f"{'ISL dropout':>12} {'min':>7} {'median':>7} {'max':>7}  (WCTC, h)")
            for r, v in wctc.items():
                print(f"{r:>12.0%} {min(v):>7.2f} {np.median(v):>7.2f} {max(v):>7.2f}")


if __name__ == "__main__":
    main()
