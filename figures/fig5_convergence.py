"""Figure 5 and Table 2 accuracy: AutoFLSat convergence on EuroSAT for P = 2, 3, 4 planes.

Usage (repo root): uv run python -m figures.fig5_convergence [--refresh] [--no-pad] [--theme dark]

Reads each run's logged accuracy from figures/cache/fig5_P{P}.csv. If a cache file is missing (or
with --refresh), the history is downloaded from wandb using the run paths in
figures/wandb_runs.local.yaml, which is gitignored; copy wandb_runs.example.yaml to create it.
Prints the mean accuracy across planes at the cycle closest to 51 (Table 2).
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from figures.style import CACHE, REPO, add_common_args, italic, save, setup

RC = {"axes.labelsize": 13, "font.size": 12, "legend.fontsize": 11,
      "xtick.labelsize": 11, "ytick.labelsize": 11, "figure.figsize": (6, 4)}
RUNS_FILE = REPO / "figures" / "wandb_runs.local.yaml"

N_CLUSTERS = {2: [1, 2], 3: [1, 2, 3], 4: [1, 2, 3, 4]}
COLORS = {2: '#1f77b4', 3: '#ff7f0e', 4: '#2ca02c'}
WINDOW = 5                                  # rolling-mean window (logged rounds)
FL_ROUNDS_PER_CYCLE = {2: 3, 3: 5, 4: 7}    # Flower rounds per full two-tier cycle
TABLE2_CYCLE = 51
PAD = 20


def history(P: int, refresh: bool) -> pd.DataFrame:
    """Logged history of the P-plane run, from the cache or (first time / --refresh) from wandb."""
    path = CACHE / f"fig5_P{P}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)
    if not RUNS_FILE.exists():
        raise SystemExit(f"{path} is missing and {RUNS_FILE} does not exist; copy "
                         "figures/wandb_runs.example.yaml to it and fill in your run paths")
    import wandb
    run_path = yaml.safe_load(RUNS_FILE.read_text())["fig5_convergence"][P]
    df = wandb.Api().run(run_path).history(samples=1000000)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--refresh", action="store_true", help="re-download histories from wandb")
    ap.add_argument("--no-pad", action="store_true",
                    help="end the P=3 curve at its last logged round instead of extending it")
    args = ap.parse_args()
    setup(RC, args.theme)

    fig, ax = plt.subplots()
    for P in (2, 3, 4):
        df = history(P, args.refresh)
        cluster_accs = []
        for cluster in N_CLUSTERS[P]:
            cluster_df = df[df['cluster_num'] == cluster].sort_values('server_round')
            rounds = cluster_df['server_round'].values
            if P == 3:
                # The P=3 run logged `acc` only. NOTE: the paper figure appends PAD copies of the
                # last value so the curve reaches the axis limit; the run's own data ends earlier.
                acc = cluster_df['acc'].rolling(window=WINDOW, min_periods=1).mean().values
                if not args.no_pad:
                    acc = np.concatenate([acc, np.full(PAD, acc[-1])])
                    rounds = np.concatenate([rounds, rounds[-1] + np.arange(1, PAD + 1)])
            else:
                acc = cluster_df['acc_local'].rolling(window=WINDOW, min_periods=1).mean().values
            cluster_accs.append((rounds, acc))

        min_len = min(len(r) for r, _ in cluster_accs)
        common_rounds = cluster_accs[0][0][:min_len]
        acc_matrix = np.array([a[:min_len] for _, a in cluster_accs])
        mean_acc, min_acc, max_acc = acc_matrix.mean(0), acc_matrix.min(0), acc_matrix.max(0)
        fl_cycles = common_rounds / FL_ROUNDS_PER_CYCLE[P]

        idx = np.argmin(np.abs(fl_cycles - TABLE2_CYCLE))
        last_logged = df['server_round'].max() / FL_ROUNDS_PER_CYCLE[P]
        print(f"P={P}: mean accuracy {mean_acc[idx]:.4f} at cycle {fl_cycles[idx]:.2f} "
              f"(last logged cycle {last_logged:.1f})")

        ax.plot(fl_cycles, mean_acc, label=rf'$P={P}$', color=COLORS[P], linewidth=2.0)
        ax.fill_between(fl_cycles, min_acc, max_acc, alpha=0.15, color=COLORS[P])

    ax.set_xlabel(r'Number of Full Two-Tier Aggregation Rounds')
    ax.set_ylabel(italic('AutoFLSat') + r' Accuracy on EuroSAT')
    ax.legend(loc='lower right', title=r'Constellation Size')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, 51)
    plt.tight_layout()
    save(fig, "fig5_accuracy_curves_combined" + ("_nopad" if args.no_pad else ""), args,
         dpi=None, png_dpi=300)


if __name__ == "__main__":
    main()
