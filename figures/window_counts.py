"""Appendix: ISL windows per adjacent plane pair over the first orbit (P = 4 Walker star).

Usage (repo root): uv run python -m figures.window_counts [--windows CSV] [--theme dark]

Counts windows of at least 600 s that open within the first 90 minutes, and their total duration,
for plane pairs 1-2, 2-3 and 3-4.

NOTE: the `_inter` CSVs list every window once per direction (a->b and b->a), so each link is
counted twice; halve the numbers for distinct links.
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from figures.style import STK_ISL_4C, add_common_args, save, setup

RC = {"axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9}
MIN_WINDOW_SECS = 600      # shortest window counted (s)
T0 = 1711987200            # scenario start on the strategies' clock (s)
ORBITAL_PERIOD = 90 * 60   # counting span (s)
PAIRS = [(1, 2), (2, 3), (3, 4)]


def count_windows(path):
    """Per-pair window count and total duration (s) for windows opening in the first orbit."""
    df = pd.read_csv(path)
    df = df[(df['Duration (sec)'] < 7862400) & (df['cluster_num_1'] != df['cluster_num_2'])]
    c1 = df[['cluster_num_1', 'cluster_num_2']].min(axis=1)
    c2 = df[['cluster_num_1', 'cluster_num_2']].max(axis=1)
    start = df['Start Time Seconds Cumulative']
    w = df[(start >= T0) & (start <= T0 + ORBITAL_PERIOD) & (df['Duration (sec)'] >= MIN_WINDOW_SECS)]
    pair = list(zip(c1[w.index], c2[w.index]))
    counts = {p: sum(1 for q in pair if q == p) for p in PAIRS}
    total = {p: float(w['Duration (sec)'][[q == p for q in pair]].sum()) for p in PAIRS}
    return counts, total


def plot(counts, total, c):
    """Two bar panels: window count and total available duration per pair."""
    labels = [r'$P_1{\leftrightarrow}P_2$', r'$P_2{\leftrightarrow}P_3$', r'$P_3{\leftrightarrow}P_4$']
    colors = ['#378ADD', '#7F77DD', '#D85A30']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    panels = [([counts[p] for p in PAIRS], 'Number of valid windows', 'Window count per pair (one orbit)', str),
              ([total[p] / 60 for p in PAIRS], 'Total available duration (minutes)',
               'Total window duration per pair (one orbit)', lambda v: f'{v:.1f}')]
    for ax, (vals, ylabel, title, fmt) in zip(axes, panels):
        bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor=c["fg"], linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, fmt(val),
                    ha='center', va='bottom', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel('Adjacent plane pair', fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, max(vals) * 1.2)
    plt.tight_layout()
    return fig


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--windows", default=STK_ISL_4C, help="10s_4c `_inter` window CSV")
    args = ap.parse_args()
    colours = setup(RC, args.theme)
    counts, total = count_windows(args.windows)
    for p in PAIRS:
        print(f"  {p}: {counts[p]} windows, {total[p] / 60:.1f} min total")
    save(plot(counts, total, colours), "window_counts", args)


if __name__ == "__main__":
    main()
