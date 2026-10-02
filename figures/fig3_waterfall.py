"""Figure 3: AutoFLSat waterfall aggregation schedule on ISL access windows (P = 4, 5 hours).

Usage (repo root): uv run python -m figures.fig3_waterfall [--windows CSV] [--theme dark]

Simulates the scatter / middle-exchange / allgather sequence for planes 1-4 over the first five
hours of an `_inter` window CSV and draws the windows each phase selects on top of every
available window. Defaults to the STK export used in the paper; pass a brahe-generated
`10s_4c_..._inter.csv` to draw the same schedule from brahe windows.
"""

import argparse

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from figures.style import STK_ISL_4C, add_common_args, italic, save, setup

RC = {"axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
      "legend.fontsize": 9, "figure.titlesize": 11}

NUM_EPOCHS = 10
EPOCH_SECS = 60
EPOCHS = NUM_EPOCHS * EPOCH_SECS     # local training time between cycles (s)
MIN_WINDOW_SECS = 200                # shortest usable ISL window (s), as in scheduleAdjacentISL
ORBITAL_PERIOD = 60 * 60 * 5         # plotted span (s)
T0 = 1711987200                      # scenario start on the strategies' clock (s)
ALWAYS_ON_S = 7862400


def load_df(path):
    """Inter-plane windows from an `_inter` CSV, with order-normalised plane pair `c1 < c2`."""
    df = pd.read_csv(path)
    df = df[df['Duration (sec)'] < ALWAYS_ON_S].reset_index(drop=True)
    df = df[df['cluster_num_1'] != df['cluster_num_2']].reset_index(drop=True)
    df['c1'] = df[['cluster_num_1', 'cluster_num_2']].min(axis=1)
    df['c2'] = df[['cluster_num_1', 'cluster_num_2']].max(axis=1)
    df = df.sort_values('Start Time Seconds Cumulative').reset_index(drop=True)
    return df


def find_window(df, counter, start_time_og, pair_left, pair_right):
    """First windows after training ends for one or two plane pairs; see scheduleAdjacentISL."""
    training_complete = start_time_og + EPOCHS
    found_left = False
    found_right = pair_right is None
    left_start = left_end = right_start = right_end = 0
    i = counter
    while not (found_left and found_right):
        if i >= len(df):
            raise ValueError(f"No window found for left={pair_left} right={pair_right}")
        row = df.iloc[i]
        c1, c2 = int(row['c1']), int(row['c2'])
        t_start = row['Start Time Seconds Cumulative']
        t_end = row['End Time Seconds Cumulative']
        dur = row['Duration (sec)']
        if t_start > training_complete and dur > MIN_WINDOW_SECS:
            if not found_left and pair_left is not None:
                l1, l2 = min(pair_left), max(pair_left)
                if c1 == l1 and c2 == l2:
                    left_start, left_end = t_start, t_end
                    found_left = True
            if not found_right and pair_right is not None:
                r1, r2 = min(pair_right), max(pair_right)
                if c1 == r1 and c2 == r2:
                    right_start, right_end = t_start, t_end
                    found_right = True
        i += 1
    return left_start, left_end, right_start, right_end, i


def simulate_waterfall(df):
    """Run P = 4 waterfall cycles until the plotted span ends; returns the selected windows."""
    counter = 0
    start_time = T0
    selected = []
    cycle = 0
    T_END = T0 + ORBITAL_PERIOD
    while start_time < T_END:
        cycle += 1
        try:
            left_pair, right_pair = (1, 2), (3, 4)
            ls, le, rs, re, counter = find_window(df, counter, start_time, left_pair, right_pair)
            t_selected = max(ls, rs)
            if t_selected > T_END:
                break
            selected.append(dict(cycle=cycle, phase="scatter", left_pair=left_pair,
                                 right_pair=right_pair, left_start=ls, left_end=le,
                                 right_start=rs, right_end=re, selected_time=t_selected))
            start_time = t_selected

            mid_pair = (2, 3)
            ls, le, _, _, counter = find_window(df, counter, start_time, mid_pair, None)
            if ls > T_END:
                break
            selected.append(dict(cycle=cycle, phase="middle_exchange", left_pair=mid_pair,
                                 right_pair=None, left_start=ls, left_end=le,
                                 right_start=None, right_end=None, selected_time=ls))
            start_time = ls

            left_pair, right_pair = (1, 2), (3, 4)
            ls, le, rs, re, counter = find_window(df, counter, start_time, left_pair, right_pair)
            t_selected = max(ls, rs)
            if t_selected > T_END:
                break
            selected.append(dict(cycle=cycle, phase="allgather", left_pair=left_pair,
                                 right_pair=right_pair, left_start=ls, left_end=le,
                                 right_start=rs, right_end=re, selected_time=t_selected))
            start_time = max(le, re) + EPOCHS
        except ValueError as e:
            print(f"  Stopping: {e}")
            break
    return selected


def plot(df, selected, c):
    """Draw the schedule; `c` is the theme colour dict from `figures.style.setup`."""
    fig, ax = plt.subplots(figsize=(13, 5))
    T_END = T0 + ORBITAL_PERIOD
    plane_y = {1: 3.0, 2: 2.0, 3: 1.0, 4: 0.0}
    gap_y = {(1, 2): 2.5, (2, 3): 1.5, (3, 4): 0.5}
    gap_h = 0.35
    Y_PHASE, Y_LOCAL, Y_ROUND = 3.45, 3.70, 3.95
    PAIR_COLORS = {(1, 2): '#378ADD', (2, 3): '#7F77DD', (3, 4): '#D85A30'}
    PHASE_HATCH = {'scatter': '', 'middle_exchange': '///', 'allgather': '...'}

    def t2x(t):
        return (t - T0) / ORBITAL_PERIOD

    for _, row in df.iterrows():
        pair = (int(row['c1']), int(row['c2']))
        if pair not in gap_y:
            continue
        ts = row['Start Time Seconds Cumulative']
        te = row['End Time Seconds Cumulative']
        if ts > T_END or te < T0:
            continue
        ts, te = max(ts, T0), min(te, T_END)
        ax.barh(gap_y[pair], t2x(te) - t2x(ts), left=t2x(ts), height=gap_h * 0.6,
                color=PAIR_COLORS[pair], alpha=0.18, zorder=2)

    cycles = sorted(set(s['cycle'] for s in selected))
    cycle_ends = {}
    for sel in selected:
        te = sel.get('right_end') or sel['left_end']
        cycle_ends[sel['cycle']] = max(cycle_ends.get(sel['cycle'], 0), te)
    for cyc in cycles:
        end = cycle_ends.get(cyc, 0)
        next_scatter = next((s['selected_time'] for s in selected
                             if s['cycle'] == cyc + 1 and s['phase'] == 'scatter'), None)
        if next_scatter and end < next_scatter:
            ax.axvspan(t2x(end), t2x(next_scatter), color=c["shade"], alpha=0.5, zorder=0)
            mid = (t2x(end) + t2x(next_scatter)) / 2
            ax.text(mid, Y_LOCAL, italic('local training'), ha='center', va='center',
                    fontsize=10, color=c["muted"])

    for cyc in cycles:
        scatter = next((s for s in selected if s['cycle'] == cyc and s['phase'] == 'scatter'), None)
        if scatter:
            ax.text(t2x(scatter['selected_time']), Y_ROUND, rf'Round {cyc}', ha='center',
                    va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc=c["box"], ec='none', alpha=0.9),
                    zorder=10)

    phase_labels = {'scatter': 'Scatter', 'middle_exchange': 'Middle Ex.', 'allgather': 'Allgather'}
    drawn = set()
    for sel in selected:
        for pair, ts, te in [(sel['left_pair'], sel['left_start'], sel['left_end']),
                             (sel['right_pair'], sel['right_start'], sel['right_end'])]:
            if pair is None or ts is None:
                continue
            p = (min(pair), max(pair))
            if p not in gap_y:
                continue
            lbl = phase_labels[sel['phase']] if sel['phase'] not in drawn else None
            drawn.add(sel['phase'])
            ax.barh(gap_y[p], t2x(te) - t2x(ts), left=t2x(ts), height=gap_h,
                    color=PAIR_COLORS[p], edgecolor=c["fg"], linewidth=1.5,
                    hatch=PHASE_HATCH[sel['phase']], alpha=0.95, zorder=4, label=lbl)
        ax.vlines(t2x(sel['selected_time']), ymin=-0.2, ymax=3.25, color=c["marker"], lw=0.8,
                  linestyle='--', alpha=0.6, zorder=3)
        ax.text(t2x(sel['selected_time']), Y_PHASE, phase_labels[sel['phase']], rotation=90,
                ha='center', va='bottom', fontsize=9,
                bbox=dict(fc=c["box"], ec="none", alpha=0.8, pad=0.2), zorder=10)

    for p, y in plane_y.items():
        ax.axhline(y, color=c["rail"], lw=1.2, zorder=1)
        ax.text(-0.085, y, rf'$P_{p}$', ha='right', va='center', fontsize=12, color=c["fg"],
                zorder=10)

    tick_secs = list(range(0, ORBITAL_PERIOD + 1, 1800))
    tick_labels = []
    for s in tick_secs:
        h, m = s // 3600, (s % 3600) // 60
        if h == 0:
            tick_labels.append(rf'${m}$m')
        elif m == 0:
            tick_labels.append(rf'${h}$h')
        else:
            tick_labels.append(rf'${h}$h\,${m}$m')
    ax.set_xticks([s / ORBITAL_PERIOD for s in tick_secs])
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(-0.10, 1.02)
    ax.set_ylim(-0.35, 4.2)
    ax.set_xlabel('Time into simulation')
    ax.set_yticks([])

    legend_handles = [
        mpatches.Patch(color=c["rail"], alpha=0.4, label='Available window'),
        mpatches.Patch(facecolor='#378ADD', edgecolor=c["fg"],
                       label=r'Scatter ($P_1{\leftrightarrow}P_2$, $P_3{\leftrightarrow}P_4$)'),
        mpatches.Patch(facecolor='#7F77DD', edgecolor=c["fg"], hatch='///',
                       label=r'Middle exchange ($P_2{\leftrightarrow}P_3$)'),
        mpatches.Patch(facecolor='#D85A30', edgecolor=c["fg"], hatch='...',
                       label=r'Allgather ($P_1{\leftrightarrow}P_2$, $P_3{\leftrightarrow}P_4$)'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', framealpha=0.9)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    return fig


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--windows", default=STK_ISL_4C, help="10s_4c `_inter` window CSV")
    args = ap.parse_args()
    colours = setup(RC, args.theme)
    df = load_df(args.windows)
    selected = simulate_waterfall(df)
    for s in selected:
        print(f"  {s['phase']:16s} selected @ {(s['selected_time'] - T0) / 60:6.1f} min  "
              f"left={s['left_pair']} right={s['right_pair']}")
    save(plot(df, selected, colours), "fig3_waterfall_windows", args)


if __name__ == "__main__":
    main()
