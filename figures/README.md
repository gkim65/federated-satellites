# Paper figures and tables

Scripts that regenerate the AutoFLSat results in the IEEE Aerospace 2027 paper. Run each one
from the repository root with the main environment. Outputs go to `figures/out/`, which is
gitignored, as PDF, SVG and PNG.

| Paper item | Command | Input |
|---|---|---|
| Fig. 3, waterfall schedule (P = 4, 5 h) | `uv run python -m figures.fig3_waterfall` | `10s_4c` ISL CSV |
| Fig. 5 and Table 2 accuracy, EuroSAT convergence (P = 2, 3, 4) | `uv run python -m figures.fig5_convergence` | wandb run histories |
| Table 3, WCTC under ISL dropout (P = 4) | `uv run python -m figures.table3_dropout` | `10s_4c` ISL CSV |
| Appendix, ISL windows per adjacent plane pair | `uv run python -m figures.window_counts` | `10s_4c` ISL CSV |

Common options:

- **`--theme dark`** switches to light-on-dark colours for slides. **`--transparent`** saves with
  a transparent background.
- **`--windows CSV`** takes any `10s_4c` `_inter` file (Figure 3, Table 3 and window counts).
  The default is the STK export `datasets/landsat/10s_4c_s_landsat_star_inter.csv`, from
  `python -m project.utils.stk`. A brahe file from `windowgen` (see `windowgen/README.md`) gives
  the same Figure 3 window selections, the same window counts, and an identical Table 3.

## Accuracy curves and wandb

`fig5_convergence` reads each run's logged accuracy from `figures/cache/fig5_P{P}.csv`. On the
first run, or with `--refresh`, it downloads these histories from wandb:

1. Copy `figures/wandb_runs.example.yaml` to `figures/wandb_runs.local.yaml`.
2. Fill in your run paths (`entity/project/run_id`).
3. Log in with `wandb login`.

The local YAML and the cache are gitignored, so run paths and downloaded histories are not
committed.

The P = 3 run logged only `acc` and stopped at about cycle 48. As in the paper figure, its curve
is extended to the axis limit with copies of its last value. `--no-pad` draws it ending at its
last logged round instead.

## Table 3 settings

The 0% row is deterministic. 23.81 h is reproduced by any local training time from 90 to 215 s;
the script's default is `--local-train-s 120`. The other rows depend on which windows are
randomly dropped:

- **`--seed N`** makes a single table repeatable.
- **`--seeds 30`** prints min / median / max over seeds 0-29. The values reported in the paper
  fall within that range.
