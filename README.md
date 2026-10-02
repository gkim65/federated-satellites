# federated-satellites

This repository contains FLySTacK, a simulation platform for federated learning (FL) in satellite
constellations, and the code for the AutoFLSat algorithm. Satellites are Flower clients, and
contact windows decide when each one can talk to a ground station or to another satellite.
Those windows can come from two sources, which produce the same CSV format:

- **brahe (recommended).** [`windowgen/`](windowgen/README.md) generates ground-station and
  inter-satellite-link windows for any circular Walker constellation with the open-source
  [brahe](https://docs.brahe.space/latest/) astrodynamics library.
- **STK.** The pre-computed exports used in earlier papers, downloaded with
  `python -m project.utils.stk`. brahe reproduces them: inter-satellite windows to within 1 s,
  ground passes to within 0.4% in count (see `windowgen/README.md`).

Papers: https://arxiv.org/abs/2411.00263 and https://arxiv.org/abs/2511.14889

## Quickstart

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```
git clone https://github.com/gkim65/federated-satellites.git
cd federated-satellites
uv sync

# 1. Contact windows for a 3-plane x 10-satellite Walker star, generated with brahe (about 30 s)
cd windowgen
uv sync
uv run flystack-windows configs/stk_fit_10s_3c.yaml --out ../datasets/landsat
cd ..

# 2. A short AutoFLSat run on MNIST over those windows (MNIST downloads automatically)
uv run python -m project.fed.server wandb.use=False alg=AutoFLSatWaterfall dataset=MNIST \
    stk.sim_fname=datasets/landsat/10s_3c_brahe_star_inter.csv \
    stk.n_sat_in_cluster=10 stk.n_cluster=3 fl.round=6 fl.epochs=1 trial=1
```

`windowgen` has its own environment, because brahe and flwr 1.19 need incompatible versions of
`rich`. To simulate your own constellation, copy a file in `windowgen/configs/` and edit it. The
orbit, phasing, ground stations, elevation mask and link model are all fields in the YAML.

## Datasets

Run these from the repository root. Each one fills `datasets/`.

- **FEMNIST:** `uv run python -m project.utils.femnist`
- **EuroSAT:** `uv run python -m project.utils.eurosat`
- **CIFAR10, MNIST:** downloaded by torchvision on first use
- **STK window CSVs (optional):** `uv run python -m project.utils.stk`

## Running simulations

Settings live in `project/config/config.yaml` and can be overridden on the command line with
Hydra:

```
uv run python -m project.fed.server alg=AutoFLSatWaterfall dataset=EUROSAT stk.n_cluster=4
uv run python -m project.fed.server --multirun stk.n_cluster=2,3,4
```

Set `wandb.use=False` to run without a Weights & Biases account. With `wandb.use=True`, set
`wandb.entity` to your own entity.

## Paper figures

[`figures/`](figures/README.md) regenerates the AutoFLSat paper's Figures 3 and 5 and Tables 2
and 3, from either STK or brahe windows.

## Repository layout

- **`project/config/config.yaml`** — all simulation settings. Key fields:
  - `alg` — FL algorithm:
    - `fedAvgSat`, `fedProxSat`, `fedBuffSat` — ground-station FL.
    - `...2Sat` variants — add scheduling.
    - `...3Sat` variants — add intra-plane links; use with 10+ satellites per plane.
    - `AutoFLSat2` — hierarchical, ground-free.
    - `AutoFLSatWaterfall` — the AutoFLSat waterfall all-reduce over inter-plane links.
  - `dataset` — `FEMNIST`, `EUROSAT`, `CIFAR10` or `MNIST`.
  - `fl.round`, `fl.epochs`, `trial` — Flower rounds, local epochs per round, repeated trials.
  - `stk.sim_fname` — the window CSV. Ground-station algorithms take a ground file
    (`{S}s_{P}c_..._star.csv`); AutoFLSat algorithms take an inter-satellite file
    (`..._inter.csv`).
  - `stk.n_sat_in_cluster`, `stk.n_cluster` — satellites per plane and planes to simulate. They
    must divide the S and P in the file name (e.g. 1, 2, 5 or 10 satellites from a 10-per-plane
    file).
  - `stk.client_limit`, `stk.gs_locations` — clients per round, and which ground stations to use.
  - `data_rate`, `power_consumption_per_epoch` — model transfer time per pass (s) and training
    time per epoch (s).
  - `dropout_rate` — probability that an inter-satellite window fails (AutoFLSatWaterfall).
- **`project/fed/server.py`** — entry point; runs Flower's simulation with `FedSatGen`
  (`project/fed/strategies/fedsat_gen.py`), which selects clients from the contact windows.
- **`project/fed/strategies/`** — one module per algorithm.
- **`project/client/client.py`** — the Flower client and per-dataset data partitioning.
- **`windowgen/`** — brahe contact-window generator (own `uv` environment).
- **`figures/`** — scripts that regenerate the AutoFLSat paper figures and tables.

NOTE: the CIFAR10, MNIST and EuroSAT partitions are fixed at 100 non-IID splits, so a simulated
constellation can have at most 100 satellites with those datasets.


If you would like to use this repo, or this work in any way, please cite the following paper in your research!

```
@article{kim2024space,
  title={Space for Improvement: Navigating the Design Space for Federated Learning in Satellite Constellations},
  author={Kim, Grace and Powell, Luca and Svoboda, Filip and Lane, Nicholas},
  journal={arXiv preprint arXiv:2411.00263},
  year={2024}
}
```
