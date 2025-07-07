# federated-satellites


## Instructions for installing dependencies and cloning this repository

First, clone the repository on the folder you'd like to run this in.

```
git clone https://github.com/gkim65/federated-satellites.git
```

You can then `cd` into the folder created from this command, `federated-satellites`

```
cd loc-gsopt
```

And create a virtual environment to download all of your dependencies. I recommend using `uv` which can be installed using the documentation linked [here](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2). 

### Library management using `uv`:

Create a new virtual environment with `uv` on a Mac inside the git folder, and install dependencies existing in the `pyproject.toml` file.

```
uv venv
source .venv/bin/activate 
uv sync
```




## Using this Repository

First, make sure to download the datasets needed for testing, which can be done using the following commands from the base folder `federated-satellites`:

**FEMNIST dataset:**
```
python -m project.utils.femnist
```
**EUROSAT dataset**
```
python -m project.utils.eurosat
```

**CIFAR10** (already included in torch libraries)

**STK CSVs**

```
python -m project.utils.stk
```

These commands should make a new folder called datasets which now has all of the data downloaded for you to use in your experiments.

## Running Scripts

Using the `hydra` config file manager, you can run files by running:

```
python -m project.fed.server
```

You can run sweeps/perform parameter runs

```
python -m project.fed.server --multirun problem.sat_num=1,5,10
```

or just change the parameters directly in `config\config.yaml`. 


## Additional notes:

The code within this repository is arranged in the following format:

At the base of the folder, you'll find 3 main files that one would interact with:

- `project/config/config.yaml` : The main file that would be modified by the user, where a multitude of parameters are available to play around with. Some most important parameters are outlined below:
    - **Round**: Number of FL rounds to complete
    - **Epochs**: Number of epochs to train on each round, dependent on model
    - **Trial**: Number of times to run this one script, for comparison of runs
    - **Clients**: Number of clients tested in the simulation
    - **Client Limit**: Number of clients that are limited to join in each FL round
    - **Dataset**: Currently supports: "FEMNIST","EUROSAT", "CIFAR10", however will need to download individual datasets using the instructions above
    - **Alg**: The different FL algorithms that can be tested, ranges from: 
        - "FedAvgSat"
        - "FedAvg2Sat" (with scheduling)
        - "FedAvg3Sat" (with scheduling and intra sat links, use for clients of 10+ on one cluster)
        - "FedProxSat"
        - "FedProx2Sat" (with scheduling)
        - "FedProx3Sat" (with scheduling and intra sat links, use for clients of 10+ on one cluster)
        - "FedBuffSat"
        - "FedBuff2Sat" (with scheduling)
        - "FedBuff3Sat" (with scheduling and intra sat links, use for clients of 10+ on one cluster)
        - "AutoFL2Sat" (Proper hierarchical framework)
        ** all of these FL algorithms can be looked into deeper in the `/Strategies` folder
    - **sim_fname**: this is the file of the stk csv that is entered into the flower pipeline, will need to be saved by downloading from the steps above. The current example file that is put in should work for all algorithms except the AutoFLSat files (which need the `datasets/landsat/10s_4c_s_landsat_star_inter.csv` instead)
    - **n_sat_in_cluster**: Depending on the size of the initial constellation tested in the sim_fname file (which is saved in the format 10s_10c), the two numbers indicates how many satellites per cluster and #s of clusters are in the constellation. Factors of these numbers can be tested within the simulation, so for simulations with 10 satellites per cluster, parameter sweeps can be done over 1, 2, 5, and 10 satellites.
    - **n_cluster**: Same story for this!
    - **prox_term**: only for fedProx, testing how much to add in for proximal term.
- `project/fed/server.py`: Template file, copies are made with config_maker for various parameter sweeps, as each server.py file looks at all of the config files in the config folder made by config_maker
- `project/client/client.py`: A generalized client that can accomodate datasets of "EUROSAT", "FEMNIST", and "CIFAR10" just need to toggle between the options inside `config.yaml`, additional clients can be made by adding in clients here
- `project/strategies`: a list of strategies already implemented using the csv times listed inside STK


If you would like to use this repo, or this work in any way, please cite the following paper in your research!

```
@article{kim2024space,
  title={Space for Improvement: Navigating the Design Space for Federated Learning in Satellite Constellations},
  author={Kim, Grace and Powell, Luca and Svoboda, Filip and Lane, Nicholas},
  journal={arXiv preprint arXiv:2411.00263},
  year={2024}
}
```
