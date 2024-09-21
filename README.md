# federated-satellites


The code within this repository is arranged in the following format:

At the base of the folder, you'll find 3 main files that one would interact with:

- `config_maker.py` : The main script that would be modified by the user, where a multitude of parameters are available to play around with. Some most important parameters are outlined below:
    - **Name**: First name of the simulation
    - **Round**: Number of FL rounds to complete
    - **Epochs**: Number of epochs to train on each round, dependent on model
    - **Trial**: Number of times to run this one script, for comparison of runs
    - **Clients**: Number of clients tested in the simulation
    - **Client Limit**: Number of clients that are limited to join in each FL round
    - **Dataset**: Currently supports: "FEMNIST","EUROSAT", "CIFAR10", however will need to open individual scripts to download each dataset in the ../datasets folder in the future.
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
        - "AutoFLSat" (still has bugs)
        - "AutoFL2Sat" (Proper hierarchical framework)
        ** all of these FL algorithms can be looked into deeper in the `/Strategies` folder
    - **sim_fname**: this is the file of the stk csv that is entered into the flower pipeline, will need to be saved by downloading from a google drive link here: [TODO]
    - **n_sat_in_cluster**: Depending on the size of the initial constellation tested in the sim_fname file (which is saved in the format 10s_10c), the two numbers indicates how many satellites per cluster and #s of clusters are in the constellation. Factors of these numbers can be tested within the simulation, so for simulations with 10 satellites per cluster, parameter sweeps can be done over 1, 2, 5, and 10 satellites.
    - **n_cluster**: Same story for this!
    - **prox_term**: only for fedProx, testing how much to add in for proximal term.
- `server.py`: Template file, copies are made with config_maker for various parameter sweeps, as each server.py file looks at all of the config files in the config folder made by config_maker
- `client.py`: A generalized client that can accomodate datasets of "EUROSAT", "FEMNIST", and "CIFAR10" just need to toggle between the options inside the `config_maker`


## Running an FL experiment

1) First in the base `federated_satellites` directory, open the `config_maker.py` file and make all the config files you want to run for different tests. 
    - This will all be saved in a folder saved into your local directories `config_files<number>`. 
2) Then, just run `python server<number>.py` in the base `federated_satellites` directory. 
    - If you don't have the stk csv file ready on hand, it will download the main one for you for running your tests.

### Running the FEMNIST Model
Need to first go into the `FEMNIST_tests` folder, and run the femnist.py in the directory indicated; this way the femnist dataset gets downloaded into the correct folder.

Then, you can start using FEMNIST. This can be done the same way for EUROSAT. The CIFAR10 model should run immediately when you run `python server.py`
