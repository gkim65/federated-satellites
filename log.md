# Notes:

### Current Implementations


**April 1st 2024**
- revisited github page
- Things implemented so far:
    - Trying to ensure that the different datasets can be easily switched between; set up a `FEMNIST_tests` folder that has all the models and functionality for FEMNISt datasets
    - Weights and biases are now implemented for basic info (config parameters, along with accumulated losses and accuracies) didn't realize `wandb` stood for this lol
    - `config_maker.py` provides an easy way to make config files to just run over for different parameter sweeps
- TODOS:
    - implement `CIFAR-10` and `CIFAR-100` datasets in separate folders
    - `client_fn` can't take in any inputs so I can't force it to take in config parameter to run a specific dataset... right now, still manual but can probably find a way around it later
    


**March 27th 2024**
- Set up on the GPU Cluster, both HPC and CamMLSys
- wrote up how to set up conda for CamMLSys cluster
