# federated-satellites


## Running an FL experiment

1) First in the base `federated_satellites` directory, open the `config_maker.py` file and make all the config files you want to run for different tests. 
    - This will all be saved in a folder saved into your local directories `config_files`. 
2) Then, just run `python server.py` in the base `federated_satellites` directory. 
    - If you don't have the stk csv file ready on hand, it will download the main one for you for running your tests.

## Setup
### Running the FEMNIST Model
Need to first go into the `FEMNIST_tests` folder, and run the femnist.py in the directory indicated; this way the femnist dataset gets downloaded into the correct folder.

Then, you can start using FEMNIST.


#### Active Todos: UPDATE PLS
- `client_fn` can't take in any inputs so I can't force it to take in config parameter to run a specific dataset
- federated_sat function/strategy