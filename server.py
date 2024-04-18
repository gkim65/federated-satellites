import flwr as fl
import ray
import gc
from client import client_fn
from typing import Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    EvaluateRes,
    FitRes,
    Scalar,
)
import numpy as np
import pandas as pd

from Strategies.fedavg_sat import FedAvgSat

from configparser import ConfigParser
import os

import wandb
import gdown

# #############################################################################
# Federating pipeline with Flower
# #############################################################################

### Run all configuration files
for file_name in os.listdir("config_files"):

    # Read config.ini file
    config_object = ConfigParser()
    config_object.read("config_files/"+file_name)

    if not os.path.exists(config_object["TEST_CONFIG"]["sim_fname"]):
        # url = "https://drive.google.com/file/d/1zEiHCmMmx_qz17nSmrDzgqHlheYTbsvF/view?usp=sharing"
        url = "https://drive.google.com/file/d/1ab37NCbS1EUx5cqDaMv2V7Fk_g5chhlv/view?usp=sharing"
        gdown.download(url, config_object["TEST_CONFIG"]["sim_fname"], fuzzy=True)
    
    # making sure to run multiple trials for each run
    for i in range(int(config_object["TEST_CONFIG"]["trial"])):

        # TODO: FIX Ways i send files in for testing for sim so i don't need to send full file
        t_name = "Run"
        for keys in config_object["TEST_CONFIG"].keys():
            print(keys)
            if keys != "sim_fname" and keys != "gs_locations":
                t_name = t_name + "_"+keys[:1]+str(config_object["TEST_CONFIG"][keys])
        
        ### Saving to Weights and Biases
        wandb.init(
            # set the wandb project where this run will be logged
            project=t_name,
            # track hyperparameters and run metadata
            config= {"name": config_object["TEST_CONFIG"]["name"],
                "round": config_object["TEST_CONFIG"]["round"],
                "epochs": config_object["TEST_CONFIG"]["epochs"],
                "trial": config_object["TEST_CONFIG"]["trial"],
                "clients": config_object["TEST_CONFIG"]["clients"],
                "client_limit": config_object["TEST_CONFIG"]["client_limit"],
                "dataset": config_object["TEST_CONFIG"]["dataset"],
                "learning_rate": config_object["TEST_CONFIG"]["learning_rate"],
                "momentum": config_object["TEST_CONFIG"]["momentum"],
                "wait_time" : config_object["TEST_CONFIG"]["wait_time"],
                "sim_fname" : config_object["TEST_CONFIG"]["sim_fname"],
                "n_sat_in_cluster" : config_object["TEST_CONFIG"]["n_sat_in_cluster"],
                "n_cluster" : config_object["TEST_CONFIG"]["n_cluster"],
                "gs_locations" : config_object["TEST_CONFIG"]["gs_locations"][1:-1].split(",")
            }
        )
        results = {}

        def fit_config(server_round: int):  
            config = config_object["TEST_CONFIG"]
            return config

        try:
            my_client_resources = {'num_cpus': 1, 'num_gpus': 0.1}
            results = fl.simulation.start_simulation(
                num_clients= int(config_object["TEST_CONFIG"]["clients"]),
                clients_ids =[str(c_id) for c_id in range(int(config_object["TEST_CONFIG"]["clients"]))],
                client_fn=client_fn,
                config=fl.server.ServerConfig(num_rounds=int(config_object["TEST_CONFIG"]["round"])),
                strategy=FedAvgSat(
                    on_fit_config_fn=fit_config, 
                    satellite_access_csv = config_object["TEST_CONFIG"]["sim_fname"],
                    time_wait = int(config_object["TEST_CONFIG"]["wait_time"])
                    ),
                client_resources = my_client_resources 
                    
            )
        except:
            ray.shutdown()
            gc.collect()
            wandb.finish()

        ray.shutdown()
        gc.collect()
        wandb.finish()

