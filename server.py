import flwr as fl

import os
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

from Strategies.fedsat_gen import FedSatGen

from configparser import ConfigParser
import shutil

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

    # TODO: Bring back in the future but for now accessing from the datasets
    # if not os.path.exists(config_object["TEST_CONFIG"]["sim_fname"]):
    #     # url = "https://drive.google.com/file/d/1zEiHCmMmx_qz17nSmrDzgqHlheYTbsvF/view?usp=sharing"
    #     url = "https://drive.google.com/file/d/1ab37NCbS1EUx5cqDaMv2V7Fk_g5chhlv/view?usp=sharing"
    #     gdown.download(url, config_object["TEST_CONFIG"]["sim_fname"], fuzzy=True)
    # TODO: FIX Ways i send files in for testing for sim so i don't need to send full file
    t_name = "WorkFB_5_14"
    for keys in config_object["TEST_CONFIG"].keys():
        print(keys)
        if keys != "sim_fname" and keys != "gs_locations" and keys != "slrum"  and keys != "client_cpu"  and keys != "client_gpu":
            t_name = t_name + "_"+keys[:1]+str(config_object["TEST_CONFIG"][keys])
        
    # making sure to run multiple trials for each run
    for i in range(int(config_object["TEST_CONFIG"]["trial"])):

        
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
                "alg": config_object["TEST_CONFIG"]["alg"],
                "learning_rate": config_object["TEST_CONFIG"]["learning_rate"],
                "momentum": config_object["TEST_CONFIG"]["momentum"],
                "wait_time" : config_object["TEST_CONFIG"]["wait_time"],
                "sim_fname" : config_object["TEST_CONFIG"]["sim_fname"],
                "n_sat_in_cluster" : config_object["TEST_CONFIG"]["n_sat_in_cluster"],
                "n_cluster" : config_object["TEST_CONFIG"]["n_cluster"],
                "slrum" : config_object["TEST_CONFIG"]["slrum"],
                "client_cpu": config_object["TEST_CONFIG"]["client_cpu"],
                "client_gpu": config_object["TEST_CONFIG"]["client_gpu"],
                "prox_term": config_object["TEST_CONFIG"]["prox_term"],
                "gs_locations" : config_object["TEST_CONFIG"]["gs_locations"][1:-1].split(",")
            }
        )
        results = {}

        def fit_config(server_round: int):  
            config = config_object["TEST_CONFIG"]
            return config
    
        try:
            
            try:
                
                alg = config_object["TEST_CONFIG"]["alg"]
                name = config_object["TEST_CONFIG"]["name"]
                if os.path.exists(f'/datasets/{alg}/times_{name}.csv'):
                    os.remove(f'/datasets/{alg}/times_{name}.csv')
                    print(f'/datasets/{alg}/times_{name}.csv')
                if os.path.exists(f"/datasets/{alg}/model_files_{name}"):
                    shutil.rmtree(f"/datasets/{alg}/model_files_{name}")
                    print(f"/datasets/{alg}/model_files_{name}")
                print("deleted")
                folder_name = f"/datasets/{alg}/model_files_{name}"
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)

            except:
                print("no files to delete")
            my_client_resources = {'num_cpus': float(config_object["TEST_CONFIG"]["client_cpu"]), 'num_gpus': float(config_object["TEST_CONFIG"]["client_gpu"])}
            results = fl.simulation.start_simulation(
                num_clients= int(config_object["TEST_CONFIG"]["clients"]),
                clients_ids =[str(c_id) for c_id in range(int(config_object["TEST_CONFIG"]["clients"]))],
                client_fn=client_fn,
                config=fl.server.ServerConfig(num_rounds=int(config_object["TEST_CONFIG"]["round"])),
                
                strategy=FedSatGen(
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
            try:
                name = config_object["TEST_CONFIG"]["name"]
                alg = config_object["TEST_CONFIG"]["alg"]
                if os.path.exists(f'/datasets/{alg}/times_{name}.csv'):
                    os.remove(f'/datasets/{alg}/times_{name}.csv')
                    print(f'/datasets/{alg}/times_{name}.csv')
                if os.path.exists(f"/datasets/{alg}/model_files_{name}"):
                    shutil.rmtree(f"/datasets/{alg}/model_files_{name}")
                    print(f"/datasets/{alg}/model_files_{name}")
                print("deleted")
            except:
                print("no model files to delete")
        ray.shutdown()
        gc.collect()
        wandb.finish()
        try:
            name = config_object["TEST_CONFIG"]["name"]
            alg = config_object["TEST_CONFIG"]["alg"]
            if os.path.exists(f'/datasets/{alg}/times_{name}.csv'):
                os.remove(f'/datasets/{alg}/times_{name}.csv')
                print(f'/datasets/{alg}/times_{name}.csv')
            if os.path.exists(f"/datasets/{alg}/model_files_{name}"):
                shutil.rmtree(f"/datasets/{alg}/model_files_{name}")
                print(f"/datasets/{alg}/model_files_{name}")
            print("deleted")
        except:
            print("no model files to delete")

