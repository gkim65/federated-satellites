import flwr as fl

import os
import ray

import gc
from project.client.client import client_fn_femnist, client_fn_EuroSAT,client_fn_CIFAR10
from typing import Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    EvaluateRes,
    FitRes,
    Scalar,
)
import numpy as np
import pandas as pd

from omegaconf import DictConfig, OmegaConf
import hydra

from project.fed.strategies.fedsat_gen import FedSatGen

from configparser import ConfigParser
import shutil

import wandb
import gdown
from pathlib import Path

# #############################################################################
# Federating pipeline with Flower
# #############################################################################
@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    print("\nConfig:")
    print(OmegaConf.to_yaml(cfg))

    #TODO: set up multirun
    ### Run all configuration files
    # for file_name in os.listdir("config_files"):

    config_dict= {"name": cfg.name,
                "round": cfg.fl.round,
                "epochs": cfg.fl.epochs,
                "client_cpu": cfg.fl.client_cpu,
                "client_gpu": cfg.fl.client_gpu,
                "trial": cfg.trial,
                "dataset": cfg.dataset,
                "alg": cfg.alg,
                "clients": cfg.stk.n_sat_in_cluster*cfg.stk.n_cluster,
                "client_limit": cfg.stk.client_limit,
                "sim_fname" : cfg.stk.sim_fname,
                "n_sat_in_cluster" : cfg.stk.n_sat_in_cluster,
                "n_cluster" : cfg.stk.n_cluster,
                "gs_locations" : cfg.stk.gs_locations,
                "learning_rate": cfg.ml.learning_rate,
                "momentum": cfg.ml.momentum,
                "wait_time" : cfg.ml.wait_time,
                "slrum" : cfg.slrum,
                "prox_term": cfg.prox_term,
                "data_rate": cfg.data_rate,
                "power_consumption_per_epoch": cfg.power_consumption_per_epoch
            }
    

    t_name = cfg.wandb.proj_name
    for keys in config_dict.keys():
        print(keys)
        if keys != "sim_fname" and keys != "gs_locations" and keys != "slrum"  and keys != "client_cpu"  and keys != "client_gpu":
            t_name = t_name + "_"+keys[:1]+str(config_dict[keys])
        
    # making sure to run multiple trials for each run
    for i in range(int(config_dict["trial"])):
        
        if cfg.wandb.use:
            ### Saving to Weights and Biases
            wandb.init(
                entity=cfg.wandb.entity,
                # set the wandb project where this run will be logged
                project=t_name,
                # track hyperparameters and run metadata
                config=config_dict
            )
            results = {}

        def fit_config(server_round: int):  
            config = config_dict
            return config
    
        try:
            
            try:
                
                alg = config_dict["alg"]
                name = config_dict["name"]
                if os.path.exists(f'/datasets/{alg}/times_{name}.csv'):
                    os.remove(f'/datasets/{alg}/times_{name}.csv')
                    print(f'/datasets/{alg}/times_{name}.csv')
                    print("deleted")
                if os.path.exists(f"/datasets/{alg}/model_files_{name}"):
                    shutil.rmtree(f"/datasets/{alg}/model_files_{name}")
                    print(f"/datasets/{alg}/model_files_{name}")
                    print("deleted")
                folder_name = f"/datasets/{alg}/model_files_{name}"
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                    print("made ", folder_name )

            except:
                print("no files to delete")
            
            if config_dict["dataset"] == "FEMNIST":
                client_fn = client_fn_femnist
            if config_dict["dataset"] == "EUROSAT":
                client_fn = client_fn_EuroSAT
            if config_dict["dataset"] == "CIFAR10":
                client_fn = client_fn_CIFAR10
            print(config_dict["clients"])
            my_client_resources = {'num_cpus': float(config_dict["client_cpu"]), 'num_gpus': float(config_dict["client_gpu"])}
            results = fl.simulation.start_simulation(
                num_clients= int(config_dict["clients"]),
                # clients_ids =[str(c_id) for c_id in range(int(config_dict["clients"]))],
                client_fn=client_fn,
                config=fl.server.ServerConfig(num_rounds=int(config_dict["round"])),
                
                strategy=FedSatGen(
                    on_fit_config_fn=fit_config, 
                    satellite_access_csv = config_dict["sim_fname"],
                    time_wait = int(config_dict["wait_time"])
                    ),
                client_resources = my_client_resources 
                    
            )
        except:
            ray.shutdown()
            gc.collect()

            if cfg.wandb.use:
                wandb.finish()
            try:
                name = config_dict["name"]
                alg = config_dict["alg"]
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

        if cfg.wandb.use:
            wandb.finish()
        try:
            name = config_dict["name"]
            alg = config_dict["alg"]
            if os.path.exists(f'/datasets/{alg}/times_{name}.csv'):
                os.remove(f'/datasets/{alg}/times_{name}.csv')
                print(f'/datasets/{alg}/times_{name}.csv')
            if os.path.exists(f"/datasets/{alg}/model_files_{name}"):
                shutil.rmtree(f"/datasets/{alg}/model_files_{name}")
                print(f"/datasets/{alg}/model_files_{name}")
            print("deleted")
        except:
            print("no model files to delete")

if __name__ == "__main__":
    main()