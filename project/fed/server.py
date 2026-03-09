import flwr as fl

import os
import ray

import gc
from project.client.client import client_fn_femnist, client_fn_EuroSAT,client_fn_CIFAR10,client_fn_mnist
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
                "power_consumption_per_epoch": cfg.power_consumption_per_epoch,
                "dropout_rate": cfg.dropout_rate
            }
    

    t_name = cfg.wandb.proj_name
    for keys in config_dict.keys():
        print(keys)
        if keys != "sim_fname" and keys != "gs_locations" and keys != "slrum"  and keys != "client_cpu"  and keys != "client_gpu":
            t_name = t_name + "_"+keys[:1]+str(config_dict[keys])
        
    for i in range(int(config_dict["trial"])):

        # Reinitialize Ray cleanly for each trial
        if ray.is_initialized():
            ray.shutdown()
        ray.init(ignore_reinit_error=True)

        if cfg.wandb.use:
            wandb.init(
                entity=cfg.wandb.entity,
                project=t_name,
                config=config_dict
            )


        def fit_config(server_round: int):  
            config = config_dict
            return config
        # Clean up any leftover files from previous trial
        try:
            alg  = config_dict["alg"]
            name = config_dict["name"]
            if os.path.exists(f'/datasets/{alg}/times_{name}.csv'):
                os.remove(f'/datasets/{alg}/times_{name}.csv')
            if os.path.exists(f"/datasets/{alg}/model_files_{name}"):
                shutil.rmtree(f"/datasets/{alg}/model_files_{name}")
            folder_name = f"/datasets/{alg}/model_files_{name}"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
                print("made ", folder_name)
        except Exception as e:
            print(f"File cleanup warning: {e}")

        # Select client function
        if config_dict["dataset"] == "FEMNIST":
            client_fn = client_fn_femnist
        elif config_dict["dataset"] == "EUROSAT":
            client_fn = client_fn_EuroSAT
        elif config_dict["dataset"] == "CIFAR10":
            client_fn = client_fn_CIFAR10
        elif config_dict["dataset"] == "MNIST":
            client_fn = client_fn_mnist

        try:
            print(f"\n=== Starting Trial {i+1}/{config_dict['trial']} ===")
            print(config_dict["clients"])

            my_client_resources = {
                'num_cpus': float(config_dict["client_cpu"]),
                'num_gpus': float(config_dict["client_gpu"])
            }

            results = fl.simulation.start_simulation(
                num_clients=int(config_dict["clients"]),
                client_fn=client_fn,
                config=fl.server.ServerConfig(num_rounds=int(config_dict["round"])),
                strategy=FedSatGen(
                    on_fit_config_fn=fit_config,
                    satellite_access_csv=config_dict["sim_fname"],
                    time_wait=int(config_dict["wait_time"])
                ),
                client_resources=my_client_resources
            )
            print(f"Trial {i+1} completed successfully")

        except Exception as e:
            print(f"Trial {i+1} crashed with error: {e}")
            import traceback
            traceback.print_exc()  # print full stack trace so you can see what happened

        finally:
            # Always clean up after each trial regardless of success or failure
            ray.shutdown()
            gc.collect()

            if cfg.wandb.use:
                wandb.finish()

            try:
                alg  = config_dict["alg"]
                name = config_dict["name"]
                if os.path.exists(f'/datasets/{alg}/times_{name}.csv'):
                    os.remove(f'/datasets/{alg}/times_{name}.csv')
                if os.path.exists(f"/datasets/{alg}/model_files_{name}"):
                    shutil.rmtree(f"/datasets/{alg}/model_files_{name}")
                print(f"Trial {i+1} cleanup complete")
            except Exception as e:
                print(f"Post-trial cleanup warning: {e}")

if __name__ == "__main__":
    main()