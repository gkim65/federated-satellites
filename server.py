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


class Strategy_Sat(FedAvgSat):
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation accuracy using weighted average."""

        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        print(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")
        
        # log metrics to wandb
        run.log({"acc": aggregated_accuracy, "loss": aggregated_loss})
        
        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}


# #############################################################################
# Federating pipeline with Flower
# #############################################################################

### Run all configuration files
for file_name in os.listdir("config_files"):

    # Read config.ini file
    config_object = ConfigParser()
    config_object.read("config_files/"+file_name)

    # making sure to run multiple trials for each run
    for i in range(int(config_object["TEST_CONFIG"]["trial"])):

        t_name = "Run"
        for keys in config_object["TEST_CONFIG"].keys():
            print(keys)
            t_name = t_name + "_"+keys[:1]+str(config_object["TEST_CONFIG"][keys])
        
        ### Saving to Weights and Biases
        run = wandb.init(
            # set the wandb project where this run will be logged
            project=t_name,
            # track hyperparameters and run metadata
            config=config_object["TEST_CONFIG"]
        )
        results = {}


        def fit_config(server_round: int):  
            config = config_object["TEST_CONFIG"]
            return config

        results = fl.simulation.start_simulation(
            num_clients= int(config_object["TEST_CONFIG"]["clients"]),
            clients_ids =[str(c_id) for c_id in range(int(config_object["TEST_CONFIG"]["clients"]))],
            client_fn=client_fn,
            config=fl.server.ServerConfig(num_rounds=int(config_object["TEST_CONFIG"]["round"])),
            strategy=Strategy_Sat(
                on_fit_config_fn=fit_config, 
                satellite_access_csv = "Strategies/csv_stk/Chain2_Access_Data_1_sat_5_plane.csv",
                time_wait = int(config_object["TEST_CONFIG"]["wait_time"])
            )
        )

        # losses_distributed = pd.DataFrame.from_dict({"test": [acc for _, acc in results.losses_distributed]})
        # accuracies_distributed = pd.DataFrame.from_dict({"test": [acc for _, acc in results.metrics_distributed['accuracy']]})
        # if not os.path.exists("results/"+t_name):
        #     os.makedirs("results/"+t_name)
        # losses_distributed.to_csv('results/'+t_name+"/losses_distributed.csv")
        # accuracies_distributed.to_csv('results/'+t_name+"/accuracies_distributed.csv")
        ray.shutdown()
        gc.collect()
        run.finish()

