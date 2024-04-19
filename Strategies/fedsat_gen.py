"""
Federated Averaging (FedAvg) modified with satellite constellation implementations

Originally from 2020 Flower Labs GmbHl, implmenting:
Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.
Paper: arxiv.org/abs/1602.05629
"""
import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from typing import Callable, Dict, List, Optional, Tuple, Union

from Strategies.utils import choose_sat_csv, fedAvgSat, fedAvg2Sat

import pandas as pd
import wandb

class FedSatGen(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        satellite_access_csv: "/datasets/landsat/10s_10c_s_landsat.csv",
        time_wait: 7
    ) -> None:

        super().__init__()
        self.on_fit_config_fn = on_fit_config_fn
        self.counter = 0
        self.time_wait = time_wait
        self.satellite_access_csv_name = satellite_access_csv
        self.satellite_access_csv = pd.read_csv(satellite_access_csv)

        # TODO: Change this it will keep causing index erros if you don't and switch the csv files
        og_s = int(self.satellite_access_csv_name[18:].split("_")[0][:-1])
        og_c = int(self.satellite_access_csv_name[18:].split("_")[1][:-1])
        config = self.on_fit_config_fn(1)
        gs = config["gs_locations"][1:-1].split(",")
        self.factor_s = og_s / int(config["n_sat_in_cluster"])
        self.factor_c = og_c / int(config["n_cluster"])
        # choose only satellites that we want
        self.satellite_access_csv = choose_sat_csv(self.satellite_access_csv, og_s, og_c, int(config["n_sat_in_cluster"]), int(config["n_cluster"]),gs)
        self.satellite_client_list = []
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}

        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)
        
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        # clients = client_manager.all()
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        if config["alg"] == "fedAvgSat":
            chosen_clients, self.counter = fedAvgSat(self.satellite_access_csv, 
                                       self.counter,
                                       int(config["clients"]),
                                       int(config["client_limit"]),
                                       int(config["n_sat_in_cluster"]),
                                       self.factor_s,
                                       self.factor_c,
                                       server_round,
                                       clients)
        elif config["alg"] == "fedAvg2Sat":
            chosen_clients, self.counter = fedAvg2Sat(self.satellite_access_csv, 
                                       self.counter,
                                       int(config["clients"]),
                                       int(config["client_limit"]),
                                       int(config["n_sat_in_cluster"]),
                                       self.factor_s,
                                       self.factor_c,
                                       server_round,
                                       clients)
            
        return [(client, fit_ins) for client in chosen_clients]
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}

        # using the previous config but usually this:
        """
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        """
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        if config["alg"] == "fedAvgSat":
            chosen_clients, self.counter = fedAvgSat(self.satellite_access_csv, 
                                       self.counter,
                                       int(config["clients"]),
                                       int(config["client_limit"]),
                                       int(config["n_sat_in_cluster"]),
                                       self.factor_s,
                                       self.factor_c,
                                       server_round,
                                       clients)
        elif config["alg"] == "fedAvg2Sat":
            chosen_clients, self.counter = fedAvg2Sat(self.satellite_access_csv, 
                                       self.counter,
                                       int(config["clients"]),
                                       int(config["client_limit"]),
                                       int(config["n_sat_in_cluster"]),
                                       self.factor_s,
                                       self.factor_c,
                                       server_round,
                                       clients)
            
        # Return client/config pairs
        return [(client, evaluate_ins) for client in chosen_clients]

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
        wandb.log({"acc": aggregated_accuracy, "loss": aggregated_loss, "server_round": server_round})
        
        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}
