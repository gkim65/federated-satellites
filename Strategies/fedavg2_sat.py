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

from Strategies.utils import choose_sat_csv
from datetime import timedelta
from datetime import datetime
import wandb
import numpy as np
import pandas as pd
import ray
import gc

class FedAvg2Sat(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        satellite_access_csv: "Strategies/csv_stk/Chain2_Access_Data_1_sat_5_plane.csv",
        time_wait: 7
    ) -> None:

        super().__init__()
        self.on_fit_config_fn = on_fit_config_fn
        self.counter = 0
        self.time_wait = time_wait
        self.satellite_access_csv_name = satellite_access_csv
        self.satellite_access_csv = pd.read_csv(satellite_access_csv)

        og_s = int(self.satellite_access_csv_name[19:].split("_")[0][:-1])
        og_c = int(self.satellite_access_csv_name[19:].split("_")[1][:-1])
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
        
        # start_time = self.satellite_access_csv['Start Time (UTCG)'].iloc[self.counter]
        start_time_sec = self.satellite_access_csv['Start Time Seconds Cumulative'].iloc[self.counter]

        delta = timedelta(hours=2)
        client_list = np.zeros(int(config["clients"]))
        client_time_list = np.zeros(int(config["clients"]))
        
        if int(config["clients"]) < int(config["client_limit"]):
            limit = int(config["clients"])
        else:
            limit = int(config["client_limit"])
        done_count = 0
        client_twice = []
        while done_count < limit:
            cluster_id = ((self.satellite_access_csv['cluster_num'].iloc[self.counter]))
            satellite_id = ((self.satellite_access_csv['sat_num'].iloc[self.counter]))
            client_id = int(int(config["n_sat_in_cluster"])*(cluster_id/self.factor_c)-(int(config["n_sat_in_cluster"])-(satellite_id/self.factor_s))-1)
            client_list[client_id] += 1
            if client_list[client_id] == 2:
                done_count +=1
                client_twice.append(client_id)
            if client_time_list[client_id] == 0 :
                client_time_list[client_id] = self.satellite_access_csv['Start Time Seconds Cumulative'].iloc[self.counter]
            else:
                client_time_list[client_id] = self.satellite_access_csv['Start Time Seconds Cumulative'].iloc[self.counter] - client_time_list[client_id]
            self.counter += 1

        # stop_time = self.satellite_access_csv['Start Time (UTCG)'].iloc[self.counter]
        stop_time_sec = self.satellite_access_csv['Start Time Seconds Cumulative'].iloc[self.counter]
        idle_time_total = 0
        # print(client_time_list)
        client_time_list = [client_time_list[x] for x in range(len(client_time_list)) if x in client_twice]
        # print(client_time_list)
        for time in client_time_list:
            idle_time_total += (stop_time_sec - start_time_sec)-time 
        idle_time_avg = idle_time_total/int(config["client_limit"])
        wandb.log({"start_time_sec": start_time_sec, "stop_time_sec": stop_time_sec, "server_round": server_round,"duration" : stop_time_sec - start_time_sec, "idle_time_total": idle_time_total,"idle_time_avg": idle_time_avg})


        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        # clients = client_manager.all()
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        x = [client for client in clients if int(client.cid) in client_twice]
        print(x)
        self.satellite_client_list = [int(client.cid) for client in clients]
        # Return client/config pairs
        return [(client, fit_ins) for client in x]
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        # Return client/config pairs

        x = [client for client in clients if int(client.cid) in self.satellite_client_list]
        return [(client, evaluate_ins) for client in x]

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
