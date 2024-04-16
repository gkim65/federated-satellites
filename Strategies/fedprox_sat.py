"""
### TODO: FIX TITLE
Federated PROX (FedAvg) modified with satellite constellation implementations

Originally from 2020 Flower Labs GmbHl, implmenting:
Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.
Paper: arxiv.org/abs/1602.05629
"""

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

from Strategies.utils import read_sat_csv
from datetime import timedelta
from datetime import datetime
import wandb
import math

class FedProxSat(fl.server.strategy.FedAvg):
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
        self.satellite_access_csv = read_sat_csv(satellite_access_csv)
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

        start_time = self.satellite_access_csv['Start Time Seconds datetime'].iloc[self.counter]
        delta = timedelta(hours=2)
        client_list = []
        
        # Check if I need this later
        # n_s = int(self.satellite_access_csv_name[19:].split("_")[0][:-1])
        # n_c = int(self.satellite_access_csv_name[19:].split("_")[1][:-1])

        
        while (timedelta(hours=self.time_wait) > delta):
            client_list.append(((self.satellite_access_csv['cluster_num'].iloc[self.counter])-1)*5+self.satellite_access_csv['sat_num'].iloc[self.counter])
            # if self.satellite_access_csv_name == "Strategies/csv_stk/Chain1_Access_Data_9sat_5plane.csv":
            #     client_list.append((int(self.satellite_access_csv['From Object'].iloc[self.counter][-2:-1])-1)*9+int(self.satellite_access_csv['From Object'].iloc[self.counter][-1:]))
            # else:
            #     client_list.append(int(self.satellite_access_csv['From Object'].iloc[self.counter][-2:-1])-1)
            self.counter +=1
            delta = self.satellite_access_csv['Start Time Seconds datetime'].iloc[self.counter]-start_time
            print(delta)
        print(client_list)
        client_twice =[]
        for item in client_list:
            if client_list.count(item)>1:
                client_twice.append(item)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        # clients = client_manager.all()
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        x = [client for client in clients if int(client.cid) in client_twice]
        print("client_list:")
        print(x)
        self.satellite_client_list = client_twice
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
            wandb.log({"acc": 0, "loss": 0, "server_round": server_round})
            print(f"Round {server_round} accuracy aggregated from client results: 0")
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
