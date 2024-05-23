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

from Strategies.utils import choose_sat_csv , choose_sat_csv_auto
from Strategies.fedavg_sat import fedAvgSat
from Strategies.fedavg2_sat import fedAvg2Sat
from Strategies.fedavg3_sat import fedAvg3Sat
from Strategies.fedprox_sat import fedProxSat
from Strategies.fedprox2_sat import fedProx2Sat
from Strategies.fedprox3_sat import fedProx3Sat
from Strategies.fedbuff_sat import fedBuffSat
from Strategies.fedbuff2_sat import fedBuff2Sat
from Strategies.fedbuff3_sat import fedBuff3Sat
from Strategies.AutoFLSat import AutoFLSat

import pandas as pd
import wandb
from copy import deepcopy

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
        config = self.on_fit_config_fn(1)
        self.satellite_access_csv_name = config['sim_fname']
        try:
            self.satellite_access_csv = pd.read_csv(self.satellite_access_csv_name)
        except:
            self.satellite_access_csv = pd.read_csv(".."+self.satellite_access_csv_name)

        # TODO: Change this it will keep causing index erros if you don't and switch the csv files
        og_s = int(self.satellite_access_csv_name.split("/")[-1].split("_")[0][:-1])
        og_c = int(self.satellite_access_csv_name.split("/")[-1].split("_")[1][:-1])


        gs = config["gs_locations"][1:-1].split(",")
        self.factor_s = og_s / int(config["n_sat_in_cluster"])
        self.factor_c = og_c / int(config["n_cluster"])
        # choose only satellites that we want
        if config['alg'] == "AutoFLSat":
            self.satellite_access_csv = choose_sat_csv_auto(self.satellite_access_csv, og_s, og_c, int(config["n_sat_in_cluster"]), int(config["n_cluster"]),gs)
        else:
            self.satellite_access_csv = choose_sat_csv(self.satellite_access_csv, og_s, og_c, int(config["n_sat_in_cluster"]), int(config["n_cluster"]),gs)
        self.satellite_client_list = []
        self.sim_times_start = [0 for i in range(int(config["n_cluster"]))]
        self.sim_times_currents = [0 for i in range(int(config["n_cluster"]))]
        self.cluster_round_starts = [0 for i in range(int(config["n_cluster"]))]
        self.cluster_round_currents = [0 for i in range(int(config["n_cluster"]))]
        self.model_type = "local_cluster"
        self.cluster_num = 0
    
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
        elif config["alg"] == "fedAvg3Sat":
            chosen_clients, self.counter = fedAvg3Sat(self.satellite_access_csv, 
                                        self.counter,
                                        int(config["clients"]),
                                        int(config["client_limit"]),
                                        int(config["n_sat_in_cluster"]),
                                        int(config["epochs"]),
                                        int(config["n_cluster"]),
                                        self.factor_s,
                                        self.factor_c,
                                        server_round,
                                        clients)
        elif config["alg"] == "fedProxSat":
            chosen_clients_times, self.counter = fedProxSat(self.satellite_access_csv, 
                                       self.counter,
                                       int(config["clients"]),
                                       int(config["client_limit"]),
                                       int(config["n_sat_in_cluster"]),
                                       self.factor_s,
                                       self.factor_c,
                                       server_round,
                                       clients)
            return_clients = []
            for client,time in chosen_clients_times:
                fit_ins.config["duration"] = str(time)
                return_clients.append((client, deepcopy(fit_ins)))
            return return_clients
        
        elif config["alg"] == "fedProx2Sat":
            chosen_clients_times, self.counter = fedProx2Sat(self.satellite_access_csv, 
                                       self.counter,
                                       int(config["clients"]),
                                       int(config["client_limit"]),
                                       int(config["n_sat_in_cluster"]),
                                       self.factor_s,
                                       self.factor_c,
                                       server_round,
                                       clients)
            return_clients = []
            for client,time in chosen_clients_times:
                fit_ins.config["duration"] = str(time)
                return_clients.append((client, deepcopy(fit_ins)))
            return return_clients
        
        elif config["alg"] == "fedProx3Sat":
            chosen_clients_times, self.counter = fedProx3Sat(self.satellite_access_csv, 
                                        self.counter,
                                        int(config["clients"]),
                                        int(config["client_limit"]),
                                        int(config["n_sat_in_cluster"]),
                                        int(config["epochs"]),
                                        int(config["n_cluster"]),
                                        self.factor_s,
                                        self.factor_c,
                                        server_round,
                                        clients)
            return_clients = []
            for client,time in chosen_clients_times:
                fit_ins.config["duration"] = str(time)
                return_clients.append((client, deepcopy(fit_ins)))
            return return_clients

        elif config["alg"] == "fedBuffSat":
            self.satellite_client_list = []
            chosen_clients_times, self.counter = fedBuffSat(self.satellite_access_csv, 
                                       self.counter,
                                       int(config["clients"]),
                                       int(config["client_limit"]),
                                       int(config["n_sat_in_cluster"]),
                                       self.factor_s,
                                       self.factor_c,
                                       server_round,
                                       clients,
                                       config["name"],
                                       config["alg"])
            return_clients = []
            for client,time in chosen_clients_times:
                fit_ins.config["duration"] = str(time)
                self.satellite_client_list.append(int(client.cid))
                return_clients.append((client, deepcopy(fit_ins)))
            return return_clients

        elif config["alg"] == "fedBuff2Sat":
            self.satellite_client_list = []
            chosen_clients_times, self.counter = fedBuff2Sat(self.satellite_access_csv, 
                                       self.counter,
                                       int(config["clients"]),
                                       int(config["client_limit"]),
                                       int(config["n_sat_in_cluster"]),
                                       self.factor_s,
                                       self.factor_c,
                                       server_round,
                                       clients,
                                       config["name"],
                                       config["alg"])
            return_clients = []
            for client,time in chosen_clients_times:
                fit_ins.config["duration"] = str(time)
                self.satellite_client_list.append(int(client.cid))
                return_clients.append((client, deepcopy(fit_ins)))
            return return_clients
            
        elif config["alg"] == "fedBuff3Sat":
            self.satellite_client_list = []
            chosen_clients_times, self.counter = fedBuff3Sat(self.satellite_access_csv, 
                                        self.counter,
                                        int(config["clients"]),
                                        int(config["client_limit"]),
                                        int(config["n_sat_in_cluster"]),
                                        int(config["epochs"]),
                                        int(config["n_cluster"]),
                                        self.factor_s,
                                        self.factor_c,
                                        server_round,
                                        clients,
                                        config["name"],
                                        config["alg"])
            return_clients = []
            for client,time in chosen_clients_times:
                fit_ins.config["duration"] = str(time)
                self.satellite_client_list.append(int(client.cid))
                return_clients.append((client, deepcopy(fit_ins)))
            return return_clients
        
        elif config["alg"] == "AutoFLSat":
            # self.sim_times_start = []
            # self.sim_times_currents = []
            # self.cluster_round_starts = []
            # self.cluster_round_currents = []
            chosen_clients, agg_clients, self.counter, self.sim_times_start, self.sim_times_currents, self.cluster_round_starts, self.cluster_round_currents, self.model_type = AutoFLSat(self.satellite_access_csv, 
                                                        self.counter, 
                                                        int(config["clients"]), 
                                                        int(config["client_limit"]), 
                                                        int(config["n_sat_in_cluster"]), 
                                                        int(config["n_cluster"]),
                                                        self.factor_s, 
                                                        self.factor_c, 
                                                        server_round,
                                                        clients,
                                                        config["name"],
                                                        config["alg"],
                                                        self.sim_times_start,
                                                        self.sim_times_currents,
                                                        self.cluster_round_starts,
                                                        self.cluster_round_currents,
                                                        int(config["epochs"]))
            return_clients = []
            self.satellite_client_list = []

            if self.model_type == "local_cluster":
                for client,cluster,agg_cluster in chosen_clients:
                    self.cluster_num = cluster
                    fit_ins.config["model_type"] = str(self.model_type)
                    fit_ins.config["cluster_identifier"] = str(cluster)
                    fit_ins.config["agg_cluster"] = str(agg_cluster)
                    self.satellite_client_list.append(int(client.cid))
                    return_clients.append((client, deepcopy(fit_ins)))
                return return_clients
            elif self.model_type == "global_cluster":
                for client,cluster,agg_cluster in chosen_clients:
                    self.cluster_num = cluster
                    self.satellite_client_list.append(int(client.cid))
                
                for client,cluster,agg_cluster in agg_clients:
                    fit_ins.config["model_type"] = str(self.model_type)
                    fit_ins.config["cluster_identifier"] = str(cluster)
                    fit_ins.config["agg_cluster"] = str(agg_cluster)
                    return_clients.append((client, deepcopy(fit_ins)))
                return return_clients
                
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
        elif config["alg"] == "fedAvg3Sat":
            chosen_clients, self.counter = fedAvg3Sat(self.satellite_access_csv, 
                                        self.counter,
                                        int(config["clients"]),
                                        int(config["client_limit"]),
                                        int(config["n_sat_in_cluster"]),
                                        int(config["epochs"]),
                                        int(config["n_cluster"]),
                                        self.factor_s,
                                        self.factor_c,
                                        server_round,
                                        clients)
        elif config["alg"] == "fedProxSat":
            chosen_clients_times, self.counter = fedProxSat(self.satellite_access_csv, 
                                       self.counter,
                                       int(config["clients"]),
                                       int(config["client_limit"]),
                                       int(config["n_sat_in_cluster"]),
                                       self.factor_s,
                                       self.factor_c,
                                       server_round,
                                       clients)
            return_clients = []
            for client,time in chosen_clients_times:
                evaluate_ins.config["duration"] = str(time)
                return_clients.append((client, deepcopy(evaluate_ins)))
            return return_clients
        elif config["alg"] == "fedProx2Sat":
            chosen_clients_times, self.counter = fedProx2Sat(self.satellite_access_csv, 
                                       self.counter,
                                       int(config["clients"]),
                                       int(config["client_limit"]),
                                       int(config["n_sat_in_cluster"]),
                                       self.factor_s,
                                       self.factor_c,
                                       server_round,
                                       clients)
            return_clients = []
            for client,time in chosen_clients_times:
                evaluate_ins.config["duration"] = str(time)
                return_clients.append((client, deepcopy(evaluate_ins)))
            return return_clients
        elif config["alg"] == "fedProx3Sat":
            chosen_clients_times, self.counter = fedProx3Sat(self.satellite_access_csv, 
                                        self.counter,
                                        int(config["clients"]),
                                        int(config["client_limit"]),
                                        int(config["n_sat_in_cluster"]),
                                        int(config["epochs"]),
                                        int(config["n_cluster"]),
                                        self.factor_s,
                                        self.factor_c,
                                        server_round,
                                        clients)
            return_clients = []
            for client,time in chosen_clients_times:
                evaluate_ins.config["duration"] = str(time)
                return_clients.append((client, deepcopy(evaluate_ins)))
            return return_clients
        elif config["alg"] == "fedBuffSat":
            chosen_clients_times, self.counter = fedBuffSat(self.satellite_access_csv, 
                                       self.counter,
                                       int(config["clients"]),
                                       int(config["client_limit"]),
                                       int(config["n_sat_in_cluster"]),
                                       self.factor_s,
                                       self.factor_c,
                                       server_round,
                                       clients,
                                       config["name"],
                                       config["alg"])
            return_clients = []
            sid_count = 0
            for client,time in chosen_clients_times:
                evaluate_ins.config["model_update"] = str(self.satellite_client_list[sid_count])
                return_clients.append((client, deepcopy(evaluate_ins)))
                sid_count += 1
            return return_clients
        elif config["alg"] == "fedBuff2Sat":
            chosen_clients_times, self.counter = fedBuff2Sat(self.satellite_access_csv, 
                                       self.counter,
                                       int(config["clients"]),
                                       int(config["client_limit"]),
                                       int(config["n_sat_in_cluster"]),
                                       self.factor_s,
                                       self.factor_c,
                                       server_round,
                                       clients,
                                       config["name"],
                                       config["alg"])
            return_clients = []
            sid_count = 0
            for client,time in chosen_clients_times:
                evaluate_ins.config["model_update"] = str(self.satellite_client_list[sid_count])
                return_clients.append((client, deepcopy(evaluate_ins)))
                sid_count += 1
            return return_clients
        elif config["alg"] == "fedBuff3Sat":
            chosen_clients_times, self.counter = fedBuff3Sat(self.satellite_access_csv, 
                                        self.counter,
                                        int(config["clients"]),
                                        int(config["client_limit"]),
                                        int(config["n_sat_in_cluster"]),
                                        int(config["epochs"]),
                                        int(config["n_cluster"]),
                                        self.factor_s,
                                        self.factor_c,
                                        server_round,
                                        clients,
                                        config["name"],
                                        config["alg"])
            return_clients = []
            sid_count = 0
            for client,time in chosen_clients_times:
                evaluate_ins.config["model_update"] = str(self.satellite_client_list[sid_count])
                return_clients.append((client, deepcopy(evaluate_ins)))
                sid_count += 1
            return return_clients
        elif config["alg"] == "AutoFLSat":
            evaluate_ins.config["model_update"] = self.model_type
            evaluate_ins.config["cluster_identifier"] = str(self.cluster_num)
            evaluate_ins.config["agg_cluster"] = str(self.cluster_num)
            chosen_clients = [client for client in clients if int(client.cid) in self.satellite_client_list]
        
            
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
        wandb.log({"acc": aggregated_accuracy, 
                   "loss": aggregated_loss, 
                   "server_round": server_round,
                   "cluster_round": self.cluster_round_currents[self.cluster_num -1],
                   "cluster_num": self.cluster_num})
        
        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}
