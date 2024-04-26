import numpy as np
import wandb
###################    FedAvgSat   ########################

def fedAvgSat(sat_df, 
              counter, 
              client_n, 
              client_max, 
              sat_n, 
              factor_s, 
              factor_c, 
              server_round,
              clients):

    # Track starting time for reaching out to satellites
    start_time_sec = sat_df['Start Time Seconds Cumulative'].iloc[counter]

    # lists for tracking satellites
    client_list = np.zeros(client_n)
    client_time_list = np.zeros(client_n)
    client_twice = []
    done_count = 0
    idle_time_total = 0

    # Check if the number of satellites exceeds maximum 
    # we want to train with each round
    if client_n < client_max:
        limit = client_n
    else:
        limit = client_max

    # Looping through the CSV
    while done_count < limit:

        # need cluster and satellite id to calculate client_id
        cluster_id = ((sat_df['cluster_num'].iloc[counter]))
        satellite_id = ((sat_df['sat_num'].iloc[counter]))

        # calculate client_id from which cluster and which satellite
        client_id = int(sat_n*(cluster_id/factor_c)-
                        (sat_n-(satellite_id/factor_s))-1)

        # Track the first 10 satellites that make contact with a groundstation 
        if client_list[client_id] == 0 and len(client_twice) < limit:
            client_twice.append(client_id)
            client_time_list[client_id] = sat_df['Start Time Seconds Cumulative'].iloc[counter]
        
        # Track every single pass of every client satellite
        client_list[client_id] += 1

        # Track when the first 10 satellites that made contact reach back to a groundstation to give their information
        if client_list[client_id] == 2 and (client_id in client_twice):
            done_count +=1
            client_time_list[client_id] = sat_df['Start Time Seconds Cumulative'].iloc[counter] - client_time_list[client_id]
        
        # Going through the csv rows
        counter += 1

    # TODO: Delete whenever i realize i know how to track this in wandb
    print(client_list)
    print(client_twice)
    print(client_time_list)

    # Track when we finish reach out to all satellites
    stop_time_sec = sat_df['Start Time Seconds Cumulative'].iloc[counter]
    
    # Caculate idle time totals and averages
    for time in client_time_list:
        idle_time_total += (stop_time_sec - start_time_sec)-time 
    idle_time_avg = idle_time_total/client_n
    wandb.log({"start_time_sec": start_time_sec, 
               "stop_time_sec": stop_time_sec, 
               "server_round": server_round,
               "duration" : stop_time_sec - start_time_sec, 
               "idle_time_total": idle_time_total,
               "idle_time_avg": idle_time_avg})

    x = [client for client in clients if int(client.cid) in client_twice]

    return x, counter
