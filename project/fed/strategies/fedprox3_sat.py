import numpy as np
import wandb
###################    FedProx3Sat   ########################

def fedProx3Sat(sat_df, 
              counter, 
              client_n, 
              client_max, 
              sat_n, 
              epoch_min,
              cluster_n, 
              factor_s, 
              factor_c, 
              server_round,
              clients):

    # Track starting time for reaching out to satellites
    start_time_sec = sat_df['Start Time Seconds Cumulative'].iloc[counter]

    # lists for tracking satellites
    client_list = np.zeros(client_n)
    cluster_list = np.zeros((cluster_n,sat_n))
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

        cluster_id = int(cluster_id/factor_c)-1
        satellite_id = int(satellite_id/factor_s)-1

        # If not chosen to train already, track every single pass of satellites, and which cluster they are in
        if client_list[client_id] != -1:
            client_list[client_id] += 1
            cluster_list[cluster_id,satellite_id] += 1

        # If first time joining training line, track when they start
        if client_time_list[client_id] == 0:
            client_time_list[client_id] = sat_df['Start Time Seconds Cumulative'].iloc[counter]

        # If the satellite sees groundstation twice and we don't already meet limit of clients
        if client_list[client_id] == 2 and len(client_twice) < limit:
            client_twice.append(client_id)
            client_list[client_id] = -1
            cluster_list[cluster_id,satellite_id] = 0
            client_time_list[client_id] = sat_df['End Time Seconds Cumulative'].iloc[counter] - client_time_list[client_id]
            done_count +=1
            
        # If the current cluster seems to have other agents that are currently training and could give results back
        if sum(cluster_list[cluster_id,:]) > 2:
            for sat_count,sat_id in zip(cluster_list[cluster_id,:],range(sat_n)):
                
                # if a different satellite in the cluster has already start training
                if sat_count == 1:
                    # get id of sat
                    client_id = int(sat_n*(cluster_id+1)- (sat_n-(sat_id+1))-1)

                    # if enough time has passed for training and another satellite on the same cluster can communicate with intra sat link
                    if (sat_df['Start Time Seconds Cumulative'].iloc[counter] - client_time_list[client_id]) > (epoch_min *60*5):
                        client_twice.append(client_id)
                        client_list[client_id] = -1
                        cluster_list[cluster_id,sat_id] = 0
                        client_time_list[client_id] = sat_df['End Time Seconds Cumulative'].iloc[counter] - client_time_list[client_id]
                        done_count +=1
                        
        # Going through the csv rows
        counter += 1

    for id in range(client_n):
        # Only track satellties as having been not idle if they trained this round
        if id not in client_twice:
            client_time_list[id] = 0
    

    # TODO: Delete whenever i realize i know how to track this in wandb
    print(client_list)
    print(client_twice)
    print(client_time_list)

    # Track when we finish reach out to all satellites
    stop_time_sec = sat_df['Start Time Seconds Cumulative'].iloc[counter]
    
    # Caculate idle time totals and averages
    for time in client_time_list:
        idle_time_total += (stop_time_sec - start_time_sec)-(10* 60 * 5)   
    idle_time_avg = idle_time_total/client_n
    wandb.log({"start_time_sec": start_time_sec, 
               "stop_time_sec": stop_time_sec, 
               "server_round": server_round,
               "duration" : stop_time_sec - start_time_sec, 
               "idle_time_total": idle_time_total,
               "idle_time_avg": idle_time_avg})

    # this is the only thing that differs for fedprox, now we send durations too
    # most changes are in client
    x = [(client,client_time_list[int(client.partition_id)]) for client in clients if int(client.partition_id) in client_twice]

    return x, counter

def fedProx32Sat(sat_df, 
              counter, 
              client_n, 
              client_max, 
              sat_n, 
              epoch_min,
              cluster_n, 
              factor_s, 
              factor_c, 
              server_round,
              clients):

    # Track starting time for reaching out to satellites
    start_time_sec = sat_df['Start Time Seconds Cumulative'].iloc[counter]

    # lists for tracking satellites
    client_list = np.zeros(client_n)
    cluster_list = np.zeros((cluster_n,sat_n))
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
    print("UYES")
    # Looping through the CSV
    while done_count < limit:

        # need cluster and satellite id to calculate client_id
        cluster_id = ((sat_df['cluster_num'].iloc[counter]))
        satellite_id = ((sat_df['sat_num'].iloc[counter]))

        # calculate client_id from which cluster and which satellite
        client_id = int(sat_n*(cluster_id/factor_c)-
                        (sat_n-(satellite_id/factor_s))-1)

        cluster_id = int(cluster_id/factor_c)-1
        satellite_id = int(satellite_id/factor_s)-1

        # If not chosen to train already, track every single pass of satellites, and which cluster they are in
        if client_list[client_id] != -1:
            client_list[client_id] += 1
            cluster_list[cluster_id,satellite_id] += 1

        # If first time joining training line, track when they start
        if client_time_list[client_id] == 0:
            client_time_list[client_id] = sat_df['Start Time Seconds Cumulative'].iloc[counter]

        # If the satellite sees groundstation twice and we don't already meet limit of clients
        if client_list[client_id] >= 2 and len(client_twice) < limit and (sat_df['Start Time Seconds Cumulative'].iloc[counter] - client_time_list[client_id]) > (epoch_min *60*5):
            client_twice.append(client_id)
            client_list[client_id] = -1
            cluster_list[cluster_id,satellite_id] = 0
            client_time_list[client_id] = sat_df['End Time Seconds Cumulative'].iloc[counter] - client_time_list[client_id]
            done_count +=1
            
        # If the current cluster seems to have other agents that are currently training and could give results back
        if sum(cluster_list[cluster_id,:]) > 2:
            for sat_count,sat_id in zip(cluster_list[cluster_id,:],range(sat_n)):
                
                # if a different satellite in the cluster has already start training
                if sat_count == 1:
                    # get id of sat
                    client_id = int(sat_n*(cluster_id+1)- (sat_n-(sat_id+1))-1)

                    # if enough time has passed for training and another satellite on the same cluster can communicate with intra sat link
                    if (sat_df['Start Time Seconds Cumulative'].iloc[counter] - client_time_list[client_id]) > (epoch_min *60*5):
                        client_twice.append(client_id)
                        client_list[client_id] = -1
                        cluster_list[cluster_id,sat_id] = 0
                        client_time_list[client_id] = sat_df['End Time Seconds Cumulative'].iloc[counter] - client_time_list[client_id]
                        done_count +=1
                        
        # Going through the csv rows
        counter += 1

    for id in range(client_n):
        # Only track satellties as having been not idle if they trained this round
        if id not in client_twice:
            client_time_list[id] = 0
    

    # TODO: Delete whenever i realize i know how to track this in wandb
    print(client_list)
    print(client_twice)
    print(client_time_list)

    # Track when we finish reach out to all satellites
    stop_time_sec = sat_df['Start Time Seconds Cumulative'].iloc[counter]
    
    # Caculate idle time totals and averages
    for time in client_time_list:
        idle_time_total += (stop_time_sec - start_time_sec)-(10* 60 * 5)   
    idle_time_avg = idle_time_total/client_n
    wandb.log({"start_time_sec": start_time_sec, 
               "stop_time_sec": stop_time_sec, 
               "server_round": server_round,
               "duration" : stop_time_sec - start_time_sec, 
               "idle_time_total": idle_time_total,
               "idle_time_avg": idle_time_avg})

    # this is the only thing that differs for fedprox, now we send durations too
    # most changes are in client
    x = [(client,client_time_list[int(client.partition_id)]) for client in clients if int(client.partition_id) in client_twice]

    return x, counter