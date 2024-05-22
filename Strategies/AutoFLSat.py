import numpy as np
import wandb
import os
import pandas as pd

###################    AutoFLSat   ########################

def AutoFLSat(sat_df, 
              counter, 
              client_n, 
              client_max, 
              sat_n, 
              cluster_n
              factor_s, 
              factor_c, 
              server_round,
              clients,
              name,
              alg,
              sim_times_start,
              sim_times_currents,
              cluster_round_starts,
              cluster_round_currents
              epochs):


    # Track starting time for reaching out to satellites
    start_time_sec = sat_df['Start Time Seconds Cumulative'].iloc[counter]
        
    cluster_clients = []

    # Hard code the initial value that the sims start with
    if all(sim_time == 0 for sim_time in sim_times_currents):
        for i in range(len(sim_times_currents)):
            sim_times_currents[i] = 1711987200

    if False: # all(sim_time >= start_time_sec for i in sim_times_currents) :

        # need cluster and satellite id to calculate client_id
        cluster_id_1 = ((sat_df['cluster_num_1'].iloc[counter]))
        satellite_id_1 = ((sat_df['sat_num_1'].iloc[counter]))

        cluster_id_2 = ((sat_df['cluster_num_2'].iloc[counter]))
        satellite_id_2 = ((sat_df['sat_num_2'].iloc[counter]))
        
        # calculate client_id from which cluster and which satellite
        client_id_1 = int(sat_n*(cluster_id_1/factor_c)-
                        (sat_n-(satellite_id_1/factor_s))-1)
        
        client_id_2 = int(sat_n*(cluster_id_2/factor_c)-
                        (sat_n-(satellite_id_2/factor_s))-1)

        # get all clusters
        for client in [int(factor_s*(i+1)) for i in range(sat_n)]:
            client_id = int(sat_n*(cluster/factor_c)-
                        (sat_n-(client/factor_s))-1)
            cluster_clients.append(client_id)

        wandb.log({"cluster_round_current_time": cluster_times_df[str(cluster)],
                "cluster_id": cluster,
                "cluster_rounds": cluster_rounds_df[str(cluster)],
                "agg_type": "localAgg",
                "server_round": server_round})

        x = [[clients[0],cluster_id_1],[clients[1],cluster_id_2]]
        counter += 1

        return x, counter, sim_times_start, sim_times_currents, cluster_round_starts, cluster_round_currents, "other_cluster"
        

    else:
        cluster = np.argmin(cluster_times_df)

        cluster_round_currents[str(cluster)] += 1

        sim_times_currents[str(cluster)] += epochs*60*5

        wandb.log({"cluster_round_current_time": cluster_times_df[str(cluster)],
                "cluster_id": cluster,
                "cluster_rounds": cluster_rounds_df[str(cluster)],
                "agg_type": "localAgg",
                "server_round": server_round})
        
        # get all clusters
        for client in [int(factor_s*(i+1)) for i in range(sat_n)]:
            client_id = int(sat_n*(cluster/factor_c)-
                        (sat_n-(client/factor_s))-1)
            cluster_clients.append(client_id)

        
        x = [[client,cluster] for client in clients if int(client.cid) in cluster_clients]

        return x, counter, sim_times_start, sim_times_currents, cluster_round_starts, cluster_round_currents, "local_cluster"
        


        # for each round track which server round is associated with which
        # need to track duration of each round? >> track how many rounds were needed to receive
        ,
                # "duration" : stop_time_sec - start_time_sec, 
                # "idle_time_total": idle_time_total,
                # "idle_time_avg": idle_time_avg

        # this is the only thing that differs for fedprox, now we send durations too
        # most changes are in client
        

    # sim_time

    # count_duration = 0


    # client_twice = []
    # done_count = 0
    # idle_time_total = 0

    #         client_time_list[client_id] = sat_df['End Time Seconds Cumulative'].iloc[int(counter)] - end_times[str(client_id)][0]
    #         count_duration += sat_df['Duration (sec)'].iloc[int(counter)] 
    #         end_times[str(client_id)] = sat_df['End Time Seconds Cumulative'].iloc[int(counter)]
        
    #     # Going through the csv rows
    #     counter += 1

    # # TODO: Delete whenever i realize i know how to track this in wandb
    # print(client_list)
    # print(client_twice)
    # print(client_time_list)
    # end_times.to_csv(file_name)
    

    # # Track when we finish reach out to all satellites
    # stop_time_sec = sat_df['Start Time Seconds Cumulative'].iloc[counter]
    
    # # Caculate idle time totals and averages

    # # fed_buff no need now:
    # # for time in client_time_list:
    # idle_time_total = count_duration # += (stop_time_sec - start_time_sec)*limit - count_duration
    # idle_time_avg = idle_time_total/client_n


    # wandb.log({"start_time_sec": start_time_sec, 
    #            "stop_time_sec": stop_time_sec, 
    #            "server_round": server_round,
    #            "duration" : stop_time_sec - start_time_sec, 
    #            "idle_time_total": idle_time_total,
    #            "idle_time_avg": idle_time_avg})

    # # this is the only thing that differs for fedprox, now we send durations too
    # # most changes are in client
    # x = [[client,client_time_list[int(client.cid)]] for client in clients if int(client.cid) in client_twice]

    # return x, counter


"""

    # lists for tracking satellites
    client_list = np.zeros(client_n)
    client_time_list = np.zeros(client_n)

    # check if there is a local model saved to the disk, if so use that
    file_name = f'/datasets/{alg}/rounds_{name}.csv'
    file_name_times = f'/datasets/{alg}/times_{name}.csv'

    if os.path.exists(file_name):
        end_times = pd.read_csv(file_name)
        print("READING")
        print(file_name)
    else:
        cluster_rounds = {}
        for i in range(cluster_n):
            cluster_rounds[str(i)] = 0
        cluster_rounds_df = pd.DataFrame.from_dict([cluster_rounds])
        cluster_rounds_df.to_csv(file_name)
        print("SAVING")
        print(file_name)

    if os.path.exists(file_name_times):
        times = pd.read_csv(file_name_times)
        print("READING")
        print(file_name_times)
    else:
        cluster_times = {}
        for i in range(cluster_n):
            cluster_times[str(i)] = start_time_sec
        cluster_times_df = pd.DataFrame.from_dict([cluster_times])
        cluster_times_df.to_csv(file_name_times)
        print("SAVING")
        print(file_name_times)
        
        
            # Track the first 10 satellites that make contact with a groundstation 
            if client_list[client_id] == 0 and len(client_twice) < limit:
                client_twice.append(client_id)
                # fed_buff no need now:
                # client_time_list[client_id] = sat_df['Start Time Seconds Cumulative'].iloc[counter]
            
            # Track every single pass of every client satellite
            client_list[client_id] += 1

            # Track when the first 10 satellites that made contact reach back to a groundstation to give their information
            if client_list[client_id] == 2 and (client_id in client_twice):
                done_count +=1

                # fed_buff no need now:
                # print(client_id)
                # print(end_times)
                client_time_list[client_id] = sat_df['End Time Seconds Cumulative'].iloc[int(counter)] - end_times[str(client_id)][0]
                count_duration += sat_df['Duration (sec)'].iloc[int(counter)] 
                end_times[str(client_id)] = sat_df['End Time Seconds Cumulative'].iloc[int(counter)]
        
        counter += 1
        sim_time = sat_df['Start Time Seconds Cumulative'].iloc[counter]

    # fed_buff no need now:
    # for time in client_time_list:
    idle_time_total = count_duration # += (stop_time_sec - start_time_sec)*limit - count_duration
    idle_time_avg = idle_time_total/client_n
        
        """