import numpy as np
import wandb
import os
import pandas as pd
import shutil

###################    AutoFLSat   ########################

def AutoFLSat2(sat_df, 
              counter, 
              client_n, 
              client_max, 
              sat_n, 
              cluster_n,
              factor_s, 
              factor_c, 
              server_round,
              clients,
              name,
              alg,
              epochs_list, # not needed
              sim_times_currents,
              cluster_round_starts,
              cluster_round_currents,
              epochs,
              start_time_og,
              agg_true):


    # Track starting time for reaching out to satellites
    start_time_sec = sat_df['Start Time Seconds Cumulative'].iloc[counter]
    if start_time_og == 0:
        start_time_og = start_time_sec
    
    #end_time_sec = sat_df['End Time Seconds Cumulative'].iloc[counter]
        
    cluster_clients = []


    # Hard code the initial value that the sims start with
    if all(sim_time == 0 for sim_time in sim_times_currents):
        for i in range(len(sim_times_currents)):
            sim_times_currents[i] = 1711987200

    
    """
    START scheduling if this is good:
    """
    if all(i >= start_time_og for i in sim_times_currents): # only do scheduling if all clients are done training
        
        if agg_true:
        # if time to aggregate 
                    
            # cluster clients to aggregate
            y = [[clients[i],i,i] for i in range(1,cluster_n+1)]

            # delete after aggregate
            wandb.log({"cluster_round_current_time": sim_times_currents[0],
                "cluster_id": 0,
                "cluster_rounds": cluster_round_currents[0],
                "agg_type": "globalAgg",
                "server_round": server_round})
        
            # get all clusters
            for client in [i for i in range(sat_n*cluster_n)]:
                cluster_clients.append(client)
            
            x = [[client,1,1] for client in clients if int(client.cid) in cluster_clients]
            agg_true = False
            return x, y, counter, epochs_list, sim_times_currents, cluster_round_starts, cluster_round_currents, "global_cluster", epochs, start_time_og, agg_true
        
        else:
        # then schedule clients

            start_time, end_time, counter, epoch_train, idle_time, duration_round = scheduleISL(sat_df, 
                                                                    counter,
                                                                    cluster_n,
                                                                    factor_c,
                                                                    start_time_og,
                                                                    10*60*5)
                # return nubmer of epochs, as well as counter number to start from
                # next time, and 
        
            duration_full = start_time - start_time_og + duration_round
            print(start_time)
            wandb.log({"idle_time_avg": idle_time,
                "epoch_train": epoch_train,
                "duration_full": duration_full,
                "duration_agg": duration_round,
                "start_time": cluster_round_currents[0],
                "agg_type": "globalAgg",
                "server_round": server_round,
                "start_time": start_time_og,
                "end_time": end_time})
            
            start_time_og = start_time
            epochs = epoch_train

    """
    Normal fL stuff:
    """

    # if anything is less than the current start_time set
    if any(i < start_time_og for i in sim_times_currents):

        ""

        cluster_index = np.argmin(cluster_round_currents)
        cluster = cluster_index+1
        cluster_round_currents[cluster_index] += 1

        sim_times_currents[cluster_index] += epochs

        if all(i >= start_time_og for i in sim_times_currents):
            agg_true = True

        wandb.log({"cluster_round_current_time": sim_times_currents[(cluster_index)],
                "cluster_id": cluster,
                "cluster_rounds": cluster_round_currents[(cluster_index)],
                "agg_type": "localAgg",
                "server_round": server_round})
        
        # get all clusters
        for client in [i for i in range(sat_n)]:
            client_id = int(sat_n*(cluster)-
                        (sat_n-(client)))
            cluster_clients.append(client_id)
        # print(cluster_clients)

    
        x = [[client,cluster,cluster] for client in clients if int(client.cid) in cluster_clients]

        return x, x, counter, epochs_list, sim_times_currents, cluster_round_starts, cluster_round_currents, "local_cluster", epochs, start_time_og, agg_true
        
        
                
        """
        # check if any thing is comms ing right now
        cluster_id_1 = int(((sat_df['cluster_num_1'].iloc[counter]))/factor_c)

        cluster_id_2 = int(((sat_df['cluster_num_2'].iloc[counter]))/factor_c)

        duration = ((sat_df['Duration (sec)'].iloc[int(counter)]))
        send_time = 100
        counter_start = counter
        sent = np.zeros([cluster_n,cluster_n])
        while (sim_times_currents[cluster_id_1-1] >= start_time_sec and sim_times_currents[cluster_id_2-1] >= start_time_sec and 
            duration > send_time):#and sim_times_currents[cluster_id_1-1] < end_time_sec-send_time and sim_times_currents[cluster_id_2-1]  < end_time_sec-send_time) :

            print(sim_times_currents)
            print(start_time_sec)
            print(duration)
            print(sent)

            file_name = f'/datasets/{alg}/model_files_{name}/{cluster}_{agg_cluster}.pth'
            
            if sent[cluster_id_1-1,cluster_id_2-1] == 0:
                if os.path.exists(f'/datasets/{alg}/model_files_{name}/{cluster_id_1}_{cluster_id_1}.pth'):
                    shutil.copy(f'/datasets/{alg}/model_files_{name}/{cluster_id_1}_{cluster_id_1}.pth',
                                f'/datasets/{alg}/model_files_{name}/{cluster_id_1}_{cluster_id_2}.pth')
                    sent[cluster_id_1-1,cluster_id_2-1] = 1
                print("sent from ", str(cluster_id_1), " to ", str(cluster_id_2))
                print("Pass complete")

            if sent[cluster_id_2-1,cluster_id_1-1] == 0 :
                if os.path.exists(f'/datasets/{alg}/model_files_{name}/{cluster_id_2}_{cluster_id_2}.pth'):
                    shutil.copy(f'/datasets/{alg}/model_files_{name}/{cluster_id_2}_{cluster_id_2}.pth',
                                f'/datasets/{alg}/model_files_{name}/{cluster_id_2}_{cluster_id_1}.pth')
                    sent[cluster_id_2-1,cluster_id_1-1] = 1
                print("sent from ", str(cluster_id_2), " to ", str(cluster_id_1))
                print("Pass complete")
            counter += 1
            wandb.log({"comms_cluster_1": cluster_id_1,
                    "comms_cluster_2": cluster_id_2,
                    "cluster_1_round": cluster_round_currents[(cluster_id_1-1)],
                    "cluster_2_round": cluster_round_currents[(cluster_id_2-1)],
                    "saved_into_1": sent[cluster_id_1-1,cluster_id_2-1],
                    "saved_into_2": sent[cluster_id_2-1,cluster_id_1-1],
                        "server_round": server_round})
                # check if any thing is comms ing right now
            cluster_id_1 = int(((sat_df['cluster_num_1'].iloc[counter]))/factor_c)

            cluster_id_2 = int(((sat_df['cluster_num_2'].iloc[counter]))/factor_c)

            duration = ((sat_df['Duration (sec)'].iloc[int(counter)]))

            # Track starting time for reaching out to satellites
            start_time_sec = sat_df['Start Time Seconds Cumulative'].iloc[counter]

        if counter_start == counter:
            counter +=1
        """

    
def scheduleISL(sat_df, 
                counter,
                cluster_n,
                factor_c,
                start_time_og,
                epochs):
    
    start_times = np.zeros([cluster_n+1,cluster_n+1])
    end_times = np.zeros([cluster_n+1,cluster_n+1])

    print(start_times)
    print(end_times)

    count_temp = counter
    count_middle = counter
    count_accesses = 0
    connections = (cluster_n-1) * (((cluster_n-1)) + 1) // 2

    # Looping through the CSV    
    while count_accesses < connections or (count_temp-count_middle<20): #Need to figure out when to stop the optimization
        
        # need cluster ids
        cluster_id_1 = int(((sat_df['cluster_num_1'].iloc[count_temp]))/factor_c)
        cluster_id_2 = int(((sat_df['cluster_num_2'].iloc[count_temp]))/factor_c)
        # get all times info
        temp_start = sat_df['Start Time Seconds Cumulative'].iloc[count_temp]
        temp_end = sat_df['End Time Seconds Cumulative'].iloc[count_temp]
        duration = sat_df['Duration (sec)'].iloc[count_temp]

        # make sure right order for array
        if cluster_id_1 > cluster_id_2:
            temp = cluster_id_1
            cluster_id_1 = cluster_id_2
            cluster_id_2 = temp

        # check if access window large enough
        if duration > 200:
            # Then squeeze down
            if start_times[cluster_id_1,cluster_id_2] == 0 and ((start_time_og + epochs) < temp_start) :

                print(start_times[cluster_id_1,cluster_id_2])
                print(end_times[cluster_id_1,cluster_id_2])
                start_times[cluster_id_1,cluster_id_2] = temp_start
                if end_times[cluster_id_1,cluster_id_2] == 0:
                    end_times[cluster_id_1,cluster_id_2] = temp_end
                    count_accesses += 1
                    count_middle = count_temp


            if count_accesses >= connections:
                if start_times[cluster_id_1,cluster_id_2] < temp_start and temp_end < np.max(end_times):
                    start_times[cluster_id_1,cluster_id_2] = temp_start
                    end_times[cluster_id_1,cluster_id_2] = temp_end
                    count_middle = count_temp
                if end_times[cluster_id_1,cluster_id_2] > temp_end and temp_end > np.min(start_times):
                    start_times[cluster_id_1,cluster_id_2] = temp_start
                    end_times[cluster_id_1,cluster_id_2] = temp_end
                    count_middle = count_temp

        count_temp += 1

    print("WE MADE IT")
    print(count_temp)
    print(count_accesses)
    print(count_middle)
    print(start_times)
    print(end_times)

    new_start_time = np.min(start_times[np.nonzero(start_times)])
    new_end_time = np.max(end_times[np.nonzero(end_times)])
    idle_time = 0

    for cluster_id_1 in range(1,cluster_n+1):
        for cluster_id_2 in range(cluster_id_1+1,cluster_n+1):
            idle_time += start_times[cluster_id_1,cluster_id_2]-new_start_time
            idle_time += end_times[cluster_id_1,cluster_id_2]-new_end_time


    counter = count_middle  
    epoch_train = (new_start_time-start_time_og)#/60/5
    duration_round = new_end_time -  new_start_time
    return new_start_time, new_end_time, counter, epoch_train, idle_time/cluster_n,duration_round