import pandas as pd
from datetime import datetime
from datetime import timedelta
import math
import numpy as np

import wandb


############################################################
###################    ALGORITHMS   ########################
############################################################

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


############################################################
###################    CSV THINGS   ########################
############################################################

def time_to_int(dateobj):

    total = int(dateobj.strftime('%f'))*.000001
    total += int(dateobj.strftime('%S'))
    total += int(dateobj.strftime('%M')) * 60
    total += int(dateobj.strftime('%H')) * 60 * 60
    total += (int(dateobj.strftime('%j')) - 1) * 60 * 60 * 24
    total += (int(dateobj.strftime('%Y')) - 1970) * 60 * 60 * 24 * 365
    return total

def read_sat_csv(filename):
    sat_df = pd.read_csv(filename)
    n_s = int(filename[19:].split("_")[0][:-1])
    n_c = int(filename[19:].split("_")[1][:-1])
    new_keys = ['Start Time Seconds Cumulative','End Time Seconds Cumulative','Start Time Seconds datetime','End Time Seconds datetime']
    old_keys = ['Start Time (UTCG)','Stop Time (UTCG)']
    for i in range(2):
        sat_df[new_keys[i]] = [time_to_int(datetime.strptime(x, "%d %b %Y %H:%M:%S.%f")) for x in sat_df[old_keys[i]]]
        sat_df[new_keys[i+2]] = [(datetime.strptime(x, "%d %b %Y %H:%M:%S.%f")) for x in sat_df[old_keys[i]]]
    sat_df['ground_station'] = [x[9:] for x in sat_df['To Object']]
    sat_df['cluster_num'] = [int(x[24:25+int(math.log10(n_c))]) for x in sat_df['From Object']]
    sat_df['sat_num'] = [int(x[-1-int(math.log10(n_s)):]) for x in sat_df['From Object']]
    sat_df_sorted = sat_df.sort_values('Start Time Seconds Cumulative').reset_index()
    print(sat_df_sorted)
    return sat_df_sorted

def choose_sat_csv(df, og_s, og_c, new_s, new_c, gs):
    if og_s%new_s == 0 and og_c%new_c == 0:
        client_list = [int(og_s/new_s*(i+1)) for i in range(new_s)]
        cluster_list = [int(og_c/new_c*(i+1)) for i in range(new_c)]
    mask = (df['cluster_num'].isin(cluster_list) & (df['sat_num'].isin(client_list)) & (df['ground_station'].isin(gs)))
    sat_new = df[mask]
    sat_new_sorted = sat_new.sort_values('Start Time Seconds Cumulative').reset_index()
    return sat_new_sorted

def choose_sat_csv_auto(df, og_s, og_c, new_s, new_c):
    if og_s%new_s == 0 and og_c%new_c == 0:
        client_list = [int(og_s/new_s*(i+1)) for i in range(new_s)]
        cluster_list = [int(og_c/new_c*(i+1)) for i in range(new_c)]
    mask = (df['cluster_num_1'].isin(cluster_list) & (df['sat_num_1'].isin(client_list)) & df['cluster_num_2'].isin(cluster_list) & (df['sat_num_2'].isin(client_list)) & (df['Duration (sec)'] != 7862400.000))
    sat_new = df[mask]
    sat_new_sorted = sat_new.sort_values('Start Time Seconds Cumulative').reset_index()
    return sat_new_sorted



# TODO: CLEAN THIS UP
if __name__ == "__main__":
    filepath2 = "/nfs-share/grk27/Documents/federated-satellites/Strategies/csv_stk/Chain1_Access_Data_9sat_5plane.csv"
    satellite_data2 = pd.read_csv(filepath2)
    satellite_data2['Start Time Seconds Cumulative'] = [time_to_int(datetime.strptime(x, "%d %b %Y %H:%M:%S.%f")) for x in satellite_data2['Start Time (UTCG)']]
    satellite_data2['End Time Seconds Cumulative'] = [time_to_int(datetime.strptime(x, "%d %b %Y %H:%M:%S.%f")) for x in satellite_data2['Stop Time (UTCG)']]
    satellite_data2['Start Time Seconds datetime'] = [(datetime.strptime(x, "%d %b %Y %H:%M:%S.%f")) for x in satellite_data2['Start Time (UTCG)']]
    satellite_data2['End Time Seconds datetime'] = [(datetime.strptime(x, "%d %b %Y %H:%M:%S.%f")) for x in satellite_data2['Stop Time (UTCG)']]
    satellite_data3 = satellite_data2.sort_values('Start Time Seconds Cumulative').reset_index()

    # print(satellite_data3['End Time Seconds datetime'])

    # print(satellite_data3['Start Time Seconds datetime'])

    # print(satellite_data3['End Time Seconds Cumulative'])

    # print(satellite_data3['Start Time Seconds Cumulative'])
    counter = 0
    for i in range(10):
        start_time = satellite_data3['Start Time Seconds datetime'].iloc[counter]
        delta = timedelta(hours=2)
        client_list = []

        while (timedelta(hours=16) > delta):
            client_list.append((int(satellite_data3['From Object'].iloc[counter][-2:-1])-1)*9+int(satellite_data3['From Object'].iloc[counter][-1:]))
            counter +=1
            delta = satellite_data3['Start Time Seconds datetime'].iloc[counter]-start_time
            print(delta)
        print(client_list)
        client_twice =[]
        for item in client_list:
            if client_list.count(item)>1:
                client_twice.append(item)