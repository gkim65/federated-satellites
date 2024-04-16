import pandas as pd
from datetime import datetime
from datetime import timedelta
import math

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