from configparser import ConfigParser
import os

if not os.path.exists("config_files"):
    os.makedirs("config_files")

#Get the configparser object
config_object = ConfigParser()

####################################################################
#####  Put the specific variables you want to test in a list   #####
####################################################################
variable_name = "wait_time"
epochs = [10]#[5,10,25,50,100]
sats = [50]#[1,2,5,10,25,50]
clusters = [1]#[1,5,25]
# ground_station_list = ["Seattle", "Hawthorne", "Cape_Canaveral","Boston","Colorado_Springs"]
gs_list = ["[Boston,Seattle,Cape_Canaveral,Colorado_Springs,Hawthorne]"]#["[Boston,]","[Boston,Seattle]","[Boston,Seattle,Cape_Canaveral]","[Boston,Seattle,Cape_Canaveral,Colorado_Springs]","[Boston,Seattle,Cape_Canaveral,Colorado_Springs,Hawthorne]"]
n_sat_c = 2
n_c = 5

config_object["TEST_CONFIG"] = {
        "name": "fed_avg_NEWTESTS",
        "round": 500,
        "epochs": 3,
        "trial": 5,
        "clients": n_sat_c*n_c,
        "dataset": "FEMNIST",
        "learning_rate": 0.001,
        "momentum": 0.9,
        "wait_time" : 7,
        "sim_fname" : "Strategies/csv_stk/50s_25c_smaller_us.csv",
        "n_sat_in_cluster" : n_sat_c,
        "n_cluster" : n_c,
        "gs_locations" : "[Boston,]"
    }

test_num = 0
for epoch in epochs:
    for sat in sats:
        for cluster in clusters:
            for gs in gs_list:

                # Change all of the configs
                config_object["TEST_CONFIG"]["epochs"] = str(epoch)
                config_object["TEST_CONFIG"]["n_sat_in_cluster"] = str(sat)
                config_object["TEST_CONFIG"]["clients"] = str(sat*cluster)
                config_object["TEST_CONFIG"]["n_cluster"] = str(cluster)
                config_object["TEST_CONFIG"]["gs_locations"] = gs

                #Write the above sections to config.ini file
                with open('config_files/config_'+str(test_num)+'.ini', 'w') as conf:
                    config_object.write(conf)
                test_num +=1

