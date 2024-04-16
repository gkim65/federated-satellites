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
# test_cases = [4,8,16,32,64]
# rounds = [50,40,25,15,9]

n_sat_c = 1
n_c = 5

config_object["TEST_CONFIG"] = {
        "name": "fedavgsats_9_5",
        "round": 10,
        "epochs": 3,
        "trial": 5,
        "clients": n_sat_c*n_c,
        "dataset": "FEMNIST",
        "learning_rate": 0.001,
        "momentum": 0.9,
        "wait_time" : 7,
        "sim_fname" : "Strategies/csv_stk/1s_5c_us.csv",
        "n_sat_in_cluster" : n_sat_c,
        "n_cluster" : n_c,
        "gs_locations" : "us"
    }

# for test,round in zip(test_cases,rounds):
for test,round in zip([10],[10]):
    #Write the above sections to config.ini file
    config_object["TEST_CONFIG"][variable_name] = str(test)

    config_object["TEST_CONFIG"]["round"] = str(round)
    with open('config_files/config_'+str(test)+'.ini', 'w') as conf:
        config_object.write(conf)

