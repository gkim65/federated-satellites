from configparser import ConfigParser
import os

if not os.path.exists("config_files"):
    os.makedirs("config_files")

#Get the configparser object
config_object = ConfigParser()

####################################################################
#####  Put the specific variables you want to test in a list   #####
####################################################################
variable_name = "epochs"
test_cases = [1,2,3,4]

config_object["TEST_CONFIG"] = {
        "name": "yoo",
        "round": 5,
        "epochs": 25,
        "trial": 5,
        "clients": 8,
        "dataset": "FEMNIST",
        "learning_rate": 0.001,
        "momentum": 0.9
    }

for i in test_cases:
    #Write the above sections to config.ini file
    config_object["TEST_CONFIG"][variable_name] = str(i)
    with open('config_files/config_'+str(i)+'.ini', 'w') as conf:
        config_object.write(conf)

