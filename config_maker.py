from configparser import ConfigParser
import os
import shutil

def search_and_replace(file_path, search_word, replace_word):
   with open(file_path, 'r') as file:
      file_contents = file.read()

      updated_contents = file_contents.replace(search_word, replace_word)

   with open(file_path, 'w') as file:
      file.write(updated_contents)

####################################################################
#####  Put the specific variables you want to test in a list   #####
####################################################################

variable_name = "wait_time"
epochs = [10] #[5,10,25,50,100]
sats = [10]#[1,2,5,10] #[1,2,5,10,25,50]
clusters = [1,4]#[1,2,5,10] #[1,5,25]
# ground_station_list = ["Seattle", "Hawthorne", "Cape_Canaveral","Boston","Colorado_Springs"]
# gs_list = ["[Boston,Seattle,Cape_Canaveral,Colorado_Springs,Hawthorne]"]#["[Boston,]","[Boston,Seattle]","[Boston,Seattle,Cape_Canaveral]","[Boston,Seattle,Cape_Canaveral,Colorado_Springs]","[Boston,Seattle,Cape_Canaveral,Colorado_Springs,Hawthorne]"]
gs_list = ["[Sioux_Falls]",
        "[Sioux_Falls,Sanya]",
        "[Sioux_Falls,Sanya,Johannesburg]",
        "[Sioux_Falls,Sanya,Johannesburg,Cordoba,Tromso]",
        "[Sioux_Falls,Sanya,Johannesburg,Cordoba,Tromso,Kashi,Beijing,Neustrelitz,Parepare,Alice_Springs]",
        "[Sioux_Falls,Sanya,Johannesburg,Cordoba,Tromso,Kashi,Beijing,Neustrelitz,Parepare,Alice_Springs,Fairbanks,Prince_Albert,Shadnagar]"]
# gs_list = ["[Sioux_Falls]",]
name = ["1_gs","2_gs","3_gs","5_gs","10_gs","all_gs"]
# name = ["1_gs"]
n_sat_c = 2
n_c = 25


#Get the configparser object
config_object = ConfigParser()

config_object["TEST_CONFIG"] = {
        "name": "Pls",
        "round": 500,
        "epochs": 3,
        "trial": 5,
        "clients": n_sat_c*n_c,
        "client_limit": 10,
        "dataset": "FEMNIST",
        "alg": "AutoFLSat",
        "learning_rate": 0.001,
        "momentum": 0.9,
        "wait_time" : 7,
        "sim_fname" : "/datasets/landsat/10s_4c_s_landsat_star_inter.csv",
        "n_sat_in_cluster" : n_sat_c,
        "n_cluster" : n_c,
        "slrum" : "y",
        "client_cpu": 2,
        "client_gpu": 0,
        "prox_term": 2,
        "gs_locations" : "[Boston,]"
    }

### SAVE INTO DIFFERENT FILES:

# NOTE: delete 1 client test case in the future

folder_num = 6 #int(config_object["TEST_CONFIG"]["prox_term"])*6+18
for gs,name_list in zip(gs_list,name):

    if not os.path.exists("config_files"+str(folder_num)):
        os.makedirs("config_files"+str(folder_num))

    filename_new = "server"+str(folder_num)+".py"
    if not os.path.exists(filename_new):
        shutil.copy("server.py",filename_new)
    
    search_word = 'config_files'
    replace_word = 'config_files'+str(folder_num)
    search_and_replace(filename_new, search_word, replace_word)
    
    slrum_file = "slrum"+str(folder_num)+".sh"
    if not os.path.exists(slrum_file):
        shutil.copy("slrum_fed_sats.sh",slrum_file)
    
    search_word2 = '/nfs-share/grk27/Documents/federated-satellites/server.py'
    replace_word2 = "/nfs-share/grk27/Documents/federated-satellites/server"+str(folder_num)+".py"
    search_and_replace(slrum_file, search_word2, replace_word2)
    

    test_num = 0

    for epoch in epochs:
        for sat in sats:
            for cluster in clusters:
                if test_num != 0:
                    # Change all of the configs
                    config_object["TEST_CONFIG"]["epochs"] = str(epoch)
                    config_object["TEST_CONFIG"]["n_sat_in_cluster"] = str(sat)
                    config_object["TEST_CONFIG"]["clients"] = str(sat*cluster)
                    config_object["TEST_CONFIG"]["n_cluster"] = str(cluster)
                    config_object["TEST_CONFIG"]["gs_locations"] = gs
                    config_object["TEST_CONFIG"]["name"] = name_list
                    #Write the above sections to config.ini file
                    with open('config_files'+str(folder_num)+'/config_'+str(test_num)+'.ini', 'w') as conf:
                        config_object.write(conf)
                    test_num +=1
                else:
                    test_num +=1
    folder_num +=1


