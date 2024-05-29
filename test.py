import os
import shutil


def search_and_replace(file_path, search_word, replace_word):
   with open(file_path, 'r') as file:
      file_contents = file.read()

      updated_contents = file_contents.replace(search_word, replace_word)

   with open(file_path, 'w') as file:
      file.write(updated_contents)

for i in range(1,12):
   
    # if os.path.exists("server"+str(i)+".py"):
    if os.path.exists("config_files11/config_"+str(i)+".ini"):
        # search_word = r'"WorkFP_5_11"'
        search_word = "client_cpu = 4"
        replace_word = "client_cpu = 2"

        # replace_word = r'"WorkFB_5_21"'
        # search_and_replace("server"+str(i)+".py", search_word, replace_word)
        search_and_replace("config_files11/config_"+str(i)+".ini", search_word, replace_word)
        
### SAVE INTO DIFFERENT FILES:
"""
list_name = [11,12,13]
for folder_num in list_name:

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
    
    search_word2 = '/nfs-share/grk27/Documents/more_tests2/federated-satellites/server.py'
    replace_word2 = "/nfs-share/grk27/Documents/more_tests2/federated-satellites/server"+str(folder_num)+".py"
    search_and_replace(slrum_file, search_word2, replace_word2)"""