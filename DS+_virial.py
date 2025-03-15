import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp
import math
import os

dirc_path = '/home/habjan.e/'

import sys
sys.path.append(dirc_path + "TNG/TNG_cluster_dynamics")
import TNG_DA

### Variables 
cluster_number = 3
bootstrap_mc = 2 #100
dsp_sims = 10 #500
df_len = 10 #1000

proj_arr = np.random.uniform(-1, 1, (df_len, 3))

if __name__ == "__main__":

    corenum = os.cpu_count()                         #chosen based of the number of cores
    batch = math.ceil(df_len/corenum)     #batch determines the number of data points in each batched dataset
    projlist = [proj_arr[i:i+batch, :] for i in range(0, df_len, batch)] #make list of batched data
    
    pool = mp.Pool(processes = len(projlist))          #count processes are inititiated
    mplist = [pool.apply_async(TNG_DA.DSP_Virial_analysis, args = (cluster_number, pv, dsp_sims, bootstrap_mc)) for pv in projlist] #each batched dataset is assigned to a core 

results = [mplist[i].get() for i in range(len(mplist))] 

dsp_1 = []
dsp_2 = []
dsp_3 = []
sub_masses = []

for i in range(len(results)):

    dsp_1.append(results[i][0][0])
    dsp_2.append(results[i][0][1])
    dsp_3.append(results[i][0][2])
    sub_masses.append(results[i][1])

    if i == 0:
        df = results[i][2]
    else: 
        df_new = results[i][2]
        df = pd.concat([df, df_new], ignore_index=True)

save_path = dirc_path + "TNG/Data/data_DS+_virial_results/"
np.save(save_path + "DS+_array_1.npy", np.array(dsp_1))
np.save(save_path + "DS+_array_2.npy", np.array(dsp_2))
#np.save(save_path + "DS+_array_3.npy", np.array(dsp_3))
np.save(save_path + "subhalo_masses.npy", sub_masses)
df.to_csv(save_path + "DS+_Virial_df.csv", index=False)

print('Successfully Ran DS+_virial.py')