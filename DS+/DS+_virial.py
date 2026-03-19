import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp
import math
import os
import argparse

dirc_path = '/home/habjan.e/'

import sys
sys.path.append(dirc_path + "TNG/TNG_cluster_dynamics")
import TNG_DA

### Variables 

parser = argparse.ArgumentParser(description="DS+ Virial Mass Script")
parser.add_argument("cluster_number", type=str, help="ID of the cluster to process")
args = parser.parse_args()
cluster_number = args.cluster_number
print('Processing Cluster ' + cluster_number + f' with {len(os.sched_getaffinity(0))} CPUs')

cluster_number = int(cluster_number)
bootstrap_mc = 0
dsp_sims = 10**3
df_len = 10**3
bootstrapping = False

save_path = '/projects/mccleary_group/habjan.e/TNG/Data/coherence_data/TNG/'
proj_arr = np.load(save_path + f"projection_array_{cluster_number}.npy")[:df_len, :]
#proj_arr = np.random.uniform(-1, 1, (df_len, 3))

if __name__ == "__main__":

    corenum = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    print("SLURM_CPUS_PER_TASK =", os.environ.get("SLURM_CPUS_PER_TASK"))
    print("Affinity CPUs =", len(os.sched_getaffinity(0)))
    batch = math.ceil(df_len/corenum)
    projlist = [proj_arr[i:i+batch, :] for i in range(0, df_len, batch)]

    with mp.Pool(processes=len(projlist)) as pool:
        async_results = [
            pool.apply_async(TNG_DA.DSP_Virial_analysis, args=(cluster_number, pv, dsp_sims, bootstrap_mc, bootstrapping))
            for pv in projlist
        ]
        results = [r.get() for r in async_results]


dsp_1_chunks, dsp_2_chunks, dsp_3_chunks = [], [], []
sub_masses_chunks, vel_disp_chunks, Munari_masses_chunks = [], [], []
theta_v_chunks, coh_v_chunks, err_l_v_chunks, err_u_v_chunks = [], [], [], []

for res in results:

    for i in range(len(res[0])):
        dsp_1_chunks.append(np.asarray(res[0][i][0]))
        dsp_2_chunks.append(np.asarray(res[0][i][1]))
        dsp_3_chunks.append(np.asarray(res[0][i][2]))

    sub_masses_chunks.append(np.array(res[1], dtype=object))
    vel_disp_chunks.append(np.array(res[2], dtype=object))
    Munari_masses_chunks.append(np.array(res[3], dtype=object))
    #theta_v_chunks.append(np.array(res[4], dtype=object))
    #coh_v_chunks.append(np.array(res[5], dtype=object))
    #err_l_v_chunks.append(np.array(res[6], dtype=object))
    #err_u_v_chunks.append(np.array(res[7], dtype=object))

dsp_1 = np.array(dsp_1_chunks)
dsp_2 = np.array(dsp_2_chunks)
dsp_3 = np.array(dsp_3_chunks, dtype=object)
sub_masses = np.concatenate(sub_masses_chunks, axis=0)
vel_disp = np.concatenate(vel_disp_chunks, axis=0)
sub_Munari_masses = np.concatenate(Munari_masses_chunks, axis=0)

#theta_v = np.concatenate(theta_v_chunks, axis=0)
#coh_v = np.concatenate(coh_v_chunks, axis=0)
#err_l_v = np.concatenate(err_l_v_chunks, axis=0)
#err_u_v = np.concatenate(err_u_v_chunks, axis=0)

df = pd.concat([res[4] for res in results], ignore_index=True)

save_path = "/projects/mccleary_group/habjan.e/TNG/Data/data_DS+_virial_results/"
np.save(save_path + f"DS+_array_1_{cluster_number}.npy", dsp_1)
np.save(save_path + f"DS+_array_2_{cluster_number}.npy", dsp_2)
np.save(save_path + f"DS+_array_3_{cluster_number}.npy", dsp_3)
np.save(save_path + f"subhalo_masses_{cluster_number}.npy", sub_masses)
np.save(save_path + f"velocity_dispersion_{cluster_number}.npy", vel_disp)
np.save(save_path + f"subhalo_Munari_masses_{cluster_number}.npy", sub_Munari_masses)
#np.save(save_path + f"theta_{cluster_number}.npy", np.array(theta_v))
#np.save(save_path + f"coh_{cluster_number}.npy", np.array(coh_v))
#np.save(save_path + f"coh_err_l_{cluster_number}.npy", np.array(err_l_v))
#np.save(save_path + f"coh_err_u_{cluster_number}.npy", np.array(err_u_v))
df.to_csv(save_path + f"DS+_Virial_df_{cluster_number}.csv", index=False)

print('Successfully Ran DS+_virial.py')