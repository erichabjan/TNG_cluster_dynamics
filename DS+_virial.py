import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os

import sys
sys.path.append("/work/mccleary_group/habjan.e/TNG/TNG_cluster_dynamics")
import TNG_DA

### Variables 
cluster_number = 3
proj_vector = np.array([0, 0, 1])
dsp_sims = 1
bootstrap_mc = 10

### Grab TNG data
pos, vel, group, sub_masses_TNG, h, halo_mass_TNG = TNG_DA.get_cluster_props(cluster_number)

### Project data to be observation-like
pos_2d, vel_los = TNG_DA.project_3d_to_2d(pos, vel, viewing_direction=proj_vector)

### Run DS+ and get back purity and completeness 
test, C, P = TNG_DA.run_dsp(pos_2d, vel_los, group, n_sims=dsp_sims)

### Bootstrapping on purity and completeness
C_err, P_err = TNG_DA.bootstrap_compleness_purity(mc_in = bootstrap_mc, pos_in = pos_2d, vel_in = vel_los, in_groups = group, n_sims=dsp_sims);

### Find the DS+ Groups
dsp_groups = TNG_DA.dsp_group_finder(dsp_output = test)

### Find Virial Masses
halo_mass_Virial, sub_masses_Virial = TNG_DA.virial_mass(position_2d = pos_2d, los_velocity = vel_los, groups = dsp_groups)

df = pd.DataFrame({
    "Cluster Index": [cluster_number],
    "TNG Position": [pos],
    "TNG Velocity": [vel],
    "TNG Group": [group],
    "TNG Subhalo Masses": [sub_masses_TNG],
    "TNG Halo Mass": [halo_mass_TNG],
    "Projection Vector": [proj_vector],
    "2D Position": [pos_2d],
    "LOS Velocity": [vel_los],
    "DS+ Results": [test],
    "Completeness": [C],
    "Completeness Uncertainty": [C_err],
    "Purity": [P],
    "Purity Uncertainty": [P_err],
    "Virial Subhalo Masses": [sub_masses_Virial],
    "Virial Halo Mass": [halo_mass_Virial]
})

save_path = "/work/mccleary_group/habjan.e/TNG/Data/data_DS+_virial_results/test_run.csv"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
df.to_csv(save_path, index=False)

print('Successfully Ran DS+_virial.py')