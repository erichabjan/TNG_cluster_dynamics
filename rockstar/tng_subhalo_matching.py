dirc_path = '/home/habjan.e/'

import sys
sys.path.append(dirc_path + "TNG/TNG_cluster_dynamics")
import TNG_DA
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
import statistics

sys.path.append(dirc_path + 'TNG/Codes/TNG_workshop')
import iapi_TNG as iapi

import h5py
import argparse

# Cluster ID 
parser = argparse.ArgumentParser(description="TNG subhalo matching")
parser.add_argument("cluster_ID", type=str, help="ID of the cluster to process")
args = parser.parse_args()
cluster_id = args.cluster_ID
print('Subhalo matching for cluster ' + cluster_id)
cluster_in = int(cluster_id)

# Import SubhaloLenType and SubhaloGrNr
sim = 'TNG300-1'
TNG_data_path = dirc_path + 'TNG/Data/'

baseUrl = 'http://www.tng-project.org/api/'
r=iapi.get(baseUrl)

SubhaloLenType = iapi.getSubhaloField('SubhaloLenType', simulation=sim, snapshot=99, fileName=TNG_data_path+'TNG_data/'+sim+'_SubhaloLenType', rewriteFile=0)
Group_num = iapi.getSubhaloField('SubhaloGrNr', simulation=sim, fileName=TNG_data_path+'TNG_data/'+sim+'_SubhaloGrNr', rewriteFile=0)

# Index the SubhaloLenType array to get the particle lengths in the FoF halo of interest 
sub_ind = np.where(Group_num == cluster_in)[0]
sub_lens = SubhaloLenType[sub_ind, :]

# Import TNG particle IDs for the FoF halo
#fName = f'/projects/mccleary_group/habjan.e/TNG/Data/TNG_data/halo_cutouts_dm_{cluster_in}.hdf5'
fName = f'/projects/mccleary_group/habjan.e/TNG/Data/TNG_data/5r200_data/dm_within_5r200_{cluster_in}.hdf5'

with h5py.File(fName, 'r') as f:

    ids = f['PartType1']['ParticleIDs'][:]


# Make an array of subfind-subhalo COM particles
offsets = np.array([
    np.sum(sub_lens[:i, 1]) for i in range(sub_lens.shape[0])
])

com_ids = []
num_com_parts = 50

com_ids = [ids[offsets[i]: offsets[i] + sub_lens[i, 1]][:num_com_parts] for i in range(sub_lens.shape[0])] 

# Use the subfind-subhalo COM IDs to match to rockstar-subhalos 
rockstar_output = '/projects/mccleary_group/habjan.e/TNG/Data/rockstar_output/tng_rockstar_output'
members_path = rockstar_output + f'/rockstar_subhalo_members_{cluster_in}.list'

members = pd.read_csv(members_path, sep=r"\s+", names=["halo_id","particle_id"])
member_id = np.array(members['particle_id'])
mem_halo_id = np.array(members['halo_id'])

subhalo_ids = []

for i in range(len(com_ids)):

    sub_bool = np.isin(member_id, com_ids[i])

    sub_membership_arr = mem_halo_id[sub_bool]

    if sub_membership_arr.shape[0] == 0:
        subhalo_ids.append(np.nan)

    else:
        subhalo_ids.append(statistics.mode(sub_membership_arr))

subhalo_ids = np.array(subhalo_ids)

# Save rockstar-subhalo IDs to the TNG rockstar output folder

np.save(rockstar_output + f'/matched_subhalo_members_{cluster_in}.npy', subhalo_ids)