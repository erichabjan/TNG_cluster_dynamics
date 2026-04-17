import numpy as np
import os
import argparse

import sys
sys.path.append("/home/habjan.e/TNG/Codes/TNG_workshop")
sys.path.append("/home/habjan.e/TNG/TNG_cluster_dynamics")
sys.path.append("/home/habjan.e/TNG/Codes/Fourier_Analysis")

import iapi_TNG as iapi
import h5py
import TNG_DA
from fourier_analysis_code_3d import (
    make_theta_binning_3d,
    hann_window_3d,
    auto_power_3d,
    cross_power_3d,
    full_coherence_3d,
    coherence_length_single_3d
)

parser = argparse.ArgumentParser(description="3D TNG coherence analysis")
parser.add_argument("cluster_ID", type=str, help="ID of the cluster to process")
args = parser.parse_args()
cluster_id = int(args.cluster_ID)
print(f'Running 3D Coherence Length code on Cluster {cluster_id}')


### Import both the TNG gas and DM data
dm_fname = f'/projects/mccleary_group/habjan.e/TNG/Data/TNG_data/5r200_data/dm_within_5r200_{cluster_id}.hdf5'

with h5py.File(dm_fname, 'r') as f:

    dm_coordinates = f['PartType1']['Coordinates'][:]
    dm_velocities = f['PartType1']['Velocities'][:]

gas_fname = f'/projects/mccleary_group/habjan.e/TNG/Data/TNG_data/5r200_data/gas_within_5r200_{cluster_id}.hdf5'

with h5py.File(gas_fname, 'r') as f:

    gas_coordinates = f['PartType0']['Coordinates'][:]
    gas_velocities = f['PartType0']['Velocities'][:]
    gas_masses = f['PartType0']['Masses'][:]


### Import R_500 for the cluster
TNG_data_path = '/home/habjan.e/TNG/Data/'
sim='TNG300-1'

baseUrl = 'http://www.tng-project.org/api/'
simUrl = baseUrl+sim
simdata = iapi.get(simUrl)
h = simdata['hubble']

r_200_clusters = iapi.getHaloField(field = 'Group_R_Crit200', simulation=sim, snapshot=99, fileName= TNG_data_path+'TNG_data/'+sim+'_Group_R_Crit200', rewriteFile=0)
r200 = r_200_clusters[cluster_id] / h


### Voxel Parameters
ngrid_val = 512
L_val = 2150
pixel = L_val / ngrid_val

### Grid the DM particles
dm_coords = TNG_DA.coord_cm_corr(cluster_ind = cluster_id, coordinates = dm_coordinates) / h ### kpc
mass_dm = np.zeros(dm_coords.shape[0]) + 5.9 * 10**7 ### solar masses


### Grid the Gas particles
gas_coords = TNG_DA.coord_cm_corr(cluster_ind = cluster_id, coordinates = gas_coordinates) / h ### kpc
mass_gas = (gas_masses * 10**10) / h

### Deposit particles into 3D cubes (single orientation)
dm_coords_ro, _ = TNG_DA.rotate_to_viewing_frame(
    positions=dm_coords,
    velocities=np.empty_like(dm_coords),
    viewing_direction=np.array([0, 0, 1])
)
data_mass = TNG_DA.deposit_cic_scalar(
    positions=dm_coords_ro, L=L_val,
    ngrid=(ngrid_val, ngrid_val, ngrid_val),
    weights=mass_dm
)

gas_coords_ro, _ = TNG_DA.rotate_to_viewing_frame(
    positions=gas_coords,
    velocities=np.empty_like(gas_coords),
    viewing_direction=np.array([0, 0, 1])
)
data_xray = TNG_DA.deposit_cic_scalar(
    positions=gas_coords_ro, L=L_val,
    ngrid=(ngrid_val, ngrid_val, ngrid_val),
    weights=mass_gas
)

print("Mass cube shape :", data_mass.shape)
print("Xray cube shape :", data_xray.shape)
print("Mass min/max    :", np.nanmin(data_mass), np.nanmax(data_mass))
print("Xray min/max    :", np.nanmin(data_xray), np.nanmax(data_xray))

### 3D theta binning
nbins = 30
pixel_size = L_val / data_mass.shape[-1]
tet_1grid = make_theta_binning_3d(data_mass, pixel_size, nbins)

### Apply 3D Hann window
data_mass_win = hann_window_3d(data_mass)
data_xray_win = hann_window_3d(data_xray)

### 3D auto-power spectra
bin_center, pairs, amp_mass, power_mass, sig_mass = auto_power_3d(
    data_mass_win, tet_1grid, pixel_size, grad=0
)

bin_center_x, pairs_x, amp_xray, power_xray, sig_xray = auto_power_3d(
    data_xray_win, tet_1grid, pixel_size, grad=0
)

### 3D cross-power spectrum
power_cross, sig_cross, power_mass_check, power_xray_check, Nk_cross = cross_power_3d(
    data_mass_win, amp_mass, amp_xray, tet_1grid, pixel_size
)

### 3D coherence
coh_3d, coh_low_3d, coh_high_3d = full_coherence_3d(
    power_cross, sig_cross,
    power_mass, sig_mass,
    power_xray, sig_xray
)

### 3D coherence length
theta_3d = 1.0 / bin_center

theta_cr_3d, sigma_theta_cr_3d = coherence_length_single_3d(
    coh_3d,
    coh_high_3d,
    coh_low_3d,
    theta_3d,
    reliable_mask=np.ones_like(coh_3d, dtype=bool),
    threshold=0.9,
    theta_min_noise=np.nan,
    theta_floor=0.0,
    theta_fallback=0.0,
    max_drop=0.2
)

s_cr_3d = theta_cr_3d / r200
sigma_scr_3d = sigma_theta_cr_3d / r200

print(f"theta_cr_3d = {theta_cr_3d}")
print(f"sigma_theta_cr_3d = {sigma_theta_cr_3d}")
print(f"s_cr_3d = {s_cr_3d}")

### Save results
save_path = '/projects/mccleary_group/habjan.e/TNG/Data/coherence_data_3D/TNG/'
os.makedirs(save_path, exist_ok=True)

np.save(save_path + f"coherence_length_{cluster_id}.npy", np.array([s_cr_3d]))
np.save(save_path + f"coherence_length_err_{cluster_id}.npy", np.array([sigma_scr_3d]))
np.save(save_path + f"theta_cr_{cluster_id}.npy", np.array([theta_cr_3d]))
np.save(save_path + f"theta_cr_err_{cluster_id}.npy", np.array([sigma_theta_cr_3d]))

np.save(save_path + f"theta_{cluster_id}.npy", theta_3d / r200)
np.save(save_path + f"coh_{cluster_id}.npy", coh_3d)
np.save(save_path + f"coh_err_l_{cluster_id}.npy", coh_low_3d)
np.save(save_path + f"coh_err_u_{cluster_id}.npy", coh_high_3d)

np.save(save_path + f"theta_mass_{cluster_id}.npy", theta_3d)
np.save(save_path + f"power_mass_{cluster_id}.npy", power_mass)
np.save(save_path + f"power_mass_error_{cluster_id}.npy", sig_mass)

np.save(save_path + f"theta_gas_{cluster_id}.npy", theta_3d)
np.save(save_path + f"power_gas_{cluster_id}.npy", power_xray)
np.save(save_path + f"power_gas_error_{cluster_id}.npy", sig_xray)

np.save(save_path + f"theta_cross_{cluster_id}.npy", theta_3d)
np.save(save_path + f"power_cross_{cluster_id}.npy", power_cross)
np.save(save_path + f"power_cross_error_{cluster_id}.npy", sig_cross)

print('Successfully Ran coherence_analysis_3D_tng.py')
