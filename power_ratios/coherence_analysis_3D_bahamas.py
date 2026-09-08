import numpy as np
import os
import argparse

import sys
sys.path.append("/home/habjan.e/TNG/TNG_cluster_dynamics")
sys.path.append("/home/habjan.e/TNG/Codes/Fourier_Analysis")

import TNG_DA
from fourier_analysis_code_3d import (
    make_theta_binning_3d,
    hann_window_3d,
    auto_power_3d,
    cross_power_3d,
    full_coherence_3d,
    coherence_length_single_3d
)

parser = argparse.ArgumentParser(description="3D BAHAMAS coherence analysis")
parser.add_argument("cluster_id", type=str, help="ID of the cluster to process")
parser.add_argument("dm_model", type=str, help="The Dark Matter model to process")
args = parser.parse_args()
cluster_id = args.cluster_id
dm_folder = args.dm_model
print(f'Running 3D Coherence Length code on Cluster {cluster_id} in {dm_folder}')

### Import both the BAHAMAS gas and DM data
data = np.load("/projects/mccleary_group/habjan.e/TNG/Data/" + dm_folder + "/GrNm_" + cluster_id + ".npz")

boxsize = 400
difpos = np.subtract(data['dm_pos'], data['CoP'])
difpos = (difpos + 0.5 * boxsize) % boxsize - 0.5 * boxsize

dm_coordinates = difpos # c Mpc / h
dm_velocities = data['dm_vel']

difpos = np.subtract(data['gas_pos'], data['CoP'])
difpos = (difpos + 0.5 * boxsize) % boxsize - 0.5 * boxsize

gas_coordinates = difpos
gas_velocities = data['gas_vel']


### Import R_200 for the cluster
h = data['h']

r200 = (data['R200'] * 10**3) / h

### Conversion from comoving coordinates to proper coordinates and Mpc -> kpc

dm_coordinates = (dm_coordinates * 10**3) / h

gas_coordinates = (gas_coordinates * 10**3) / h

### Voxel Parameters
ngrid_val = 512
L_val = 2150
pixel = L_val / ngrid_val

### DM particles
mass_dm = np.zeros(dm_coordinates.shape[0]) + 5.5 * 10**9


### Gas particles
mass_gas = np.zeros(gas_coordinates.shape[0]) + 1.09 * 10**9

### Deposit particles into 3D cubes (single orientation)
dm_coords_ro, _ = TNG_DA.rotate_to_viewing_frame(
    positions=dm_coordinates,
    velocities=np.empty_like(dm_coordinates),
    viewing_direction=np.array([0, 0, 1])
)
data_mass = TNG_DA.deposit_cic_scalar(
    positions=dm_coords_ro, L=L_val,
    ngrid=(ngrid_val, ngrid_val, ngrid_val),
    weights=mass_dm
)

gas_coords_ro, _ = TNG_DA.rotate_to_viewing_frame(
    positions=gas_coordinates,
    velocities=np.empty_like(gas_coordinates),
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
save_path = '/projects/mccleary_group/habjan.e/TNG/Data/coherence_data_3D/' + dm_folder + '/'
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

print('Successfully Ran coherence_analysis_3D_bahamas.py')
