import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd

import os
import argparse

import sys
sys.path.append("/home/habjan.e/TNG/Codes/TNG_workshop")
sys.path.append("/home/habjan.e/TNG/TNG_cluster_dynamics")
sys.path.append("/home/habjan.e/TNG/Codes/Fourier_Analysis")

import iapi_TNG as iapi
import h5py
import TNG_DA
import Full_Fourier_analysis_code

parser = argparse.ArgumentParser(description="DS+ Virial Mass Script")
parser.add_argument("cluster_ID", type=str, help="ID of the cluster to process")
args = parser.parse_args()
cluster_id = int(args.cluster_ID)
print(f'Running Coherence Length code on Cluster {cluster_id}')


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

### Simulated data mask (all pixels)
data_mask = np.ones((ngrid_val, ngrid_val),dtype=float)

### Grid the DM particles
dm_coords = TNG_DA.coord_cm_corr(cluster_ind = cluster_id, coordinates = dm_coordinates) / h ### kpc
mass_dm = np.zeros(dm_coords.shape[0]) + 5.9 * 10**7 ### solar masses


### Grid the Gas particles
gas_coords = TNG_DA.coord_cm_corr(cluster_ind = cluster_id, coordinates = gas_coordinates) / h ### kpc
mass_gas = (gas_masses * 10**10) / h #np.zeros(gas_coords.shape[0]) + 1.1 * 10**7  ### solar masses

co_len = []
co_len_err = []
theta_v_arr = []
coh_v_arr = []
err_l_v_arr = []
err_u_v_arr = []

projections = 10**3
proj_vector = np.random.uniform(-1, 1, (projections, 3))

for i in range(projections):

    dm_coords_ro, _ = TNG_DA.rotate_to_viewing_frame(positions = dm_coords, velocities = np.empty_like(dm_coords), viewing_direction = proj_vector[i])
    dm_cube = TNG_DA.deposit_cic_scalar(positions = dm_coords_ro, L = L_val, ngrid = (ngrid_val, ngrid_val, ngrid_val), weights=mass_dm)
    dm_2d = np.nansum(dm_cube, axis = 2)

    gas_coords_ro, _ = TNG_DA.rotate_to_viewing_frame(positions = gas_coords, velocities = np.empty_like(gas_coords), viewing_direction = proj_vector[i])
    gas_cube = TNG_DA.deposit_cic_scalar(positions = gas_coords_ro, L = L_val, ngrid = (ngrid_val, ngrid_val, ngrid_val), weights=mass_gas)
    gas_2d = np.nansum(gas_cube, axis = 2)


    ### Angular binning
    n_bins = 30
    tet_1grid = Full_Fourier_analysis_code.make_theta_binning(
        dm_2d,        
        pixel_size=pixel,
        nbins=n_bins
    )

    ### Power spectra
    sample_size= ngrid_val * ngrid_val

    #Fluctuation maps
    image_mass = np.nan_to_num(dm_2d, nan=0.0)
    image_gas = np.nan_to_num(gas_2d, nan=0.0)
    average_mass = np.average(image_mass)
    average_gas = np.average(image_gas)   
    fluc_mass = (image_mass-average_mass)    
    fluc_gas = (image_gas-average_gas)

    #Application of window function for edge effect corrections
    windowed_mass = Full_Fourier_analysis_code.hann_window_power_spectrum(fluc_mass)
    windowed_gas = Full_Fourier_analysis_code.hann_window_power_spectrum(fluc_gas)

    #This variable is 1 if we want to remove eventual gradients from the maps in Fourier space
    #IT IS 0 FOR MOSTLY ALL APPLICATIONS 
    grad=0
    #This variable is 1 if we want to compute the FOV in sterad from arcsec units
    unit=0

    #COMPUTE POWER SPECTRA AND COHERENCE
    k_p, pairs, amp, power, sig_p = Full_Fourier_analysis_code.auto_power_obs(windowed_mass, data_mask, tet_1grid, pixel, grad, unit, outfile = None, writefits = None )
    k_p2, pairs2, amp2, power2, sig_p2 = Full_Fourier_analysis_code.auto_power_obs(windowed_gas, data_mask, tet_1grid, pixel, grad, unit, outfile = None, writefits = None )

    power_cross, sig_p_cross = Full_Fourier_analysis_code.cross_power(windowed_mass, data_mask, amp, amp2, tet_1grid, pixel, unit)
    c_ratio, err_l, err_u = Full_Fourier_analysis_code.full_coherence(power_cross, sig_p_cross, power, sig_p, power2, sig_p2, sample_size)

    theta = 1.0 / k_p
    coh_lower = err_l
    coh_upper = err_u

    valid = (
        np.isfinite(theta) &
        np.isfinite(c_ratio) &
        np.isfinite(err_l) &
        np.isfinite(err_u) &
        (c_ratio >= 0)
        )

    s_cr, sigma_scr, theta_cr, sigma_theta = Full_Fourier_analysis_code.coherence_length_single(
        c_ratio[valid],
        coh_upper[valid],
        coh_lower[valid],
        theta[valid],
        r200
    )

    co_len.append(s_cr)
    co_len_err.append(sigma_scr)
    theta_v_arr.append(theta[valid] / r200)
    coh_v_arr.append(c_ratio[valid])
    err_l_v_arr.append(np.clip(err_l[valid], 0.0, 1.0))
    err_u_v_arr.append(np.clip(err_u[valid], 0.0, 1.0))


save_path = '/projects/mccleary_group/habjan.e/TNG/Data/coherence_data/TNG/'

np.save(save_path + f"coherence_length_{cluster_id}.npy", np.array(co_len))
np.save(save_path + f"coherence_length_err_{cluster_id}.npy", np.array(co_len_err))
np.save(save_path + f"theta_{cluster_id}.npy", np.array(theta_v_arr))
np.save(save_path + f"coh_{cluster_id}.npy", np.array(coh_v_arr))
np.save(save_path + f"coh_err_l_{cluster_id}.npy", np.array(err_l_v_arr))
np.save(save_path + f"coh_err_u_{cluster_id}.npy", np.array(err_u_v_arr))
np.save(save_path + f"projection_array_{cluster_id}.npy", proj_vector)

print('Successfully Ran coherence_analysis.py')