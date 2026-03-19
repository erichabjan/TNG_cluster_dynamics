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

parser = argparse.ArgumentParser(description="BAHAMAS script")
parser.add_argument("cluster_id", type=str, help="ID of the cluster to process")
parser.add_argument("dm_model", type=str, help="The Dark Matter model to process")
args = parser.parse_args()
cluster_id = args.cluster_id
dm_folder = args.dm_model
print(f'Running Coherence Length code on Cluster {cluster_id} in {dm_folder}')

### Import both the TNG gas and DM data
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


### Import R_500 for the cluster
h = data['h']

r200 = (data['R200'] * 10**3) / h

### Conversion from comoving coordinates to proper coordinates and Mpc -> kpc

dm_coordinates = (dm_coordinates * 10**3) / h

gas_coordinates = (gas_coordinates * 10**3) / h

### Voxel Parameters
ngrid_val = 512
L_val = 2150
pixel = L_val / ngrid_val

### Simulated data mask (all pixels)
data_mask = np.ones((ngrid_val, ngrid_val),dtype=float)

### DM particles
mass_dm = np.zeros(dm_coordinates.shape[0]) + 5.5 * 10**9


### Gas particles
mass_gas = np.zeros(gas_coordinates.shape[0]) + 1.09 * 10**9

co_len = []
co_len_err = []
theta_v_arr = []
coh_v_arr = []
err_l_v_arr = []
err_u_v_arr = []

theta_mass_list = []
P_mass_list = []
P_mass_sig_list  = []

theta_gas_list = []
P_gas_list = []
P_gas_sig_list  = []

theta_cross_list = []
P_cross_list = []
P_cross_sig_list = []

projections = 10**3
proj_vector = np.random.uniform(-1, 1, (projections, 3))

for i in range(projections):

    dm_coords_ro, _ = TNG_DA.rotate_to_viewing_frame(positions = dm_coordinates, velocities = np.empty_like(dm_coordinates), viewing_direction = proj_vector[i])
    dm_cube = TNG_DA.deposit_cic_scalar(positions = dm_coords_ro, L = L_val, ngrid = (ngrid_val, ngrid_val, ngrid_val), weights=mass_dm)
    dm_2d = np.nansum(dm_cube, axis = 2)

    gas_coords_ro, _ = TNG_DA.rotate_to_viewing_frame(positions = gas_coordinates, velocities = np.empty_like(gas_coordinates), viewing_direction = proj_vector[i])
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

    
    valid_mass = (
        np.isfinite(theta) &
        np.isfinite(power) &
        np.isfinite(sig_p) &
        (power > 0)
    )

    theta_mass = theta[valid_mass]
    P_mass = power[valid_mass]
    P_mass_sig  = np.maximum(sig_p[valid_mass], 1e-30)

    valid_gas = (
        np.isfinite(theta) &
        np.isfinite(power2) &
        np.isfinite(sig_p2) &
        (power2 > 0)
    )

    theta_gas = theta[valid_gas]
    P_gas = power2[valid_gas]
    P_gas_sig  = np.maximum(sig_p2[valid_gas], 1e-30)


    valid_cross = (
        np.isfinite(theta) &
        np.isfinite(power_cross) &
        np.isfinite(sig_p_cross) &
        (power2 > 0)
    )

    theta_cross = theta[valid_cross]
    P_cross     = power_cross[valid_cross]
    P_cross_sig  = np.maximum(sig_p_cross[valid_cross], 1e-30)
    co_len.append(s_cr)
    co_len_err.append(sigma_scr)
    theta_v_arr.append(theta[valid] / r200)
    coh_v_arr.append(c_ratio[valid])
    err_l_v_arr.append(np.clip(err_l[valid], 0.0, 1.0))
    err_u_v_arr.append(np.clip(err_u[valid], 0.0, 1.0))

    theta_mass_list.append(theta_mass)
    P_mass_list.append(P_mass)
    P_mass_sig_list.append(P_mass_sig)

    theta_gas_list.append(theta_gas)
    P_gas_list.append(P_gas)
    P_gas_sig_list.append(P_gas_sig)

    theta_cross_list.append(theta_cross)
    P_cross_list.append(P_cross)
    P_cross_sig_list.append(P_cross_sig)


save_path = '/projects/mccleary_group/habjan.e/TNG/Data/coherence_data/' + dm_folder + '/'

np.save(save_path + f"coherence_length_{cluster_id}.npy", np.array(co_len))
np.save(save_path + f"coherence_length_err_{cluster_id}.npy", np.array(co_len_err))
np.save(save_path + f"theta_{cluster_id}.npy", np.array(theta_v_arr))
np.save(save_path + f"coh_{cluster_id}.npy", np.array(coh_v_arr))
np.save(save_path + f"coh_err_l_{cluster_id}.npy", np.array(err_l_v_arr))
np.save(save_path + f"coh_err_u_{cluster_id}.npy", np.array(err_u_v_arr))
np.save(save_path + f"projection_array_{cluster_id}.npy", proj_vector)

np.save(save_path + f"theta_mass_{cluster_id}.npy", np.array(theta_mass_list))
np.save(save_path + f"power_mass_{cluster_id}.npy", np.array(P_mass_list))
np.save(save_path + f"power_mass_error_{cluster_id}.npy", np.array(P_mass_sig_list))

np.save(save_path + f"theta_gas_{cluster_id}.npy", np.array(theta_gas_list))
np.save(save_path + f"power_gas_{cluster_id}.npy", np.array(P_gas_list))
np.save(save_path + f"power_gas_error_{cluster_id}.npy", np.array(P_gas_sig_list))

np.save(save_path + f"theta_cross_{cluster_id}.npy", np.array(theta_cross_list))
np.save(save_path + f"power_cross_{cluster_id}.npy", np.array(P_cross_list))
np.save(save_path + f"power_cross_error_{cluster_id}.npy", np.array(P_cross_sig_list))

print('Successfully Ran coherence_analysis.py')