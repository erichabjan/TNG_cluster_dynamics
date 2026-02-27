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

r200 = data['R200'] / h

### Conversion from comoving coordinates to proper coordinates and Mpc -> kpc

dm_coordinates = (dm_coordinates * 10**3) / h

gas_coordinates = (gas_coordinates * 10**3) / h

### Voxel Parameters
ngrid_val = 512
L_val = 2150
pixel = L_val / ngrid_val

### Simulated data mask (all pixels)
data_mask = np.ones((ngrid_val, ngrid_val),dtype=float)

### Grid the DM particles
mass_dm = np.zeros(dm_coordinates.shape[0]) + 5.5 * 10**9
dm_cube = TNG_DA.deposit_cic_scalar(positions = dm_coordinates, L = L_val, ngrid = (ngrid_val, ngrid_val, ngrid_val), weights=mass_dm)


### Grid the Gas particles
mass_gas = np.zeros(gas_coordinates.shape[0]) + 1.09 * 10**9
gas_cube = TNG_DA.deposit_cic_scalar(positions = gas_coordinates, L = L_val, ngrid = (ngrid_val, ngrid_val, ngrid_val), weights=mass_gas)

co_len = []
co_len_err = []
theta_v_arr = []
coh_v_arr = []
err_l_v_arr = []
err_u_v_arr = []

for i in range(3):

    dm_2d = np.nansum(dm_cube, axis = i)
    gas_2d = np.nansum(gas_cube, axis = i)


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
    k_p2, pairs2, amp2, power2, sig_p2 = Full_Fourier_analysis_code.auto_power_obs(windowed_gas, data_mask, tet_1grid, pixel, unit, grad, outfile = None, writefits = None )

    power_cross, sig_p_cross = Full_Fourier_analysis_code.cross_power(windowed_mass, data_mask, amp, amp2, tet_1grid, pixel, unit)
    c_ratio, err_l, err_u = Full_Fourier_analysis_code.full_coherence(power_cross, sig_p_cross, power, sig_p, power2, sig_p2, sample_size)

    theta = 1.0 / k_p
    coh_lower = err_l
    coh_upper = err_u

    s_cr, sigma_scr, theta_cr, sigma_theta = Full_Fourier_analysis_code.coherence_length_single(
        c_ratio,
        coh_upper,
        coh_lower,
        theta,
        r200
    )

    valid = (
        np.isfinite(theta) &
        np.isfinite(c_ratio) &
        np.isfinite(err_l) &
        np.isfinite(err_u) &
        (c_ratio >= 0)
        )

    co_len.append(s_cr)
    co_len_err.append(sigma_scr)
    theta_v_arr.append(theta[valid] / r200)
    coh_v_arr.append(c_ratio[valid])
    err_l_v_arr.append(np.clip(err_l[valid], 0.0, 1.0))
    err_u_v_arr.append(np.clip(err_u[valid], 0.0, 1.0))


save_path = '/projects/mccleary_group/habjan.e/TNG/Data/coherence_data/' + dm_folder + '/'

np.save(save_path + f"coherence_length_{cluster_id}.npy", np.array(co_len))
np.save(save_path + f"coherence_length_err_{cluster_id}.npy", np.array(co_len_err))
np.save(save_path + f"theta_{cluster_id}.npy", np.array(theta_v_arr))
np.save(save_path + f"coh_{cluster_id}.npy", np.array(coh_v_arr))
np.save(save_path + f"coh_err_l_{cluster_id}.npy", np.array(err_l_v_arr))
np.save(save_path + f"coh_err_u_{cluster_id}.npy", np.array(err_u_v_arr))

print('Successfully Ran coherence_analysis.py')