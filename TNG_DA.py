dirc_path = '/home/habjan.e/'

import sys
sys.path.append(dirc_path + 'TNG/Codes/DS+/MilaDS')
import milaDS

sys.path.append(dirc_path + 'TNG/Codes/TNG_workshop')
import iapi_TNG as iapi

import numpy as np
import h5py #most TNG data is downloaded as hdf5 files
import matplotlib.pyplot as plt
import os.path
import pandas as pd
from IPython.display import display, Markdown

baseUrl = 'http://www.tng-project.org/api/'

dirc=dirc_path + 'TNG/TNG_workshop/'
sim='TNG300-1'
r=iapi.get(baseUrl)

TNG_data_path = dirc_path + 'TNG/Data/'

simUrl = baseUrl+sim
simdata = iapi.get(simUrl)

def gettree(snapnum,subid):
    snapnum=str(snapnum)
    fName = TNG_data_path+'TNG_data/'+'sublink_mpb_'+str(subid)+'_'+str(snapnum)
    if os.path.exists(fName+'.hdf5'):
        return(fName+'.hdf5')
    url='https://www.tng-project.org/api/TNG300-1/snapshots/'+snapnum+'/subhalos/'+str(subid)+'/sublink/mpb.hdf5'
    tree=iapi.get(url,fName=fName)
    return(tree)



def get_cluster_props(cluster_ind):

    ### Arrays of the M_200 of halos and the halo centers, not necessarily galaxy clusters
    Crit200 = iapi.getHaloField(field = 'Group_M_Crit200', simulation=sim, snapshot=99, fileName= TNG_data_path+'TNG_data/'+sim+'_Group_M_Crit200', rewriteFile=0)
    halo_center = iapi.getHaloField(field = 'GroupPos', simulation=sim, snapshot=99, fileName=TNG_data_path+'TNG_data/'+sim+'_GroupPos', rewriteFile=0)

    ### Make a cluster mass cut, now we have galaxy cluters
    quan_val = np.log10(5 * 10**14)
    Group_num = iapi.getSubhaloField('SubhaloGrNr', simulation=sim, fileName=TNG_data_path+'TNG_data/'+sim+'_SubhaloGrNr', rewriteFile=0) # Import array that identifies the halo each subhalo belongs to 
    h = simdata['hubble']
    halo_mass = Crit200[cluster_ind] * 10**10 / h

    if halo_mass > 10**quan_val:
        display(Markdown(f"Cluster {cluster_ind} has a $M_{{200}}$ greater than {np.round(quan_val)} $\\log(M_\\odot)$"))
    else:
        display(Markdown(f"Cluster {cluster_ind} has a $M_{{200}}$ less than {np.round(quan_val)} $\\log(M_\\odot)$"))

    ### Pick a galaxy cluster
    sub_ind = np.where(Group_num == cluster_ind)[0]

    ### Find the center of the galaxy cluster
    pos_comoving = halo_center[cluster_ind, :] ## position in units c * kpc / h
    cm_pos = pos_comoving / simdata['hubble']

    ### Import Subhalo Positons, Velocities, Photometrics 
    sub_comoving = iapi.getSubhaloField('SubhaloPos', simulation=sim, snapshot=99, fileName=TNG_data_path+'TNG_data/'+sim+'_SubhaloPos', rewriteFile=0)
    sub_vel = iapi.getSubhaloField('SubhaloVel', simulation=sim, snapshot=99, fileName=TNG_data_path+'TNG_data/'+sim+'_SubhaloVel', rewriteFile=0)
    sub_photo = iapi.getSubhaloField('SubhaloStellarPhotometrics', snapshot=99, simulation=sim, fileName=TNG_data_path+'TNG_data/'+sim+'_SubhaloStellarPhotometrics', rewriteFile=0)
    sub_masses = iapi.getSubhaloField('SubhaloMass', snapshot=99, simulation=sim, fileName=TNG_data_path+'TNG_data/'+sim+'_SubhaloMass', rewriteFile=0)
    SubhaloMassType = iapi.getSubhaloField('SubhaloMassType', snapshot=99, simulation=sim, fileName=TNG_data_path+'TNG_data/'+sim+'_SubhaloMassType', rewriteFile=0)
    sub_masses, SubhaloMassType = sub_masses * 10**10 / h, SubhaloMassType*10**10 / h

    #L is length of box, halfbox is L/2
    sub_uncorrected_pos = sub_comoving / h
    L = np.max(sub_uncorrected_pos)
    halfbox = L / 2

    difpos = np.subtract(sub_uncorrected_pos, cm_pos)
    #Replace values that are affected by boundary conditions
    difpos = np.where( abs(difpos) > halfbox, abs(difpos)- L , difpos)
    distsq = np.sum(np.square(difpos),axis=1)

    ### Center the position array relative to the cluster, make arrays with cluster subhalo parameters
    cl_pos, cl_vel, cl_photo, cl_masses, cl_SubhaloMassType = difpos[sub_ind], sub_vel[sub_ind], sub_photo[sub_ind], sub_masses[sub_ind],  SubhaloMassType[sub_ind, :]

    ### Make a magnitude cut so that subhalos are actually galaxies, not DM halos 
    mag_cut = -18
    bright_ind = cl_photo[:, 4] < mag_cut
    pos, vel, photo, subhalo_masses, subhalo_type = cl_pos[bright_ind], cl_vel[bright_ind], cl_photo[bright_ind], cl_masses[bright_ind], cl_SubhaloMassType[bright_ind, :]

    subhalos = sub_ind[bright_ind]    ### This gives the index of each subhalo in the cluster

    ### Import the rockstar outputs for the FoF halo

    rockstar_path = f'/projects/mccleary_group/habjan.e/TNG/Data/rockstar_output/tng_rockstar_output/matched_subhalo_members_{cluster_ind}.npy'
    cl_sub_groups = np.load(rockstar_path)
    groups = cl_sub_groups[bright_ind]
    
    return pos, vel, groups, subhalo_masses, subhalo_type, h, halo_mass

def coord_cm_corr(cluster_ind, coordinates):

    """
    Corrects for TNG coordinates into cluster-centric coordinates
    
    Args:
        cluster_ind (integer): TNG cluster index.
        coordinates (numpy.ndarray): Array of shape (N, 3) containing TNG simulation coordinates.

    Returns:
        coordinates (numpy.ndarray): Array of shape (N, 3) with cluster-centric coordinates.
    """

    halo_center = iapi.getHaloField(field ='GroupPos', simulation=sim, snapshot=99, fileName=TNG_data_path+'TNG_data/'+sim+'_GroupPos', rewriteFile=0)
    h = simdata['hubble']

    pos_comoving = halo_center[cluster_ind, :] ## position in units c * kpc / h
    cm_pos = pos_comoving #/ h

    sub_uncorrected_pos = coordinates #/ h
    L = np.max(sub_uncorrected_pos)
    halfbox = L / 2

    difpos = np.subtract(sub_uncorrected_pos, cm_pos)
    #Replace values that are affected by boundary conditions
    difpos = np.where( abs(difpos) > halfbox, abs(difpos)- L , difpos)
    distsq = np.sum(np.square(difpos),axis=1)

    return difpos

def project_3d_to_2d(positions, velocities, viewing_direction=np.array([0, 0, 1])):
    """
    Project 3D positions and velocities onto a 2D plane perpendicular to the viewing direction.
    
    Args:
        positions (numpy.ndarray): Array of shape (N, 3) containing 3D positions.
        velocities (numpy.ndarray): Array of shape (N, 3) containing 3D velocities.
        viewing_direction (numpy.ndarray): 1D array of shape (3,) specifying the viewing direction.

    Returns:
        tuple: 2D positions and velocities on the projected plane.
    """
    # Normalize the viewing direction
    viewing_direction = viewing_direction / np.linalg.norm(viewing_direction)
    
    # Find two orthogonal vectors in the plane perpendicular to the viewing direction
    if np.allclose(viewing_direction, [0, 0, 1]):
        orthogonal1 = np.array([1, 0, 0]).astype(np.float64)

    else:
        orthogonal1 = np.cross(viewing_direction, [0, 0, 1]).astype(np.float64)

    orthogonal1 /= np.linalg.norm(orthogonal1)
    orthogonal2 = np.cross(viewing_direction, orthogonal1)
    orthogonal2 /= np.linalg.norm(orthogonal2)
    
    # Project positions onto the 2D plane
    x_2d = np.dot(positions, orthogonal1)
    y_2d = np.dot(positions, orthogonal2)
    positions_2d = np.vstack((x_2d, y_2d)).T
    
    # Project velocities onto the 2D plane
    los_velocity = np.dot(velocities, viewing_direction)
    
    return positions_2d, los_velocity

def run_dsp(positions_2d, velocity, in_groups, n_sims=1000, Plim_P = 10, Ng_jump=1, Ng_max=None, ddof=1, cluster_name = None):

    Ng_max = int(np.sqrt(len(velocity))) if Ng_max is None else Ng_max

    dsp_results = milaDS.DSp_groups(Xcoor=positions_2d[:, 0], Ycoor=positions_2d[:, 1], Vlos=velocity, Zclus=0, nsims=n_sims, Plim_P = Plim_P, Ng_jump=Ng_jump, Ng_max=Ng_max, ddof=ddof, cluster_name = cluster_name)

    dsp_g = np.zeros(positions_2d.shape[0])
    tng_g = np.zeros(positions_2d.shape[0])

    groups = np.unique(in_groups)

    for group in groups:

        group_ind = np.where(group == in_groups)[0]
            
        if len(group_ind) > 1 and len(group_ind) < int(np.sqrt(len(velocity))):
            ### 1 represents a subhalo that belongs to a substructure
            tng_g[group_ind] = 1
    
        else:
            ### 2 represents a subhalo that does not belong to a substructure
            tng_g[group_ind] = 2

    sub_grnu, sub_count = np.unique(dsp_results[1][:, 8], return_counts=True)
    sub_grnu_arr = dsp_results[1][:, 8]

    for i in range(len(sub_grnu)):

        group_dsp_arr = np.where(sub_grnu_arr == sub_grnu[i])[0]

        if sub_count[i] > 1 and sub_count[i] < int(np.sqrt(len(velocity))):
            dsp_g[group_dsp_arr] = 1
    
        else:
            dsp_g[group_dsp_arr] = 2

    try:
        NDSp = len(np.where(dsp_g == 1)[0])
        Nreal = len(np.where(tng_g == 1)[0])
        NDSp_real = len(np.where((tng_g == 1) & (dsp_g == 1))[0])
        C = NDSp_real / Nreal
        P = NDSp_real / NDSp
    except:
        C = np.nan
        P = np.nan

    return dsp_results, C, P

def bootstrap_complteeness_purity(mc_in, pos_in, vel_in, in_groups, n_sims=1000, cluster_name = None):

    ### TNG subgroups

    C_arr, P_arr = np.zeros(mc_in), np.zeros(mc_in)
    tng_g = np.zeros(pos_in.shape[0])

    groups = np.unique(in_groups)

    for group in groups:

        group_ind = np.where(group == in_groups)[0]
            
        if len(group_ind) > 1 and len(group_ind) < int(np.sqrt(len(velocity))):
            ### 1 represents a subhalo that belongs to a substructure
            tng_g[group_ind] = 1
    
        else:
            ### 2 represents a subhalo that does not belong to a substructure
            tng_g[group_ind] = 2
    
    ### Run DS+ j times

    for j in range(mc_in):

        dsp_g = np.zeros(pos_in.shape[0])

        r_bootstrap = np.random.choice(a = vel_in)
        bool_bootstrap = vel_in != r_bootstrap

        mc_run = milaDS.DSp_groups(Xcoor=pos_in[bool_bootstrap, 0], Ycoor=pos_in[bool_bootstrap, 1], Vlos=vel_in[bool_bootstrap], Zclus=0, nsims=n_sims, Plim_P = 50, Ng_jump=1, cluster_name = cluster_name + str(j))

        ### DS+ subgroups

        sub_grnu, sub_count = np.unique(mc_run[1][:, 8], return_counts=True)
        sub_grnu_arr = mc_run[1][:, 8]

        for i in range(len(sub_grnu)):

            group_dsp_arr = np.where(sub_grnu_arr == sub_grnu[i])[0]

            if sub_count[i] > 1 and sub_count[i] < 30:
                dsp_g[group_dsp_arr] = 1
    
            else:
                dsp_g[group_dsp_arr] = 2


        NDSp = len(np.where(dsp_g == 1)[0])
        Nreal = len(np.where(tng_g == 1)[0])
        NDSp_real = len(np.where((tng_g == 1) & (dsp_g == 1))[0])
        C_arr[j] = NDSp_real / Nreal
        P_arr[j] = NDSp_real / NDSp
    
    return np.nanstd(C_arr), np.nanstd(P_arr)


def dsp_group_finder(dsp_output):

    sub_grnu, sub_count = np.unique(dsp_output[1][:, 8], return_counts=True)
    sub_grnu_arr = dsp_output[1][:, 8]

    return sub_grnu_arr


def Mass_Munari(groups, los_velcocity_arr, h = 0.6774, A_1D_sub = 1199, alpha_sub = 0.365, A_1D_halo = 1095, alpha_halo = 0.336):

    group_num = np.unique(groups)
    subhalo_mass = np.zeros(len(groups))

    for i in range(len(group_num)):

        group_i = np.where(groups == group_num[i])[0]

        if len(group_i) > 1 and len(group_i) < 30:

            mean_los_velocity_i = np.mean(los_velcocity_arr[group_i])
            los_velocity_disp_i = np.sqrt((1 / (len(group_i) - 1)) * np.sum((los_velcocity_arr[group_i] - mean_los_velocity_i)**2))

            subhalo_mass[group_i] = (los_velocity_disp_i / A_1D_sub)**(1 / alpha_sub) * h 
    
    mean_cluster_velocity = np.mean(los_velcocity_arr)
    los_velocity_disp = np.sqrt((1 / (len(groups) - 1)) * np.sum((los_velcocity_arr - mean_cluster_velocity)**2))
    
    halo_mass = (los_velocity_disp / A_1D_halo)**(1 / alpha_halo) * h 

    return halo_mass, subhalo_mass

def virial_mass_velocity(position_2d, los_velocity, groups):

    G = 6.6743 * 10**-11
    pos_2d = position_2d * 3.086 * 10**19  # from kpc to m
    vel_los = los_velocity * 10**3 # km/s to m/s

    ### Cluster mass

    N = len(pos_2d)
    r_ij = np.zeros((N, N))

    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt(np.sum((pos_2d[i, :] - pos_2d[j, :])**2))
            r_ij[i, j] = dist
            r_ij[j, i] = dist 

    i_upper, j_upper = np.triu_indices(N, k=1)
    sum_inverse_r = np.sum(1.0 / r_ij[i_upper, j_upper])

    R_harm = ( (1 / (N * (N - 1))) * sum_inverse_r)**(-1)
    mean_cluster_velocity = np.mean(vel_los)
    vel_los_disp = np.sqrt((1 / (N - 1)) * np.sum((vel_los - mean_cluster_velocity)**2))
    M_cluster = (3 * vel_los_disp**2 * R_harm) / G


    ### Substructure masses

    group_nums = np.unique(groups)
    sub_masses = []
    sub_vel_disp = []

    for k in range(len(group_nums)):

        group_i_ind = np.where(groups == group_nums[k])[0]

        if len(group_i_ind) > 1 and len(group_i_ind) < 30:

            N = len(group_i_ind)
            r_ij = np.zeros((N, N))

            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt(np.sum((pos_2d[group_i_ind][i, :] - pos_2d[group_i_ind][j, :])**2))
                    r_ij[i, j] = dist
                    r_ij[j, i] = dist 

            i_upper, j_upper = np.triu_indices(N, k=1)
            sum_inverse_r = np.sum(1.0 / r_ij[i_upper, j_upper])
            R_harm = ( (1 / (N * (N - 1))) * sum_inverse_r)**(-1)

            mean_sub_velocity = np.mean(vel_los[group_i_ind])
            vel_los_disp = np.sqrt((1 / (len(group_i_ind) - 1)) * np.sum((vel_los[group_i_ind] - mean_sub_velocity)**2))
            sub_vel_disp.append(vel_los_disp)

            M_sub = (3 * vel_los_disp**2 * R_harm) / G
            sub_masses.append(M_sub)
    
    sub_masses = np.array(sub_masses) / (1.989 * 10**30) ## Convert from kg to solar masses
    M_cluster = M_cluster / (1.989 * 10**30) ## Convert from kg to solar masses
    sub_vel_disp = np.array(sub_vel_disp) * 10**-3

    return M_cluster, sub_masses, sub_vel_disp

def DSP_Virial_analysis(cluster_number, proj_vector, dsp_sims, bootstrap_mc, bootstrap = False):

    ### Grab TNG data
    pos, vel, group, sub_masses_TNG, h, halo_mass_TNG = get_cluster_props(cluster_number)

    sub_mass_list, sub_veldisp_list, dsp_result_list = [], [], []

    for i in range(proj_vector.shape[0]):

        ### Project data to be observation-like
        pos_2d, vel_los = project_3d_to_2d(pos, vel, viewing_direction=proj_vector[i])

        ### Run DS+ and get back purity and completeness 
        dsp_results, C, P = run_dsp(pos_2d, vel_los, group, n_sims=dsp_sims, Plim_P = 50, Ng_jump=1, cluster_name = str(proj_vector[i]))

        ### Bootstrapping on purity and completeness
        if bootstrap:
            C_err, P_err = bootstrap_completeness_purity(mc_in = bootstrap_mc, pos_in = pos_2d, vel_in = vel_los, in_groups = group, 
                                                         n_sims=dsp_sims, cluster_name = str(proj_vector[i]));
        else: 
            C_err, P_err = np.nan, np.nan

        ### Find the DS+ Groups
        dsp_groups = dsp_group_finder(dsp_output = dsp_results)

        ### Find Virial Masses
        halo_mass_Virial, sub_masses_Virial, sub_veldisp_Virial = virial_mass_velocity(position_2d = pos_2d, los_velocity = vel_los, groups = dsp_groups)

        if i == 0:

            df = pd.DataFrame({
                "Cluster Index": [cluster_number],
                "Projection x-Direction": [proj_vector[i,0]],
                "Projection y-Direction": [proj_vector[i,1]],
                "Projection z-Direction": [proj_vector[i,2]],
                "Completeness": [C],
                "Completeness Uncertainty": [C_err],
                "Purity": [P],
                "Purity Uncertainty": [P_err],
                "Virial Halo Mass": [halo_mass_Virial]
            })
        
        else: 

            df_new = pd.DataFrame({
                "Cluster Index": [cluster_number],
                "Projection x-Direction": [proj_vector[i,0]],
                "Projection y-Direction": [proj_vector[i,1]],
                "Projection z-Direction": [proj_vector[i,2]],
                "Completeness": [C],
                "Completeness Uncertainty": [C_err],
                "Purity": [P],
                "Purity Uncertainty": [P_err],
                "Virial Halo Mass": [halo_mass_Virial]
            })

            df = pd.concat([df, df_new], ignore_index=True)

        sub_mass_list.append(sub_masses_Virial), sub_veldisp_list.append(sub_veldisp_Virial), dsp_result_list.append(dsp_results)

    return dsp_result_list, sub_mass_list, sub_veldisp_list, df


def compute_covariance_3d(positions):
    """
    Compute the 3x3 covariance matrix for a set of 3D points.
    
    Args:
        positions (numpy.ndarray): Array of shape (N, 3).
    
    Returns:
        numpy.ndarray: 3x3 covariance matrix.
    """
    # Center the positions
    mean_pos = np.mean(positions, axis=0)
    centered = positions - mean_pos
    
    # Covariance
    cov_3d = np.dot(centered.T, centered) / positions.shape[0]
    return cov_3d

def shape_index_3d(positions):
    """
    Compute a simple 3D shape (sphericity) index for the distribution.
    For example, the ratio of the smallest principal axis to the largest (c/a).
    
    Args:
        positions (numpy.ndarray): Array of shape (N, 3).
    
    Returns:
        float: c/a where c = sqrt(lambda_min) and a = sqrt(lambda_max).
    """
    cov_3d = compute_covariance_3d(positions)
    eigenvals, _ = np.linalg.eig(cov_3d)
    # Sort eigenvalues: largest to smallest
    sorted_vals = np.sort(eigenvals)[::-1]
    a = np.sqrt(sorted_vals[0])  # largest axis
    b = np.sqrt(sorted_vals[1])  # middle axis
    c = np.sqrt(sorted_vals[2])  # smallest axis

    T = (a**2 - b**2) / (a**2 - c**2)
    sphericity = c / a

    return sphericity, T

def compute_covariance_2d(positions, velocities, proj_arr):
    """
    Compute the 2x2 covariance matrix for a set of 2D points.
    
    Args:
        positions_2d (numpy.ndarray): Array of shape (N, 2).
    
    Returns:
        numpy.ndarray: 2x2 covariance matrix.
    """

    positions_2d, los_vel = project_3d_to_2d(positions, velocities, proj_arr)

    mean_pos = np.mean(positions_2d, axis=0)
    centered = positions_2d - mean_pos
    
    cov_2d = np.dot(centered.T, centered) / positions_2d.shape[0]
    return cov_2d

def shape_index_2d(positions, velocities, proj_arr):
    """
    Compute a simple 2D shape index for the distribution.
    For example, the ratio of the smaller principal axis to the larger (sqrt(lambda_min / lambda_max)).
    
    Args:
        positions_2d (numpy.ndarray): Array of shape (N, 2).
    
    Returns:
        float: The ratio (minor / major) of the distribution in 2D.
    """
    cov_2d = compute_covariance_2d(positions, velocities, proj_arr)
    eigenvals, _ = np.linalg.eig(cov_2d)
    # Sort eigenvalues: largest to smallest
    sorted_vals = np.sort(eigenvals)[::-1]
    major = np.sqrt(sorted_vals[0])
    minor = np.sqrt(sorted_vals[1])
    
    return minor / major

def compare_3d_2d_shape(positions_3d, velocities_3d, viewing_direction):
    """
    Compare the 3D shape index with the 2D shape index from a projection.
    
    Args:
        positions_3d (numpy.ndarray): Shape (N, 3) for 3D positions.
        velocities_3d (numpy.ndarray): Shape (N, 3) for 3D velocities (used for projection).
        viewing_direction (numpy.ndarray): 1D array of shape (3,) specifying the viewing direction.
    
    Returns:
        tuple: (shape_3d, shape_2d, difference), where 'difference' = shape_2d - shape_3d
    """
    # 3D shape index
    shape_3d, T = shape_index_3d(positions_3d)
    
    # 2D shape index
    shape_2d = shape_index_2d(positions_3d, velocities_3d, viewing_direction)
    
    # Compare 
    difference = shape_2d - shape_3d
    return shape_3d, shape_2d, difference, T

def rotate_to_viewing_frame(positions: np.ndarray,
                            velocities: np.ndarray,
                            viewing_direction: np.ndarray):
    """
    Rotate 3-D positions and velocities into a frame whose +z-axis is
    the given viewing_direction.

    Parameters
    ----------
    positions : (N, 3) array_like
        Original positions.
    velocities : (N, 3) array_like
        Original velocities.
    viewing_direction : (3,) array_like
        Non-zero vector that defines the new coordinate system.

    Returns
    -------
    pos_rot : (N, 3) ndarray
        Positions in the new frame.
    vel_rot : (N, 3) ndarray
        Velocities in the new frame.
    """

    # -- 1. Normalise viewing_direction (w) -----------------------------------
    w = np.asarray(viewing_direction, dtype=np.float64)
    if w.shape != (3,) or not np.isfinite(w).all():
        raise ValueError("viewing_direction must be a finite 3-vector.")
    w /= np.linalg.norm(w)            # unit vector along new z-axis

    # -- 2. Build an orthonormal basis (u, v, w) ------------------------------
    # Choose a helper vector 'a' that is guaranteed not to be parallel to w:
    a = np.zeros(3)
    a[np.argmin(np.abs(w))] = 1.0     # axis with smallest |component|
    u = np.cross(w, a)
    u /= np.linalg.norm(u)            # first axis in the sky-plane
    v = np.cross(w, u)                # second axis, automatically unit

    # Rotation matrix: columns are the basis vectors expressed in the old frame
    R = np.column_stack((u, v, w))    # shape (3, 3), orthonormal

    # -- 3. Rotate positions and velocities -----------------------------------
    pos_rot = positions  @ R
    vel_rot = velocities @ R

    return pos_rot, vel_rot

### Particle Gridding data for coherence analysis

def _prep_grid_shape(ngrid):
    if isinstance(ngrid, int):
        Nx = Ny = Nz = int(ngrid)
    else:
        Nx, Ny, Nz = map(int, ngrid)
    if Nx <= 0 or Ny <= 0 or Nz <= 0:
        raise ValueError("ngrid must be positive")
    return Nx, Ny, Nz

def _filter_inbounds(positions, L):
    x, y, z = positions[:,0], positions[:,1], positions[:,2]
    m = (x >= -L) & (x < L) & (y >= -L) & (y < L) & (z >= -L) & (z < L)
    return m

def _cic_indices_weights_nonperiodic(p_chunk, L, N):
    Nx, Ny, Nz = N
    dx, dy, dz = (2.0*L)/Nx, (2.0*L)/Ny, (2.0*L)/Nz
    gx = (p_chunk[:,0] + L) / dx
    gy = (p_chunk[:,1] + L) / dy
    gz = (p_chunk[:,2] + L) / dz

    ix0 = np.floor(gx).astype(np.int64)
    iy0 = np.floor(gy).astype(np.int64)
    iz0 = np.floor(gz).astype(np.int64)

    fx = gx - ix0
    fy = gy - iy0
    fz = gz - iz0

    ix1 = ix0 + 1
    iy1 = iy0 + 1
    iz1 = iz0 + 1

    return (iz0, iz1), (iy0, iy1), (ix0, ix1), (fz, fy, fx), (Nz, Ny, Nx)

def _masked_add(flat, idx, w, Ntot):
    m = (idx >= 0) & (idx < Ntot)
    if np.any(m):
        np.add.at(flat, idx[m], w[m])

def _accumulate_cic_scalar_nonperiodic(grid, idxz, idxy, idxx, fzfyfx, w):
    Nz, Ny, Nx = grid.shape
    iz0, iz1 = idxz; iy0, iy1 = idxy; ix0, ix1 = idxx
    fz, fy, fx = fzfyfx

    wz0 = 1.0 - fz; wy0 = 1.0 - fy; wx0 = 1.0 - fx
    wz1 = fz;       wy1 = fy;       wx1 = fx

    w000 = wz0 * wy0 * wx0 * w
    w001 = wz0 * wy0 * wx1 * w
    w010 = wz0 * wy1 * wx0 * w
    w011 = wz0 * wy1 * wx1 * w
    w100 = wz1 * wy0 * wx0 * w
    w101 = wz1 * wy0 * wx1 * w
    w110 = wz1 * wy1 * wx0 * w
    w111 = wz1 * wy1 * wx1 * w

    flat = grid.ravel()
    stride_y = Nx
    stride_z = Ny * Nx
    Ntot = flat.size

    idx000 = iz0 * stride_z + iy0 * stride_y + ix0
    idx001 = iz0 * stride_z + iy0 * stride_y + ix1
    idx010 = iz0 * stride_z + iy1 * stride_y + ix0
    idx011 = iz0 * stride_z + iy1 * stride_y + ix1
    idx100 = iz1 * stride_z + iy0 * stride_y + ix0
    idx101 = iz1 * stride_z + iy0 * stride_y + ix1
    idx110 = iz1 * stride_z + iy1 * stride_y + ix0
    idx111 = iz1 * stride_z + iy1 * stride_y + ix1

    mz0 = (iz0 >= 0) & (iz0 < Nz); mz1 = (iz1 >= 0) & (iz1 < Nz)
    my0 = (iy0 >= 0) & (iy0 < Ny); my1 = (iy1 >= 0) & (iy1 < Ny)
    mx0 = (ix0 >= 0) & (ix0 < Nx); mx1 = (ix1 >= 0) & (ix1 < Nx)

    m000 = mz0 & my0 & mx0
    m001 = mz0 & my0 & mx1
    m010 = mz0 & my1 & mx0
    m011 = mz0 & my1 & mx1
    m100 = mz1 & my0 & mx0
    m101 = mz1 & my0 & mx1
    m110 = mz1 & my1 & mx0
    m111 = mz1 & my1 & mx1

    _masked_add(flat, idx000[m000], w000[m000], Ntot)
    _masked_add(flat, idx001[m001], w001[m001], Ntot)
    _masked_add(flat, idx010[m010], w010[m010], Ntot)
    _masked_add(flat, idx011[m011], w011[m011], Ntot)
    _masked_add(flat, idx100[m100], w100[m100], Ntot)
    _masked_add(flat, idx101[m101], w101[m101], Ntot)
    _masked_add(flat, idx110[m110], w110[m110], Ntot)
    _masked_add(flat, idx111[m111], w111[m111], Ntot)

def deposit_cic_scalar(positions, L, ngrid, weights=None, chunksize=1_000_000, dtype=np.float64):
    Nx, Ny, Nz = _prep_grid_shape(ngrid)
    grid = np.zeros((Nz, Ny, Nx), dtype=dtype)

    m = _filter_inbounds(positions, L)
    p = positions[m]
    if weights is None:
        w = np.ones(p.shape[0], dtype=dtype)
    else:
        w_full = np.asarray(weights, dtype=dtype)
        if w_full.shape[0] != positions.shape[0]:
            raise ValueError("weights must match positions length")
        w = w_full[m]

    N = p.shape[0]
    start = 0
    while start < N:
        end = min(start + chunksize, N)
        pc = p[start:end]
        wc = w[start:end]
        idxz, idxy, idxx, fzfyfx, _ = _cic_indices_weights_nonperiodic(pc, L, (Nx, Ny, Nz))
        _accumulate_cic_scalar_nonperiodic(grid, idxz, idxy, idxx, fzfyfx, wc)
        start = end
    return grid

def deposit_cic_velocity(positions, velocities, L, ngrid, mass=None, chunksize=1_000_000, dtype=np.float64):
    Nx, Ny, Nz = _prep_grid_shape(ngrid)
    momx = np.zeros((Nz, Ny, Nx), dtype=dtype)
    momy = np.zeros((Nz, Ny, Nx), dtype=dtype)
    momz = np.zeros((Nz, Ny, Nx), dtype=dtype)
    mgrid = np.zeros((Nz, Ny, Nx), dtype=dtype)

    m = _filter_inbounds(positions, L)
    p = positions[m]
    v = velocities[m]
    if v.shape != (p.shape[0], 3):
        raise ValueError("velocities must be (N,3) matching filtered positions")
    if mass is None:
        w = np.ones(p.shape[0], dtype=dtype)
    else:
        mass = np.asarray(mass, dtype=dtype)
        if mass.shape[0] != positions.shape[0]:
            raise ValueError("mass must match positions length")
        w = mass[m]

    N = p.shape[0]
    start = 0
    while start < N:
        end = min(start + chunksize, N)
        pc = p[start:end]
        vc = v[start:end].astype(dtype, copy=False)
        mc = w[start:end]
        idxz, idxy, idxx, fzfyfx, _ = _cic_indices_weights_nonperiodic(pc, L, (Nx, Ny, Nz))

        _accumulate_cic_scalar_nonperiodic(mgrid, idxz, idxy, idxx, fzfyfx, mc)
        _accumulate_cic_scalar_nonperiodic(momx, idxz, idxy, idxx, fzfyfx, mc * vc[:,0])
        _accumulate_cic_scalar_nonperiodic(momy, idxz, idxy, idxx, fzfyfx, mc * vc[:,1])
        _accumulate_cic_scalar_nonperiodic(momz, idxz, idxy, idxx, fzfyfx, mc * vc[:,2])

        start = end

    with np.errstate(invalid='ignore', divide='ignore'):
        vx = np.where(mgrid > 0, momx / mgrid, 0.0)
        vy = np.where(mgrid > 0, momy / mgrid, 0.0)
        vz = np.where(mgrid > 0, momz / mgrid, 0.0)

    return vx, vy, vz, mgrid

def voxel_size_from_L(L, ngrid):
    Nx, Ny, Nz = _prep_grid_shape(ngrid)
    dx, dy, dz = (2.0*L)/Nx, (2.0*L)/Ny, (2.0*L)/Nz
    # coherence.compute_coherence_3d expects (dz, dy, dx)
    return (dz, dy, dx)