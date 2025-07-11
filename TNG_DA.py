dirc_path = '/home/habjan.e/'

import sys
sys.path.append(dirc_path + 'TNG/Codes/DS+/MilaDS')
import milaDS

sys.path.append(dirc_path + 'TNG/Codes')
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
    #bool_arr = np.array(Crit200 * 10**10 / simdata['hubble']) > 10**quan_val
    #bool_ind = np.where(bool_arr)[0]
    Group_num = iapi.getSubhaloField('SubhaloGrNr', simulation=sim, fileName=TNG_data_path+'TNG_data/'+sim+'_SubhaloGrNr', rewriteFile=0) # Import array that identifies the halo each subhalo belongs to 
    h = simdata['hubble']
    halo_mass = Crit200[cluster_ind] * 10**10 / h

    if halo_mass > 10**quan_val:
        display(Markdown(f"Cluster {cluster_ind} has a $M_{{200}}$ greater than {np.round(quan_val)} $\\log(M_\\odot)$"))
    else:
        display(Markdown(f"Cluster {cluster_ind} has a $M_{{200}}$ less than {np.round(quan_val)} $\\log(M_\\odot)$"))

    ### Pick a galaxy cluster
    #cluster_TNG = bool_ind[cluster_ind]
    sub_ind = np.where(Group_num == cluster_ind)[0]

    ### Find the center of the galaxy cluster
    pos_comoving = halo_center[cluster_ind, :] ## position in units c * kpc / h
    cm_pos = pos_comoving / simdata['hubble']

    ### Import Subhalo Positons, Velocities, Photometrics 
    sub_comoving = iapi.getSubhaloField('SubhaloPos', simulation=sim, snapshot=99, fileName=TNG_data_path+'TNG_data/'+sim+'_SubhaloPos', rewriteFile=0)
    sub_vel = iapi.getSubhaloField('SubhaloVel', simulation=sim, snapshot=99, fileName=TNG_data_path+'TNG_data/'+sim+'_SubhaloVel', rewriteFile=0)
    sub_photo = iapi.getSubhaloField('SubhaloStellarPhotometrics', snapshot=99, simulation=sim, fileName=TNG_data_path+'TNG_data/'+sim+'_SubhaloStellarPhotometrics', rewriteFile=0)
    sub_masses = iapi.getSubhaloField('SubhaloMass', snapshot=99, simulation=sim, fileName=TNG_data_path+'TNG_data/'+sim+'_SubhaloMass', rewriteFile=0)
    sub_masses = sub_masses * 10**10 / h

    #L is length of box, halfbox is L/2
    sub_uncorrected_pos = sub_comoving / h
    L = np.max(sub_uncorrected_pos)
    halfbox = L / 2

    difpos = np.subtract(sub_uncorrected_pos, cm_pos)
    #Replace values that are affected by boundary conditions
    difpos = np.where( abs(difpos) > halfbox, abs(difpos)- L , difpos)
    distsq = np.sum(np.square(difpos),axis=1)

    ### Center the position array relative to the cluster, make arrays with cluster subhalo parameters
    cl_pos, cl_vel, cl_photo, cl_masses = difpos[sub_ind], sub_vel[sub_ind], sub_photo[sub_ind], sub_masses[sub_ind]

    ### Make a magnitude cut so that subhalos are actually galaxies, not DM halos 
    mag_cut = -18
    bright_ind = cl_photo[:, 4] < mag_cut
    pos, vel, photo, subhalo_masses = cl_pos[bright_ind], cl_vel[bright_ind], cl_photo[bright_ind], cl_masses[bright_ind]

    subhalos = sub_ind[bright_ind]    ### This gives the index of each subhalo in the cluster

    ### This for loop extracts the merger tree history information for each subhalo

    subhalo_dict = {}

    for sub in subhalos:

        try: 

            subTreeFile = gettree(99, sub)    ### This downloads the merger tree information 
            subTree = h5py.File(subTreeFile,'r')    ### This reads the merger tree info 
            subTree['SubhaloGrNr'][:]       ### Group number with cosmic time   
            grnum_max = np.max(np.where(subTree['SubhaloGrNr'][:] == cluster_ind)[0])
            grnum_max += 1
            subhalo_dict[sub] = subTree['SubhaloGrNr'][:grnum_max]

        except:

            subhalo_dict[sub] = np.nan 
            #print(f'{sub} does not have merger tree information')

    ### This for loop groups the subhalos into subgroups/substructures

    groups = {}

    for ind, merger in subhalo_dict.items():

        if np.any(np.isnan(merger)):
            continue
    
        merger_tuple = tuple(merger)

        halo_ind = np.where(subhalos == ind)[0][0]
    
        if merger_tuple in groups:
            groups[merger_tuple].append(halo_ind)
        else:
            groups[merger_tuple] = [halo_ind]
    
    return pos, vel, groups, subhalo_masses, h, halo_mass

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
    cm_pos = pos_comoving / h

    sub_uncorrected_pos = coordinates / h
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

    for group in in_groups:
            
        if len(in_groups[group]) > 1 and len(in_groups[group]) < int(np.sqrt(len(velocity))):
            tng_g[in_groups[group]] = 1
    
        else:
            tng_g[in_groups[group]] = 2

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

def bootstrap_compleness_purity(mc_in, pos_in, vel_in, in_groups, n_sims=1000, cluster_name = None):

    ### TNG subgroups

    C_arr, P_arr = np.zeros(mc_in), np.zeros(mc_in)
    tng_g = np.zeros(pos_in.shape[0])

    for group in in_groups:
            
        if len(in_groups[group]) > 1 and len(in_groups[group]) < 30:
            tng_g[in_groups[group]] = 1
    
        else:
            tng_g[in_groups[group]] = 2
    
    ### Run DS+ j times

    for j in range(mc_in):

        dsp_g = np.zeros(pos_in.shape[0])
        #proj_vector = np.random.uniform(low=-1, high=1, size=3)

        #pos2d, v_los = project_3d_to_2d(pos_in, vel_in, viewing_direction=proj_vector)

        #print(proj_vector), print(v_los)

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

def virial_mass(position_2d, los_velocity, groups):

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
    sub_masses = np.zeros(len(groups))

    for k in range(len(group_nums)):

        group_i_ind = np.where(groups == group_nums[k])[0]

        if len(group_i_ind) > 1 and len(group_i_ind) < 30:

            N = len(pos_2d[group_i_ind])
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
            M_sub = (3 * vel_los_disp**2 * R_harm) / G
            sub_masses[group_i_ind] = M_sub
    
    sub_masses /= 1.989 * 10**30 ## Convert from kg to solar masses
    M_cluster /= 1.989 * 10**30 ## Convert from kg to solar masses

    return M_cluster, sub_masses

def DSP_Virial_analysis(cluster_number, proj_vector, dsp_sims, bootstrap_mc):

    ### Grab TNG data
    pos, vel, group, sub_masses_TNG, h, halo_mass_TNG = get_cluster_props(cluster_number)

    for i in range(proj_vector.shape[0]):

        ### Project data to be observation-like
        pos_2d, vel_los = project_3d_to_2d(pos, vel, viewing_direction=proj_vector[i])

        ### Run DS+ and get back purity and completeness 
        dsp_results, C, P = run_dsp(pos_2d, vel_los, group, n_sims=dsp_sims, Plim_P = 50, Ng_jump=1, cluster_name = str(proj_vector[i]))

        ### Bootstrapping on purity and completeness
        C_err, P_err = bootstrap_compleness_purity(mc_in = bootstrap_mc, pos_in = pos_2d, vel_in = vel_los, in_groups = group, n_sims=dsp_sims, cluster_name = str(proj_vector[i]));

        ### Find the DS+ Groups
        dsp_groups = dsp_group_finder(dsp_output = dsp_results)

        ### Find Virial Masses
        halo_mass_Virial, sub_masses_Virial = virial_mass(position_2d = pos_2d, los_velocity = vel_los, groups = dsp_groups)

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

    return dsp_results, sub_masses_Virial, df


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