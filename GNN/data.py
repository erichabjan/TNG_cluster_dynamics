dirc_path = '/home/habjan.e/'

import sys
sys.path.append(dirc_path + "TNG/TNG_cluster_dynamics")
import TNG_DA
import numpy as np 
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from astropy.io import fits

dataset_size = 10**5
num_clusters = 10**2

num_proj_per_cluster = int(dataset_size / num_clusters)
cluster_inds = np.arange(num_clusters)

rows = []

id_val = 0

for cluster_idx in cluster_inds:
    
    pos, vel, group, sub_masses_TNG, h, halo_mass_TNG = TNG_DA.get_cluster_props(cluster_idx)

    print(f'Reprojecting Cluster {cluster_idx}')

    for j in range(num_proj_per_cluster):
        
        proj_vec = np.random.uniform(-1, 1, (3,))

        ro_pos, ro_vel = TNG_DA.rotate_to_viewing_frame(pos, vel, proj_vec)

        rows.append((id_val, cluster_idx, proj_vec, halo_mass_TNG, ro_pos[:, 0], ro_pos[:, 1], ro_pos[:, 2], ro_vel[:, 0], ro_vel[:, 1], ro_vel[:, 2], sub_masses_TNG))

        id_val += 1
    
    print(f'Finished reprojecting Cluster {cluster_idx}')

dtype = np.dtype([('ID','<i8'), ('Cluster Index','<f4'), ('Projection Vector',object), ('Cluster Mass [solar masses]','<f4'), ('x position [kpc]',object), ('y position [kpc]',object), ('z position [kpc]',object), ('x velocity [km / s]',object), ('y velocity [km / s]',object), ('z velocity [km / s]',object), ('subhalo masses [solar masses]',object)])
table = np.asarray(rows, dtype=dtype)

coldefs = fits.ColDefs([
    fits.Column(name='ID', format='K', array=table['ID']),
    fits.Column(name='Cluster Index', format='E', array=table['Cluster Index']),
    fits.Column(name='Projection Vector', format='PJ()', array=table['Projection Vector']),
    fits.Column(name='Cluster Mass [solar masses]', format='E', array=table['Cluster Mass [solar masses]']),
    fits.Column(name='x position [kpc]', format='PJ()', array=table['x position [kpc]']),
    fits.Column(name='y position [kpc]', format='PJ()', array=table['y position [kpc]']),
    fits.Column(name='z position [kpc]', format='PJ()', array=table['z position [kpc]']),
    fits.Column(name='x velocity [km / s]', format='PJ()', array=table['x velocity [km / s]']),
    fits.Column(name='y velocity [km / s]', format='PJ()', array=table['y velocity [km / s]']),
    fits.Column(name='z velocity [km / s]', format='PJ()', array=table['z velocity [km / s]']),
    fits.Column(name='subhalo masses [solar masses]', format='PJ()', array=table['subhalo masses [solar masses]']),
])

hdu = fits.BinTableHDU.from_columns(coldefs)

test_size = int(num_clusters * 0.1)
subset_test = np.random.choice(cluster_inds, size = test_size, replace=False)

train_data = hdu.data[~np.isin(np.array(hdu.data['Cluster Index']), subset_test)]
test_data = hdu.data[np.isin(np.array(hdu.data['Cluster Index']), subset_test)]

test_hdu  = fits.BinTableHDU(data=test_data,  header=hdu.header)
train_hdu = fits.BinTableHDU(data=train_data, header=hdu.header)

test_path = '/home/habjan.e/TNG/Data/GNN_SBI_data/GNN_data_test.fits'
train_path = '/home/habjan.e/TNG/Data/GNN_SBI_data/GNN_data_train.fits'

train_hdu.writeto(train_path, overwrite=True)
test_hdu.writeto(test_path, overwrite=True)