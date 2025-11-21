dirc_path = '/home/habjan.e/'

import sys
sys.path.append(dirc_path + "TNG/TNG_cluster_dynamics")
import TNG_DA
import numpy as np 
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from astropy.io import fits
import os
import h5py

import jraph
import jax.numpy as jnp

### Set data directory, batch size, and maximum number of nodes (galaxies)
test_path = '/projects/mccleary_group/habjan.e/TNG/Data/GNN_SBI_data/GNN_data_test.h5'
train_path = '/projects/mccleary_group/habjan.e/TNG/Data/GNN_SBI_data/GNN_data_train.h5'
BATCH_SIZE = 1               
MAX_NODES  = 700
KNN_K      = 2
MAX_EDGES  = MAX_NODES * KNN_K
LATENT_SIZE = 128
dataset_size = 10**4

if os.path.exists(train_path):
    os.remove(train_path)
if os.path.exists(test_path):
    os.remove(test_path)

cluster_inds = np.array([f"{i:03d}" for i in range(1, 101)])
simulations = np.array(['SIDM0.1b', 'SIDM0.3b', 'vdSIDMb', 'CDMb'])

num_clusters = cluster_inds.shape[0] * simulations.shape[0]
num_proj_per_cluster = int(dataset_size / num_clusters)

test_size = int(len(cluster_inds) * 0.1)
subset_test = np.random.choice(cluster_inds, size=test_size, replace=False)
subset_test_set = set(subset_test)

train_initialized = False
test_initialized = False

id_val = 0

### Functions for graph making

def make_graph(nodes_np: np.ndarray) -> jraph.GraphsTuple:
    """Convert (N, 3) numpy array -> GraphsTuple."""

    nodes = jnp.asarray(nodes_np, dtype=jnp.float32)
    N   = nodes.shape[0]

    # Pair-wise calculation of x, y, v_z
    diffs = nodes[:, None, :] - nodes[None, :, :]
    d2 = jnp.sum(diffs ** 2, axis=-1)
    d2 = d2 + jnp.eye(N) * 1e9
    knn_idx = jnp.argsort(d2, axis=1)[:, :KNN_K]

    senders = jnp.repeat(jnp.arange(N, dtype=jnp.int32), KNN_K)
    receivers = knn_idx.reshape(-1).astype(jnp.int32)

    src = nodes[senders]
    dst = nodes[receivers]
    rel = dst - src
    dist = jnp.linalg.norm(rel, axis=-1, keepdims=True)
    edges = jnp.concatenate([rel, dist], axis=-1)

    dummy_globals = jnp.zeros((1, LATENT_SIZE), dtype=jnp.float32)

    return jraph.GraphsTuple(
        nodes=nodes,             
        edges=edges,
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([N], dtype=jnp.int32),
        n_edge=jnp.array([edges.shape[0]],  dtype=jnp.int32),
        globals=dummy_globals
    )

### Function to pad a single graph
def pad_single(g, max_nodes, max_edges):

    return jraph.pad_with_graphs(g,
                                 n_node=max_nodes,
                                 n_edge=max_edges)

def explicit_mask(g, max_nodes):
    n = g.nodes.shape[0]
    return jnp.concatenate([jnp.ones(n, bool),
                            jnp.zeros(max_nodes - n, bool)])


### This function pads the graph data so that they have the same input size
def pad_batch(graphs, max_nodes, max_edges):

    padded_list = [pad_single(g, max_nodes, max_edges) for g in graphs]
    mask_list   = [explicit_mask(g, max_nodes) for g in graphs]

    return jraph.batch(padded_list), jnp.concatenate(mask_list) 


### This function pads the target lists
def pad_targets(targets_list, batch_size: int, max_nodes: int):
    """Concatenate & zero-pad target arrays; build node mask."""
    concat_targets = jnp.concatenate([jnp.asarray(t) for t in targets_list],
                                     axis=0)        # (Σ N_i, 3)
    pad_len = max_nodes * batch_size - concat_targets.shape[0]
    padded_targets = jnp.pad(concat_targets,
                             ((0, pad_len), (0, 0)))   # (batch_nodes, 3)
    return padded_targets

def write_to_hdf5(rows, file_path):
    """Write cluster data to HDF5 file."""
    with h5py.File(file_path, 'a') as f:  # 'a' = append mode (creates if doesn't exist)
        for row in rows:
            (id_val, sim, cluster_idx, proj_vec, halo_mass,
             x_ro_pos, y_ro_pos, z_ro_pos, x_ro_vel, y_ro_vel, z_ro_vel,
             x_ro_mean, y_ro_mean, z_ro_mean, vx_ro_mean, vy_ro_mean, vz_ro_mean,
             x_ro_std, y_ro_std, z_ro_std, vx_ro_std, vy_ro_std, vz_ro_std,
             masses, padded_graph, node_mask, padded_targets) = row
            
            # Create a group for this sample
            grp = f.create_group(f'{id_val:06d}')
            
            # Store metadata as attributes
            grp.attrs['id'] = id_val
            grp.attrs['simulation'] = sim
            grp.attrs['cluster_index'] = cluster_idx
            grp.attrs['cluster_mass'] = halo_mass
            
            # Store arrays as datasets
            grp.create_dataset('projection_vector', data=proj_vec)
            grp.create_dataset('x_position', data=x_ro_pos, compression='gzip')
            grp.create_dataset('y_position', data=y_ro_pos, compression='gzip')
            grp.create_dataset('z_position', data=z_ro_pos, compression='gzip')
            grp.create_dataset('x_velocity', data=x_ro_vel, compression='gzip')
            grp.create_dataset('y_velocity', data=y_ro_vel, compression='gzip')
            grp.create_dataset('z_velocity', data=z_ro_vel, compression='gzip')
            
            # Store standardization parameters
            grp.attrs['x_position_mean'] = x_ro_mean
            grp.attrs['y_position_mean'] = y_ro_mean
            grp.attrs['z_position_mean'] = z_ro_mean
            grp.attrs['x_velocity_mean'] = vx_ro_mean
            grp.attrs['y_velocity_mean'] = vy_ro_mean
            grp.attrs['z_velocity_mean'] = vz_ro_mean

            grp.attrs['x_position_std'] = x_ro_std
            grp.attrs['y_position_std'] = y_ro_std
            grp.attrs['z_position_std'] = z_ro_std
            grp.attrs['x_velocity_std'] = vx_ro_std
            grp.attrs['y_velocity_std'] = vy_ro_std
            grp.attrs['z_velocity_std'] = vz_ro_std
            
            # Store subhalo masses
            grp.create_dataset('subhalo_masses', data=masses, compression='gzip')
            
            # Store padded graph data (convert JAX arrays to numpy)
            grp.create_dataset('padded_nodes', data=np.array(padded_graph.nodes), compression='gzip')
            grp.create_dataset('node_mask', data=np.array(node_mask), compression='gzip')
            grp.create_dataset('padded_targets', data=np.array(padded_targets), compression='gzip')

            grp.create_dataset('padded_edges', data=np.array(padded_graph.edges),     compression='gzip')
            grp.create_dataset('senders',      data=np.array(padded_graph.senders),   compression='gzip')
            grp.create_dataset('receivers',    data=np.array(padded_graph.receivers), compression='gzip')

            # Store graph metadata
            grp.attrs['n_nodes'] = int(padded_graph.nodes.shape[0]) #int(padded_graph.n_node[0])
            grp.attrs['n_edges'] = int(padded_graph.edges.shape[0]) #int(padded_graph.n_edge[0])

for cluster_idx in cluster_inds:

    print(f'Making data for cluster ID {cluster_idx}')

    is_test_cluster = cluster_idx in subset_test_set

    rows = []

    for sim in simulations:

        data = np.load("/projects/mccleary_group/habjan.e/TNG/Data/" + sim + "/GrNm_" + cluster_idx + ".npz")

        coordinates = data['sub_pos'] - data['CoP']
        # c Mpc / h 
        pos = coordinates / (data['h'] * data['a'])
        # km / s
        vel = data['sub_vel']
        # solar masses
        masses = np.nansum(data['sub_massTotal'], axis = 1)

        halo_mass = data['Mfof']

        for j in range(num_proj_per_cluster):
        
            proj_vec = np.random.uniform(-1, 1, (3,))
            ro_pos, ro_vel = TNG_DA.rotate_to_viewing_frame(pos, vel, proj_vec)

            # Standardize data (data is also unitless)
            x_ro_std, y_ro_std, z_ro_std = np.nanstd(ro_pos[:, 0]), np.nanstd(ro_pos[:, 1]), np.nanstd(ro_pos[:, 2])
            vx_ro_std, vy_ro_std, vz_ro_std = np.nanstd(ro_vel[:, 0]), np.nanstd(ro_vel[:, 1]), np.nanstd(ro_vel[:, 2])

            x_ro_mean, y_ro_mean, z_ro_mean = np.nanmean(ro_pos[:, 0]), np.nanmean(ro_pos[:, 1]), np.nanmean(ro_pos[:, 2])
            vx_ro_mean, vy_ro_mean, vz_ro_mean = np.nanmean(ro_vel[:, 0]), np.nanmean(ro_vel[:, 1]), np.nanmean(ro_vel[:, 2])

            x_ro_pos, y_ro_pos, z_ro_pos = (ro_pos[:, 0] - x_ro_mean) / x_ro_std, (ro_pos[:, 1] - y_ro_mean) / y_ro_std, (ro_pos[:, 2] - z_ro_mean) / z_ro_std
            x_ro_vel, y_ro_vel, z_ro_vel = (ro_vel[:, 0] - vx_ro_mean) / vx_ro_std, (ro_vel[:, 1] - vy_ro_mean) / vy_ro_std, (ro_vel[:, 2] - vz_ro_mean) / vz_ro_std

            # (x, y, v_z)
            inputs  = np.stack((x_ro_pos, y_ro_pos, z_ro_vel),  axis=-1)
            # (z, v_x, v_y)
            targets = np.stack((z_ro_pos, x_ro_vel, y_ro_vel),  axis=-1)

            g = make_graph(inputs)
            padded_graph, node_mask = pad_batch([g], MAX_NODES, MAX_EDGES)
            padded_targets = pad_targets([targets], BATCH_SIZE, MAX_NODES)

            rows.append((id_val, str(sim), int(cluster_idx), proj_vec, halo_mass, 
                         x_ro_pos, y_ro_pos, z_ro_pos, x_ro_vel, y_ro_vel, z_ro_vel,
                         x_ro_mean, y_ro_mean, z_ro_mean, vx_ro_mean, vy_ro_mean, vz_ro_mean,
                         x_ro_std, y_ro_std, z_ro_std, vx_ro_std, vy_ro_std, vz_ro_std,
                         masses,
                         padded_graph, node_mask, padded_targets))

            id_val += 1

    if is_test_cluster:
        write_to_hdf5(rows, test_path)
    else:
        write_to_hdf5(rows, train_path)
    
    print(f'Finished reprojecting Cluster {cluster_idx}')

print("Data generation complete!")
print(f"Train data saved to: {train_path}")
print(f"Test data saved to: {test_path}")