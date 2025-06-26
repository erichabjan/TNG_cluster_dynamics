from astropy.table import Table, vstack
import numpy as np 

import os
from pathlib import Path
import pickle

import numpy as np
import jax.numpy as jnp
import jraph

### Import FITS tables
base_path = '/home/habjan.e/TNG/Data/GNN_SBI_data/'
train = Table.read(base_path + 'GNN_data_train.fits')
test = Table.read(base_path + 'GNN_data_test.fits')

### Find the cluster with the maximum number of galaxies
data = vstack([train, test])
x_pos = np.array(data['x position'])
lens = np.array([len(x) for x in x_pos])
max_len = np.max(lens)

### Set data directory, batch size, and maximum number of nodes (galaxies)
DATA_DIR   = "/home/habjan.e/TNG/Data/GNN_SBI_data/graph_data/"
BATCH_SIZE = 32                 
MAX_NODES  = 500

print(f'The maximum value of galaxies in a cluster is {max_len} and the max nodes is {MAX_NODES}')

### Function to load in graph data
def load_raw_arrays(fits_table):
    """
    Extracts data from a FITS table
    """

    num_graphs = len(fits_table)      
    inputs_list  = []
    targets_list = []
    for i in range(num_graphs):

        # (x, y, v_z)
        inputs  = (fits_table[i]['x position'], fits_table[i]['y position'], fits_table[i]['z velocity'])
        # (z, v_x, v_y)
        targets = (fits_table[i]['z position'], fits_table[i]['x velocity'], fits_table[i]['y velocity'])

        inputs_list.append(np.stack(inputs,  axis=-1))
        targets_list.append(np.stack(targets, axis=-1))
    
    return inputs_list, targets_list


### This function makes graph data using only node information
def make_graph(nodes_np: np.ndarray) -> jraph.GraphsTuple:
    """Convert (N_i, 3) numpy array -> node-only GraphsTuple."""

    nodes = jnp.asarray(nodes_np)
    N_i   = nodes.shape[0]
    empty = jnp.zeros((0,), dtype=jnp.int32)

    return jraph.GraphsTuple(
        nodes=nodes,             
        edges=None,
        senders=empty,
        receivers=empty,
        n_node=jnp.array([N_i], dtype=jnp.int32),
        n_edge=jnp.array([0],  dtype=jnp.int32),
        globals=None
    )

### Function to pad a single graph
def pad_single(g, max_nodes):

    return jraph.pad_with_graphs(g,
                                 n_node=max_nodes,
                                 n_edge=0)


### This function pads the graph data so that they have the same input size
def pad_batch(graphs, max_nodes):

    padded_list = [pad_single(g, max_nodes) for g in graphs]

    return jraph.batch(padded_list)


### This function pads the target lists
def pad_targets(targets_list, batch_size: int, max_nodes: int):
    """Concatenate & zero-pad target arrays; build node mask."""
    concat_targets = jnp.concatenate([jnp.asarray(t) for t in targets_list],
                                     axis=0)        # (Σ N_i, 3)
    pad_len = max_nodes * batch_size - concat_targets.shape[0]
    padded_targets = jnp.pad(concat_targets,
                             ((0, pad_len), (0, 0)))   # (batch_nodes, 3)
    return padded_targets

### This function saves many batched pickle files that will be used during training and evaluation
def main(fits_file, prefix):
    inputs_list, targets_list = load_raw_arrays(fits_file)
    #DATA_DIR.mkdir(parents=True, exist_ok=True)

    batch_graphs   = []
    batch_targets  = []
    batch_counter  = 0
    file_counter   = 0

    for inp, tgt in zip(inputs_list, targets_list):
        batch_graphs.append(make_graph(inp))
        batch_targets.append(tgt)
        batch_counter += 1

        if batch_counter == BATCH_SIZE:
            # --- pad & save ------------------------------------------
            padded_graph   = pad_batch(batch_graphs, MAX_NODES)
            padded_targets = pad_targets(batch_targets,BATCH_SIZE, MAX_NODES)
            node_mask = jraph.get_node_padding_mask(padded_graph)

            out_path = DATA_DIR + prefix + f"_batch_{file_counter:04d}.pkl"

            with open(out_path, 'wb') as fh:
                pickle.dump(
                    dict(graph=padded_graph,
                         targets=padded_targets,
                         node_mask=node_mask),
                    fh
                )
            print(f"wrote {out_path}")
            # reset batch
            batch_graphs.clear()
            batch_targets.clear()
            batch_counter = 0
            file_counter += 1

    # If any graphs left that didn’t make a full batch, save them too
    if len(batch_graphs) > 0:

        padded_graph   = pad_batch(batch_graphs, MAX_NODES)
        padded_targets = pad_targets(batch_targets, len(batch_graphs), MAX_NODES)

        node_mask = jraph.get_node_padding_mask(padded_graph)

        out_path = DATA_DIR + prefix + f"_batch_{file_counter:04d}.pkl"

        with open(out_path, 'wb') as fh:
            pickle.dump(
                dict(graph=padded_graph,
                     targets=padded_targets,
                     node_mask=node_mask),
                fh
            )
        print(f"wrote {out_path} (last, smaller batch)")


if __name__ == "__main__":
    main(train, 'train')
    main(test, 'test')