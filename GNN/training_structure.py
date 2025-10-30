import optax 
from flax.training import train_state

import jax
import jax.numpy as jnp
from jax import random
import time
import numpy as np
from functools import partial

from typing import Iterator, Tuple, Dict
import jraph

import time
import os
import h5py

def preload_hdf5_to_memory(data_dir: str, file_in: str):
    """
    Load entire HDF5 file into memory as numpy arrays.
    
    Returns:
        Dictionary with keys: 'nodes', 'targets', 'masks', 'n_nodes', 'n_edges'
    """
    print(f"\nPreloading {file_in} into memory...")
    start = time.time()
    
    file_path = os.path.join(data_dir, file_in)
    
    with h5py.File(file_path, 'r') as f:
        sample_ids = list(f.keys())
        n_samples = len(sample_ids)
        
        print(f"Found {n_samples} samples in file")
        
        # Get shapes from first sample
        first_sample = f[sample_ids[0]]
        node_shape = first_sample['padded_nodes'].shape
        target_shape = first_sample['padded_targets'].shape
        mask_shape = first_sample['node_mask'].shape
        edge_shape   = first_sample['padded_edges'].shape
        send_shape   = first_sample['senders'].shape
        recv_shape   = first_sample['receivers'].shape 
        
        print(f"Sample shapes - Nodes: {node_shape}, Edges: {edge_shape}, Targets: {target_shape}, Mask: {mask_shape}")
        
        # Pre-allocate arrays
        all_nodes = np.zeros((n_samples, *node_shape), dtype=np.float32)
        all_targets = np.zeros((n_samples, *target_shape), dtype=np.float32)
        all_masks = np.zeros((n_samples, *mask_shape), dtype=np.float32)
        all_edges = np.zeros((n_samples, *edge_shape),   dtype=np.float32)
        all_senders = np.zeros((n_samples, *send_shape),   dtype=np.int32)
        all_receivers = np.zeros((n_samples, *recv_shape),   dtype=np.int32)
        all_n_nodes = np.zeros(n_samples, dtype=np.int32)
        all_n_edges = np.zeros(n_samples, dtype=np.int32)
        
        # Load all samples
        print("Loading samples...", end='', flush=True)
        for i, sample_id in enumerate(sample_ids):
            if i % 10000 == 0 and i > 0:
                print(f"\n  Loaded {i}/{n_samples} samples...", end='', flush=True)
            
            sample = f[sample_id]
            all_nodes[i] = sample['padded_nodes'][:]
            all_targets[i] = sample['padded_targets'][:]
            all_masks[i] = sample['node_mask'][:]
            all_edges[i]     = sample['padded_edges'][:]
            all_senders[i]   = sample['senders'][:]
            all_receivers[i] = sample['receivers'][:]
            all_n_nodes[i] = sample.attrs['n_nodes']
            all_n_edges[i] = sample.attrs['n_edges']
        
        print()
    
    elapsed = time.time() - start
    data_size_gb = (all_nodes.nbytes + all_targets.nbytes + all_masks.nbytes + all_edges.nbytes + all_senders.nbytes + all_receivers.nbytes) / 1e9
    
    print(f"✓ Loaded {n_samples} samples in {elapsed:.2f}s ({data_size_gb:.2f} GB)")
    
    return {
        'nodes': all_nodes,
        'targets': all_targets,
        'masks': all_masks,
        'edges': all_edges,
        'senders': all_senders,
        'receivers': all_receivers,
        'n_nodes': all_n_nodes,
        'n_edges': all_n_edges
    }

def data_loader(
    data_dict: Dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool = True,
    latent_size: int = 128
) -> Iterator[Tuple[jraph.GraphsTuple, jnp.ndarray, jnp.ndarray]]:
    """
    Data loader using preloaded memory.
    """
    
    n_samples = len(data_dict['nodes'])
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        
        batch_graphs = []
        for idx in batch_indices:
            graph = jraph.GraphsTuple(
                nodes = jnp.array(data_dict['nodes'][idx]),
                edges = jnp.array(data_dict['edges'][idx]),
                senders = jnp.array(data_dict['senders'][idx]),
                receivers = jnp.array(data_dict['receivers'][idx]),
                n_node = jnp.array([data_dict['n_nodes'][idx]], dtype=jnp.int32),
                n_edge = jnp.array([data_dict['n_edges'][idx]], dtype=jnp.int32),
                globals = jnp.zeros((1, latent_size), dtype=jnp.float32)
            )
            batch_graphs.append(graph)
        
        batched_graph = jraph.batch(batch_graphs)
        batched_targets = jnp.array(data_dict['targets'][batch_indices]).reshape(-1, 3)
        batched_masks = jnp.array(data_dict['masks'][batch_indices]).reshape(-1)
        
        yield batched_graph, batched_targets, batched_masks



def create_train_state(model, rng_key, learning_rate, grad_clipping, example_graph):
    
    params = model.init(rng_key, example_graph, deterministic = False)['params']
    optimizer = optax.chain(optax.clip_by_global_norm(grad_clipping), optax.adam(learning_rate))
    
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

@staticmethod
def mse_loss(params, graph, target, mask, apply_fn, training, rng=None):
    """
    Calculate weighted mean squared error loss.
    """

    deterministic = not training
    kwargs = dict(deterministic=deterministic)

    if not deterministic:
        kwargs["rngs"] = {"dropout": rng}
    
    preds_graph = apply_fn({'params': params}, graph, **kwargs)
    preds = preds_graph.nodes
    mse = ((preds - target) ** 2).sum(-1)

    mse_masked = (mse * mask).sum() / mask.sum()

    return mse_masked

@jax.jit
def train_step(state, graph, target, mask, rng_key):
    """Single training step."""
    
    def loss_fn(params):
        return mse_loss(params, graph, target, mask, state.apply_fn, training=True, rng=rng_key)

    grads = jax.grad(loss_fn)(state.params)

    return state.apply_gradients(grads=grads) 

@jax.jit
def eval_step(state, graph, target, mask):

    loss = mse_loss(state.params, graph, target, mask, state.apply_fn, training=False, rng=None)

    return loss

def train_model(train_data: Dict[str, np.ndarray], test_data: Dict[str, np.ndarray], model, batch_size = 128,
                epochs=1000, learning_rate=10**-4, grad_clipping = 1, latent_size = 128):

    rng_key = jax.random.PRNGKey(42)
    rng_key, init_key = jax.random.split(rng_key)

    example_loader = data_loader(
        train_data, batch_size=1, shuffle=False, latent_size=latent_size
    )

    example_graph, _, _ = next(example_loader)
    
    state = create_train_state(model, init_key, learning_rate, grad_clipping, example_graph)
    
    train_losses = []
    test_losses = []

    for step in range(epochs):

        rng_key, batch_key = jax.random.split(rng_key)

        count = 0
        total_loss = 0

        for graph, tgt, mask in data_loader(train_data, batch_size, shuffle=True, latent_size=latent_size):

            state = train_step(state, graph, tgt, mask, rng_key)

            current_loss = eval_step(state, graph, tgt, mask)
            current_loss.block_until_ready()
            
            count += 1
            total_loss += float(current_loss)
        
        ave_train_loss = float(total_loss / max(count, 1))
        train_losses.append(ave_train_loss)
        print(f"Step {step} | Training Loss: {train_losses[step]}")

        count = 0
        total_loss = 0
        for graph, tgt, mask in data_loader(test_data, batch_size, shuffle=False, latent_size=latent_size):

            loss_val = eval_step(state, graph, tgt, mask)
            
            count += 1
            total_loss += loss_val
        
        ave_test_loss = float(total_loss / max(count, 1))
        test_losses.append(ave_test_loss)
        print(f"Step {step} | Test Loss: {test_losses[step]}")

    return state, model, np.array(train_losses), np.array(test_losses)

@staticmethod
def predict(model, params, data_dir, data_prefix = 'test'):
    """
    Make predictions with a trained model.
    """
        
    predictions = []
    mask_arr = []
    tgt_arr = []
        
    # Get predictions batch by batch
    for graph, tgt, mask in create_dataloader(data_dir, data_prefix, shuffle=False):
        # Forward pass without dropout
        preds = model.apply({'params': params}, graph, deterministic = True)

        predictions.append(preds.nodes)
        tgt_arr.append(tgt)
        mask_arr.append(mask)
        
    # Concatenate all batch predictions
    return jnp.concatenate(predictions, axis=0).squeeze(), jnp.concatenate(tgt_arr, axis=0).squeeze(), jnp.concatenate(mask_arr, axis=0).squeeze()