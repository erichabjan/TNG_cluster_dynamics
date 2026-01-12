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

from jax.tree_util import tree_leaves

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
        #batched_targets = jnp.array(data_dict['targets'][batch_indices]).reshape(-1, 1)
        batched_masks = jnp.array(data_dict['masks'][batch_indices]).reshape(-1)
        
        yield batched_graph, batched_targets, batched_masks



def create_train_state(model, rng_key, learning_rate, grad_clipping, example_graph):
    
    params = model.init(rng_key, example_graph, deterministic = False)['params']
    optimizer = optax.chain(optax.clip_by_global_norm(grad_clipping), optax.adam(learning_rate))
    
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

def mse_loss(params, graph, target, mask, apply_fn, training, rng=None):
    deterministic = not training
    kwargs = dict(deterministic=deterministic)
    if not deterministic:
        kwargs["rngs"] = {"dropout": rng}

    preds_graph = apply_fn({'params': params}, graph, **kwargs)
    preds = preds_graph.nodes          # (N, 3)

    mask_f = mask.astype(preds.dtype)  # (N,)
    count  = mask_f.sum() + 1e-12      # scalar

    # --- per-dimension masked mean/std (dims: z, vx, vy) ---
    # shape (3,)
    mean_pred = (preds  * mask_f[:, None]).sum(axis=0) / count
    mean_tgt  = (target * mask_f[:, None]).sum(axis=0) / count

    var_pred = ((preds  - mean_pred)**2  * mask_f[:, None]).sum(axis=0) / count
    var_tgt  = ((target - mean_tgt)**2  * mask_f[:, None]).sum(axis=0) / count

    std_loss = jnp.mean((var_pred - var_tgt) ** 2)

    jax.debug.print(
        "pred mean/std (z,vx,vy): {mp} {sp}\n"
        "tgt  mean/std (z,vx,vy): {mt} {st}\n"
        "std_loss: {sl}, mask count: {mc}",
        mp=mean_pred,
        sp=jnp.sqrt(var_pred),
        mt=mean_tgt,
        st=jnp.sqrt(var_tgt),
        sl=std_loss,
        mc=count,
    )

    # --- usual masked MSE ---
    mse_per_node = ((preds - target) ** 2).sum(-1)    # (N,)
    mse = (mse_per_node * mask_f).sum() / count
    loss = mse + 1.0 * std_loss

    return loss

def tree_l2_norm(tree):
    return jnp.sqrt(sum([jnp.vdot(x, x) for x in tree_leaves(tree)]))

@jax.jit
def train_step(state, graph, target, mask, rng_key):
    def loss_fn(params):
        return mse_loss(params, graph, target, mask,
                        state.apply_fn, training=True, rng=rng_key)

    grads = jax.grad(loss_fn)(state.params)

    jax.debug.print(
        "grad norm: {g}, param norm: {p}",
        g=tree_l2_norm(grads),
        p=tree_l2_norm(state.params),
    )

    return state.apply_gradients(grads=grads)

@jax.jit
def eval_step(state, graph, target, mask):

    loss = mse_loss(state.params, graph, target, mask, state.apply_fn, training=False, rng=None)
    #loss = hybrid_mse_energy_loss(state.params, graph, target, mask, state.apply_fn, training=False, rng=None)

    return loss

def train_model(train_data: Dict[str, np.ndarray], test_data: Dict[str, np.ndarray], model, batch_size = 128,
                epochs=1000, learning_rate=10**-4, grad_clipping = 1, latent_size = 128, 
                early_stopping=False, patience=5):

    rng_key = jax.random.PRNGKey(42)
    rng_key, init_key = jax.random.split(rng_key)

    example_loader = data_loader(
        train_data, batch_size=1, shuffle=False, latent_size=latent_size
    )

    example_graph, _, _ = next(example_loader)
    
    state = create_train_state(model, init_key, learning_rate, grad_clipping, example_graph)
    
    train_losses = []
    test_losses = []

    ### Early stopping initial quantities
    best_loss = float('inf')
    best_state = None
    epochs_without_improvement = 0

    for step in range(epochs):

        count = 0
        total_loss = 0

        for graph, tgt, mask in data_loader(train_data, batch_size, shuffle=True, latent_size=latent_size):

            rng_key, batch_key = jax.random.split(rng_key)
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

        # Early stopping logic
        if early_stopping:
            if ave_test_loss < best_loss:
                best_loss = ave_test_loss
                best_state = state
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {step+1}")
                if best_state is not None:
                    state = best_state
                break

    return state, model, np.array(train_losses), np.array(test_losses)

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