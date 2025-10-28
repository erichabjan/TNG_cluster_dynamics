import functools

import optax 
from flax.training import train_state

import jax
import jax.numpy as jnp
from jax import random
import time
import numpy as np
from functools import partial

import os, glob, pickle, random
from typing import Iterator, Tuple
import jraph
import h5py

import time


def data_loader(
    data_dir: str, 
    file_in: str, 
    batch_size: int, 
    example: bool = False, 
    shuffle: bool = True, 
    file_handle = None
) -> Iterator[Tuple[jraph.GraphsTuple, jnp.ndarray, jnp.ndarray]]:
    
    # Determine if we need to open the file or not
    should_close = False
    if file_handle is None:
        file_path = data_dir + file_in
        print(f"Opening file: {file_path}")
        start = time.time()
        f = h5py.File(file_path, 'r')
        print(f"File opened in {time.time() - start:.2f}s")
        should_close = True
    else:
        f = file_handle
        print("Using pre-opened file handle")
    
    try:
        # Get all sample IDs
        sample_ids = list(f.keys())
        n_samples = len(sample_ids)
        print(f"Total samples in file: {n_samples}")
        
        # Test read speed on first sample
        if should_close:  # Only test if we just opened the file
            start = time.time()
            sample = f[sample_ids[0]]
            _ = sample['padded_nodes'][:]
            print(f"First sample loaded in {time.time() - start:.2f}s")
        
        # Create indices and optionally shuffle
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)
        
        if example:
            sample_key = sample_ids[indices[0]]
            sample = f[sample_key]
            
            padded_nodes = jnp.array(sample['padded_nodes'][:])
            node_mask = jnp.array(sample['node_mask'][:])
            padded_targets = jnp.array(sample['padded_targets'][:])
            n_nodes = sample.attrs['n_nodes']
            n_edges = sample.attrs['n_edges']
            
            graph = jraph.GraphsTuple(
                nodes=padded_nodes,
                edges=None,
                senders=jnp.zeros((0,), dtype=jnp.int32),
                receivers=jnp.zeros((0,), dtype=jnp.int32),
                n_node=jnp.array([n_nodes], dtype=jnp.int32),
                n_edge=jnp.array([n_edges], dtype=jnp.int32),
                globals=jnp.zeros((1, 128), dtype=jnp.float32)  # LATENT_SIZE=128
            )
            
            yield graph, padded_targets, node_mask
            return
        

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            
            batch_graphs = []
            batch_targets = []
            batch_masks = []
            
            for idx in batch_indices:
                sample_key = sample_ids[idx]
                sample = f[sample_key]
                
                padded_nodes = jnp.array(sample['padded_nodes'][:])
                node_mask = jnp.array(sample['node_mask'][:])
                padded_targets = jnp.array(sample['padded_targets'][:])
                n_nodes = sample.attrs['n_nodes']
                n_edges = sample.attrs['n_edges']

                graph = jraph.GraphsTuple(
                    nodes=padded_nodes,
                    edges=None,
                    senders=jnp.zeros((0,), dtype=jnp.int32),
                    receivers=jnp.zeros((0,), dtype=jnp.int32),
                    n_node=jnp.array([n_nodes], dtype=jnp.int32),
                    n_edge=jnp.array([n_edges], dtype=jnp.int32),
                    globals=jnp.zeros((1, 128), dtype=jnp.float32)
                )
                
                batch_graphs.append(graph)
                batch_targets.append(padded_targets)
                batch_masks.append(node_mask)
            
            batched_graph = jraph.batch(batch_graphs)
            batched_targets = jnp.concatenate(batch_targets, axis=0)
            batched_masks = jnp.concatenate(batch_masks, axis=0)
            
            yield batched_graph, batched_targets, batched_masks
    
    finally:
        if should_close:
            f.close()
            print("File closed")



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
    print(preds.shape, target.shape, mask.shape)
    mse = ((preds - target) ** 2).sum(-1)

    print(mse)

    mse_masked = (mse * mask).sum() / mask.sum()
    print(mse_masked)

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

def train_model(
        data_dir,
        model, 
        train_file,
        test_file,
        batch_size = 128,
        epochs=1000, 
        learning_rate=10**-4,
        grad_clipping = 1
):

    rng_key = jax.random.PRNGKey(42)
    rng_key, init_key = jax.random.split(rng_key)

    print("Opening training and test files...")
    train_h5 = h5py.File(data_dir + train_file, 'r')
    test_h5 = h5py.File(data_dir + test_file, 'r')

    try:

        example_graph, example_tgt, example_mask = next(
            data_loader(data_dir = data_dir, file_in = train_file, batch_size = batch_size, example = True)
        )
    
        state = create_train_state(model, init_key, learning_rate, grad_clipping, example_graph)
    
        train_losses = []
        test_losses = []
        for step in range(epochs):

            rng_key, batch_key = jax.random.split(rng_key)

            count = 0
            total_loss = 0
            for graph, tgt, mask in data_loader(data_dir = data_dir, file_in = train_file, 
                                                batch_size = batch_size, shuffle = True,
                                                file_handle=train_h5):

                state = train_step(state, graph, tgt, mask, rng_key)
                current_loss = eval_step(state, graph, tgt, mask)
            
                count += 1
                total_loss += float(current_loss)
        
            ave_train_loss = float(total_loss / max(count, 1))
            train_losses.append(ave_train_loss)
            print(f"Step {step} | Training Loss: {train_losses[step]}")

            count = 0
            total_loss = 0
            for graph, tgt, mask in data_loader(data_dir = data_dir, file_in = test_file, 
                                                batch_size = batch_size, shuffle=False,
                                                file_handle=test_h5):

                loss_val = eval_step(state, graph, tgt, mask)
            
                count += 1
                total_loss += loss_val
        
            ave_test_loss = float(total_loss / max(count, 1))
            test_losses.append(ave_test_loss)
            print(f"Step {step} | Test Loss: {test_losses[step]}")

        return state, model, np.array(train_losses), np.array(test_losses)

    finally:
        print("\nClosing files...")
        train_h5.close()
        test_h5.close()

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