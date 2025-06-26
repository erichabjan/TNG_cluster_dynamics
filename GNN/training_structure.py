import functools

import optax 
from flax.training import train_state

import jax
import jax.numpy as jnp
from jax import random
import time
import numpy as np
from functools import partial

import tensorflow_datasets as tfds
import tensorflow as tf

import os, glob, pickle, random
from typing import Iterator, Tuple
import jraph


def create_dataloader(data_dir: str, prefix: str, shuffle: bool = True,) -> Iterator[Tuple[jraph.GraphsTuple, jnp.ndarray, jnp.ndarray]]:
    """
    Yields batches saved by `graph_maker.py`.

    Each *.pkl file is already one fixed-size batch, so the iterator
    simply loads one file at a time and returns:
        • graph_batch  (jraph.GraphsTuple)
        • targets      (float32 array,  shape: B*MAX_NODES × 3)
        • node_mask    (float32 array,  length:  B*MAX_NODES)

    Parameters
    ----------
    data_dir : str
        Directory that holds `train_batch_*.pkl` and `test_batch_*.pkl`.
    prefix : str
        Either `"train"` or `"test"`.
    shuffle : bool, default True
        When True, shuffle file order at every call
        (good for training; set False for evaluation).

    Example
    -------
    >>> for graph, tgt, mask in create_dataloader("/path/to/graph_data", "train"):
    ...     use_the_batch(graph, tgt, mask)
    """
    pattern = os.path.join(data_dir, f"{prefix}_batch_*.pkl")
    files   = sorted(glob.glob(pattern))
    if shuffle:
        random.shuffle(files)

    for fname in files:
        with open(fname, "rb") as fh:
            saved = pickle.load(fh)

        # everything is already numpy / JAX-friendly inside the pickle
        yield (
            saved["graph"],        # jraph.GraphsTuple
            saved["targets"],      # jnp.ndarray
            saved["node_mask"],    # jnp.ndarray
        )


def create_train_state(model, rng_key, learning_rate, example_graph):
    
    params = model.init(rng_key, example_graph)['params']
    optimizer = optax.adamw(learning_rate)
    
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

@staticmethod
def mse_loss(params, graph, target, mask, apply_fn, training=True):
    """
    Calculate weighted mean squared error loss.
        
    Args:
        params: Model parameters
        x: Input features
        y: Target values
        sigma: Uncertainty/error values for each target
        apply_fn: Function to apply the model
        rng: JAX random number generator key
        training: Whether in training mode (for dropout, batch norm, etc.)
            
    Returns:
        Weighted MSE loss value
    """
    preds = apply_fn({'params': params}, graph, training=training)
    mse = ((preds - target) ** 2).sum(-1)

    return (mse * mask).sum() / mask.sum()

@functools.partial(jax.jit, static_argnames="apply_fn")
def train_step(state, graph, target, mask):
    """Single training step."""
    
    def loss_fn(params):
        return mse_loss(params, graph, target, mask, state.apply_fn, training=True)

    grads = jax.grad(loss_fn)(state.params)

    return state.apply_gradients(grads=grads) 

@jax.jit
def eval_step(state, graph, target, mask):

    loss = mse_loss(state.params, graph, target, mask, state.apply_fn, training=False)

    return loss

def train_model(
        data_dir,
        model, 
        epochs=1000, 
        learning_rate=10**-4,
        train_prefix = 'train',
        test_prefix = 'test'
):
    """Main training loop."""
    
    # Initialize training state
    rng_key = jax.random.PRNGKey(42)
    rng_key, init_key = jax.random.split(rng_key)

    example_graph, example_tgt, example_mask = next(
        create_dataloader(data_dir, train_prefix, shuffle=False)
    )
    
    state = create_train_state(model, init_key, learning_rate, example_graph)
    
    # Training loop
    train_losses = []
    test_losses = []
    for step in range(epochs):

        rng_key, batch_key = jax.random.split(rng_key)

        ### Training
        count = 0
        total_loss = 0
        for graph, tgt, mask in create_dataloader(data_dir, train_prefix, shuffle=True):

            state = train_step(state, graph, tgt, mask)
            current_loss = eval_step(state, graph, tgt, mask)
            
            count += 1
            total_loss += float(current_loss)
        
        ave_train_loss = float(total_loss / max(count, 1))
        train_losses.append(ave_train_loss)
        print(f"Step {step} | Training Loss: {train_losses[step]}")

        ### Testing
        master_rng, eval_key = jax.random.split(master_rng)
        count = 0
        total_loss = 0
        for graph, tgt, mask in create_dataloader(data_dir, test_prefix, shuffle=False):

            loss_val = eval_step(state, graph, tgt, mask)
            
            count += 1
            total_loss += loss_val
        
        ave_test_loss = float(total_loss / max(count, 1))
        test_losses.append(ave_test_loss)
        print(f"Step {step} | Test Loss: {test_losses[step]}")

    return state, model, np.array(train_losses), np.array(test_losses)