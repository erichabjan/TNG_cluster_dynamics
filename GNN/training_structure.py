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


def create_dataloader(data_dir: str, prefix: str, shuffle: bool = True,) -> Iterator[Tuple[jraph.GraphsTuple, jnp.ndarray, jnp.ndarray]]:
    """
    Yields batches saved by `graph_maker.py`.

    Each *.pkl file is already one fixed-size batch, so the iterator
    simply loads one file at a time and returns:
        • graph_batch  (jraph.GraphsTuple)
        • targets      (float32 array,  shape: B*MAX_NODES x 3)
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
    
    #count = 0

    for fname in files:
        with open(fname, "rb") as fh:
            saved = pickle.load(fh)

        # everything is already numpy / JAX-friendly inside the pickle
        yield (
            saved["graph"],        # jraph.GraphsTuple
            saved["targets"],      # jnp.ndarray
            saved["node_mask"],    # jnp.ndarray
        )

        #count += 1

        #if count == 10:
         #   break



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
        epochs=1000, 
        learning_rate=10**-4,
        grad_clipping = 1,
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
    
    state = create_train_state(model, init_key, learning_rate, grad_clipping, example_graph)
    
    # Training loop
    train_losses = []
    test_losses = []
    for step in range(epochs):

        rng_key, batch_key = jax.random.split(rng_key)

        ### Training
        count = 0
        total_loss = 0
        for graph, tgt, mask in create_dataloader(data_dir, train_prefix, shuffle=False):

            state = train_step(state, graph, tgt, mask, rng_key)
            current_loss = eval_step(state, graph, tgt, mask)
            
            count += 1
            total_loss += float(current_loss)

        
        ave_train_loss = float(total_loss / max(count, 1))
        train_losses.append(ave_train_loss)
        print(f"Step {step} | Training Loss: {train_losses[step]}")

        ### Testing
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

@staticmethod
def predict(model, params, data_dir, data_prefix = 'test'):
    """
    Make predictions with a trained model.
    """
        
    predictions = []
    mask_arr = []
    tgt_arr = []

    #count = 0
        
    # Get predictions batch by batch
    for graph, tgt, mask in create_dataloader(data_dir, data_prefix, shuffle=False):
        # Forward pass without dropout
        preds = model.apply({'params': params}, graph, deterministic = True)

        predictions.append(preds.nodes)
        tgt_arr.append(tgt)
        mask_arr.append(mask)

        #count += 1

        #if count == 10:
            
         #   break
        
    # Concatenate all batch predictions
    return jnp.concatenate(predictions, axis=0).squeeze(), jnp.concatenate(tgt_arr, axis=0).squeeze(), jnp.concatenate(mask_arr, axis=0).squeeze()