import functools

import optax 
from flax.training import train_state

import jax
import jax.numpy as jnp
from jax import random
import time
import numpy as np
from functools import partial

pos_inputs = [0,1]
vel_inputs = [2]

pos_pred = [2]
vel_pred = [0,1]


@staticmethod
def create_dataloader(x, y, batch_size, shuffle=True):
    """
    Create a TensorFlow dataset from NumPy arrays.
        
    Args:
        x: Input features
        y: Target values
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the dataset
            
    Returns:
        A TensorFlow dataset
    """
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def create_train_state(model, rng_key, learning_rate, input_shape):
    """Create initial training state."""
    dummy_x = jnp.ones((1, input_shape[0], 3))  # (x, y, vz)
    dummy_t = jnp.ones((1, 1))
    params = model.init(rng_key, dummy_x, dummy_x, dummy_t, deterministic=True)
    
    optimizer = optax.adamw(learning_rate)
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )

@functools.partial(jax.jit, static_argnames=['model'])
def train_step(state, inputs, outputs, rng_key, model):
    """Single training step."""
    
    def loss_fn(params):
        # Sample random time t ∈ [0, 1]
        batch_size = inputs.shape[0]
        rng_key_t, rng_key_noise = jax.random.split(rng_key)
        t = jax.random.uniform(rng_key_t, (batch_size, 1))
        
        # Sample noise
        x0 = jax.random.normal(rng_key_noise, outputs.shape)
        
        # x_t = (1 - t) * x0 + t * x1
        alpha_t = t[..., None]  
        x_t = (1 - alpha_t) * x0 + alpha_t * outputs 
        
        # The target velocity field is x1 - x0
        target_velocity = outputs - x0 
        
        # Forward pass: predict velocity field given interpolated state and inputs
        predicted_velocity = model.apply(
            params, 
            inputs, 
            x_t, 
            t, 
            deterministic=False,  
            rngs={'dropout': rng_key}
        )
        # Compute MSE loss
        loss = jnp.mean(jnp.square(predicted_velocity - target_velocity))
        return loss, predicted_velocity
    
    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, predicted_velocity), grads = grad_fn(state.params)
    
    # Update parameters
    state = state.apply_gradients(grads=grads)
    return state, loss

def train_model(
        train_ds,
        test_ds,
        model, 
        epochs=1000, 
        batch_size=10, 
        learning_rate=1e-4,
):
    """Main training loop."""
    
    # Initialize training state
    rng_key = jax.random.PRNGKey(42)
    rng_key, init_key = jax.random.split(rng_key)
    
    state = create_train_state(
        model, init_key, learning_rate, 
        input_shape=(galaxies_per_cluster,)
    )
    
    # Training loop
    losses = []
    for step in range(epochs):
        # Generate batch
        rng_key, batch_key = jax.random.split(rng_key)

        positions, velocities, _, _, batch_key = generate_batch_fn(
            batch_key, batch_size, galaxies_per_cluster
        )
        # Prepare inputs and outputs
        # inputs - what we observe
        inputs = jnp.concatenate([
            positions[..., pos_inputs],
            velocities[..., vel_inputs]   
        ], axis=-1)
        
        # outputs - what we want to predict
        outputs = jnp.concatenate([
            positions[..., pos_pred],   
            velocities[..., vel_pred]    
        ], axis=-1)
        
        # Training step
        rng_key, step_key = jax.random.split(rng_key)
        state, loss = train_step(state, inputs, outputs, step_key, model)
        
        losses.append(float(loss))
        
        # Logging
        if step % 100 == 0:
            avg_loss = jnp.mean(jnp.array(losses[-100:] if len(losses) >= 100 else losses))
            print(f"Step {step:4d} | Loss: {loss:.6f} | Avg Loss: {avg_loss:.6f}")
    
    return state, model, losses