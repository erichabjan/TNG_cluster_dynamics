import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
import numpy as np
import h5py
import time

import os
import sys
import pickle

### Custom code 
sys.path.append(os.getcwd())
from training_structure import train_model, data_loader, create_train_state, train_step, preload_hdf5_to_memory
from gnn import GraphConvNet

### Add a suffix for a new model
suffix = '_testing'

### Import data and create data loaders
data_path = "/projects/mccleary_group/habjan.e/TNG/Data/GNN_SBI_data/"
train_file = 'GNN_data_train.h5'
test_file = 'GNN_data_test.h5'

### Train model
if __name__ == "__main__":

    train_data = preload_hdf5_to_memory(data_path, train_file)
    test_data = preload_hdf5_to_memory(data_path, test_file)

    # Define hyperparameters
    epochs = 15
    batch_size = 32
    latent_size = 128

    early_stopping = True
    patience = 10

    total_steps = epochs * 10
    warm_up = int(0.05 * total_steps)
    decay = int(total_steps - warm_up)

    #learning_rate = optax.warmup_cosine_decay_schedule(init_value = 0.0, peak_value = 3e-4, warmup_steps = warm_up, decay_steps = decay, end_value = 3e-5)
    learning_rate = 10**-4
    gradient_clipping = 1
 
    # Create and train the model
    model = GraphConvNet(latent_size = latent_size, 
                         hidden_size = 256, 
                         num_mlp_layers = 3, 
                         message_passing_steps = 5, 
                         skip_connections = True,
                         edge_skip_connections = True,
                         norm = "pair", 
                         attention = True,
                         shared_weights = False,
                         relative_updates = False,
                         output_dim = 3,
                         dropout_rate = 0.0)

    trained_state, model, train_losses, test_losses = train_model(
        train_data=train_data,
        test_data=test_data,
        model=model, 
        epochs=epochs,
        batch_size = batch_size,
        learning_rate=learning_rate,
        grad_clipping=gradient_clipping,
        latent_size=latent_size,
        early_stopping=early_stopping, 
        patience=patience
    )

    # Save model parameters
    save_path = os.getcwd() + '/GNN_models/gnn_model_params' + suffix + '.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(trained_state.params, f)
    
    save_data = '/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/Loss_arrays/'
    np.save(save_data + 'train_loss' + suffix + '.npy', train_losses)
    np.save(save_data + 'test_loss' + suffix + '.npy', test_losses)