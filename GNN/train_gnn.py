import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from sklearn.model_selection import train_test_split
from functools import partial
import numpy as np
#import tensorflow_datasets as tfds
#import tensorflow as tf
from astropy.table import Table

import numpy as np 
import os
import sys
import pickle

### Custom code 
sys.path.append(os.getcwd())
from training_structure import train_model, create_dataloader
from gnn import GraphConvNet

### Add a suffix for a new model
suffix = '_100'

### Import data and create data loaders
batch_file_path = "/projects/mccleary_group/habjan.e/TNG/Data/GNN_SBI_data/graph_data"

### Train model
if __name__ == "__main__":
    # Define hyperparameters
    epochs = 100
    learning_rate = 1e-2
 
    # Create and train the model
    model = GraphConvNet(latent_size = 128, hidden_size = 256, num_mlp_layers = 3, message_passing_steps = 5)

    trained_state, model, train_losses, test_losses = train_model(
        data_dir=batch_file_path,  
        model=model, 
        epochs=epochs,
        learning_rate=learning_rate,
        train_prefix = 'train',
        test_prefix = 'test'
    )

    # Save model parameters
    save_path = os.getcwd() + '/GNN_models/gnn_model_params' + suffix + '.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(trained_state.params, f)
    
    save_data = '/home/habjan.e/TNG/Sandbox_notebooks/phase_space_recon/Loss_arrays/'
    np.save(save_data + 'train_loss' + suffix + '.npy', train_losses)
    np.save(save_data + 'test_loss' + suffix + '.npy', test_losses)