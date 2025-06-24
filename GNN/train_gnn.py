import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from sklearn.model_selection import train_test_split
from functools import partial
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

import numpy as np 
import os
import sys
import pickle

### Custom code 
sys.path.append(os.getcwd())
from training_structure import train_model, create_dataloader
from gnn import GraphConvNet

### Add a suffix for a new model
suffix = ''

### Import data
base_path = '/home/habjan.e/TNG/Data/GNN_SBI_data/'

train = fits.open(base_path + 'GNN_data_train.fits')
test = fits.open(base_path + 'GNN_data_test.fits')

### Creat numpy arrays for data

trainx = np.array([np.array(train['x position']), np.array(train['y position']), np.array(train['z velocity'])])
trainy = np.array([np.array(train['z position']), np.array(train['x velocity']), np.array(train['y velocity'])])

testx = np.array([np.array(test['x position']), np.array(test['y position']), np.array(test['z velocity'])])
testy = np.array([np.array(test['z position']), np.array(test['x velocity']), np.array(test['y velocity'])])

# Create data loaders
batch_size = 32
train_ds = create_dataloader(trainx, trainy, batch_size=batch_size, shuffle=True)
test_ds = create_dataloader(testx, testy, batch_size=batch_size, shuffle=False)

### Train model
if __name__ == "__main__":
    # Define hyperparameters
    epochs = 5
    learning_rate = 1e-3
 
    # Create and train the model
    model = GraphConvNet()

    trained_state, model, losses = train_model(
        train_ds=train_ds, 
        test_ds=test_ds, 
        model=model, 
        epochs=epochs, 
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    # Save model parameters
    save_path = os.getcwd() + '/GNN_models/gnn_model_params' + suffix + '.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(trained_state.params, f)