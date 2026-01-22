The Graph Neural Network (GNN) developed in this folder is trained using the [BAHAMAS simulation](https://arxiv.org/abs/1603.02702) and is evaluated on the [TNG simulation](https://www.tng-project.org/data/). The models found in the `GNN_models` takes the ($x$, $y$, $v_z$) coordinates of cluster-member galaxies and outputs the cluster-centric radii and magnitude of velocities for all cluster-member galaxies. The training and validation curves of the current model is shown here: 

<p align="center">
  <img src="figures/loss_curves.png" alt="loss_curves" width="width:100%" />
</p>

And the ability of the model to reconstruct the cluster-centric radii and magnitude of velocities:

<p align="center">
  <img src="figures/tng_predictions.png" alt="tng_predictions" width="width:100%" />
</p>

The reconstruction of these quantities will allow for the dynamical state of the clsuter to be probed and for projection effects to be accounted for in both dynamical and weak lensing derived masses. To see the progress of the mode, check out my [Weights and Biases for this project](https://wandb.ai/erichabjan-northeastern-university/phase-space-GNN).