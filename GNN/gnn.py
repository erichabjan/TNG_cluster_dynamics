import jax
from jax import random

import flax.linen as nn
import jraph
from typing import Callable, Sequence
import jax.numpy as jnp

class MLP(nn.Module):
    """Multi-layer perceptron with customizable activation."""
    
    feature_sizes: Sequence[int]
    activation: Callable = nn.gelu
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x, deterministic=True):
        for i, features in enumerate(self.feature_sizes[:-1]):
            x = nn.Dense(features)(x)
            x = self.activation(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        
        # Final layer without activation
        x = nn.Dense(self.feature_sizes[-1])(x)
        return x


class PairNorm(nn.Module):
    """PairNorm normalization layer."""
    
    @nn.compact
    def __call__(self, features, rescale_factor=1.0):
        eps = 1e-5
        
        # Center features
        feature_mean = jnp.mean(features, axis=0, keepdims=True)
        feature_centered = features - feature_mean
        
        # L2 norm per node
        feature_l2 = jnp.sqrt(jnp.sum(jnp.square(feature_centered), axis=1, keepdims=True))
        
        # Mean L2 norm across all nodes
        feature_l2_mean = jnp.sqrt(jnp.mean(jnp.square(feature_l2)))
        
        # Normalize
        features_normalized = (
            feature_centered / (feature_l2 + eps) * feature_l2_mean * rescale_factor
        )
        
        return features_normalized


def get_node_mlp_updates(mlp_feature_sizes: Sequence[int], dropout_rate: float = 0.0, 
                        name: str = None) -> Callable:
    """Get node update function."""
    
    def update_fn(nodes, sent_attributes, received_attributes, globals, 
                 deterministic=True):
        if received_attributes is not None:
            inputs = jnp.concatenate([nodes, received_attributes, globals], axis=1)
        else:
            inputs = jnp.concatenate([nodes, globals], axis=1)
        
        return MLP(mlp_feature_sizes, dropout_rate=dropout_rate, name=name)(
            inputs, deterministic=deterministic
        )
    
    return update_fn


def get_edge_mlp_updates(mlp_feature_sizes: Sequence[int], dropout_rate: float = 0.0,
                        relative_updates: bool = False, name: str = None) -> Callable:
    """Get edge update function."""
    
    def update_fn(edges, senders, receivers, globals, deterministic=True):
        if edges is not None:
            if relative_updates:
                inputs = jnp.concatenate([edges, senders - receivers, globals], axis=1)
            else:
                inputs = jnp.concatenate([edges, senders, receivers, globals], axis=1)
        else:
            if relative_updates:
                inputs = jnp.concatenate([senders - receivers, globals], axis=1)
            else:
                inputs = jnp.concatenate([senders, receivers, globals], axis=1)
        
        return MLP(mlp_feature_sizes, dropout_rate=dropout_rate, name=name)(
            inputs, deterministic=deterministic
        )
    
    return update_fn


def get_attention_logit_fn(hidden_size: int, dropout_rate: float = 0.0, 
                          name: str = None) -> Callable:
    """Get attention logits function."""
    
    def attention_logit_fn(edges, senders, receivers, globals, deterministic=True):
        inputs = jnp.concatenate([edges, senders, receivers, globals], axis=-1)
        return MLP([hidden_size, 1], dropout_rate=dropout_rate, name=name)(
            inputs, deterministic=deterministic
        )
    
    return attention_logit_fn


class GraphConvNet(nn.Module):
    """Graph Convolutional Network for velocity prediction."""
    
    latent_size: int
    hidden_size: int
    num_mlp_layers: int
    message_passing_steps: int
    skip_connections: bool = True
    edge_skip_connections: bool = True
    norm: str = "layer"
    attention: bool = False
    shared_weights: bool = False
    relative_updates: bool = False
    output_dim: int = 3
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, deterministic=True) -> jraph.GraphsTuple:
        """Apply GNN to predict velocity field."""
        
        mlp_feature_sizes = [self.hidden_size] * self.num_mlp_layers + [self.latent_size]
        
        # Initial embedding
        if graph.edges is None:
            embedder = jraph.GraphMapFeatures(
                embed_node_fn=MLP(mlp_feature_sizes, dropout_rate=self.dropout_rate)
            )
        else:
            embedder = jraph.GraphMapFeatures(
                embed_node_fn=MLP(mlp_feature_sizes, dropout_rate=self.dropout_rate),
                embed_edge_fn=MLP(mlp_feature_sizes, dropout_rate=self.dropout_rate),
            )
        
        graph = embedder(graph)
        
        # Ensure globals are properly shaped
        if graph.globals.ndim > 1:
            graph = graph._replace(globals=graph.globals.reshape(graph.globals.shape[0], -1))
        
        # Message passing steps
        for step in range(self.message_passing_steps):
            if step == 0 or not self.shared_weights:
                suffix = "shared" if self.shared_weights else step
                
                update_node_fn = get_node_mlp_updates(
                    mlp_feature_sizes, 
                    dropout_rate=self.dropout_rate,
                    name=f"update_node_fn_{suffix}"
                )
                update_edge_fn = get_edge_mlp_updates(
                    mlp_feature_sizes,
                    dropout_rate=self.dropout_rate,
                    relative_updates=self.relative_updates,
                    name=f"update_edge_fn_{suffix}"
                )
                attention_logit_fn = (
                    get_attention_logit_fn(
                        self.hidden_size,
                        dropout_rate=self.dropout_rate,
                        name=f"attention_logit_fn_{suffix}"
                    ) if self.attention else None
                )
                
                graph_net = jraph.GraphNetwork(
                    update_node_fn=lambda *args: update_node_fn(*args, deterministic=deterministic),
                    update_edge_fn=lambda *args: update_edge_fn(*args, deterministic=deterministic),
                    attention_logit_fn=(
                        lambda *args: attention_logit_fn(*args, deterministic=deterministic)
                        if attention_logit_fn else None
                    ),
                    attention_reduce_fn=lambda edges, weights: edges * weights if self.attention else None,
                )
            
            # Apply graph network with skip connections
            if self.skip_connections:
                new_graph = graph_net(graph)
                graph = graph._replace(
                    nodes=graph.nodes + new_graph.nodes,
                    edges=(
                        new_graph.edges if graph.edges is None or not self.edge_skip_connections
                        else graph.edges + new_graph.edges
                    ),
                )
            else:
                graph = graph_net(graph)
            
            # Apply normalization
            if self.norm == "layer":
                norm = nn.LayerNorm()
            elif self.norm == "pair":
                norm = PairNorm()
            else:
                norm = lambda x: x  # Identity
            
            if graph.nodes is not None:
                graph = graph._replace(nodes=norm(graph.nodes))
            if graph.edges is not None:
                graph = graph._replace(edges=norm(graph.edges))
        
        # Final decoder to output dimension
        decoder = jraph.GraphMapFeatures(
            embed_node_fn=MLP(
                [self.hidden_size] * self.num_mlp_layers + [self.output_dim],
                dropout_rate=self.dropout_rate
            )
        )
        
        return decoder(graph)