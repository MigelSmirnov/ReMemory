# memory/mlp_core/mininet_regression.py
"""
MiniNetRegression: a small fully-connected feedforward neural network
used to map a semantic context vector to token IDs.

This network is deliberately simple and lightweight — designed for fast training
on small datasets representing individual memory cells.
"""

import torch.nn as nn


class MiniNetRegression(nn.Module):
    """
    A minimal regression MLP model for memory reconstruction.

    Args:
        input_dim (int): Size of the input vector (semantic embedding dimension).
        hidden_dim (int): Size of the hidden layers.
        output_dim (int): Size of the output vector (number of token IDs to predict).
    """

    def __init__(self, input_dim: int = 1000, hidden_dim: int = 512, output_dim: int = 1000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim].
        """
        return self.net(x)
