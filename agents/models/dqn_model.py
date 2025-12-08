"""
Deep Q-Network (DQN) model for reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNModel(nn.Module):
    """
    Deep Q-Network model for Q-value estimation.
    
    The network takes the game state as input and outputs Q-values for each action.
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: list = None):
        """
        Initialize the DQN model.
        
        Args:
            state_size: Size of the state/observation vector
            action_size: Number of possible actions
            hidden_sizes: List of hidden layer sizes (default: [128, 128])
        """
        super(DQNModel, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [128, 128]
        
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor of shape (batch_size, state_size) or (state_size,)
            
        Returns:
            Q-values for each action, shape (batch_size, action_size) or (action_size,)
        """
        return self.network(state)

