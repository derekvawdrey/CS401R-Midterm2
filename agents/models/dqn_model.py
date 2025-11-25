"""
Deep Q-Network (DQN) model for the Snake game.
Uses PyTorch for neural network implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple, Optional


class DQNNetwork(nn.Module):
    """Neural network for Q-value approximation."""
    
    def __init__(self, input_size: int, output_size: int, hidden_sizes: list = [128, 128, 64]):
        """
        Initialize the DQN network.
        
        Args:
            input_size: Size of the input observation (flattened grid)
            output_size: Number of actions
            hidden_sizes: List of hidden layer sizes
        """
        super(DQNNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add an experience to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network agent for reinforcement learning."""
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: Optional[str] = None
    ):
        """
        Initialize the DQN agent.
        
        Args:
            state_size: Size of the state observation
            action_size: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            memory_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency (in steps) to update target network
            device: Device to run on ('cuda' or 'cpu'), None for auto-detect
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Create networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(memory_size)
        
        # Training step counter
        self.train_step = 0
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            training: Whether we're in training mode (uses epsilon-greedy)
            
        Returns:
            Action to take
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def predict(self, state: np.ndarray) -> int:
        """
        Predict the best action (no exploration).
        
        Args:
            state: Current state observation
            
        Returns:
            Best action according to the Q-network
        """
        return self.act(state, training=False)
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step_update(self) -> Optional[float]:
        """
        Perform one training step if enough experiences are available.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_value = next_q_values.max(1)[0]
            target_q_value = rewards + (self.gamma * next_q_value * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(q_value, target_q_value)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save(self, filepath: str):
        """Save the Q-network to a file."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step
        }, filepath)
    
    def load(self, filepath: str):
        """Load the Q-network from a file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.train_step = checkpoint.get('train_step', 0)
        self.target_network.eval()
    
    def __call__(self, state: np.ndarray) -> int:
        """Make the agent callable."""
        return self.predict(state)

