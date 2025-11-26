"""
Deep Q-Network (DQN) agent implementation.
"""

import random
from collections import deque
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .base_agent import BaseAgent
    from .models.dqn_model import DQNModel
except ImportError:
    from base_agent import BaseAgent
    from models.dqn_model import DQNModel


class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent for reinforcement learning.
    
    Implements:
    - Experience replay buffer
    - Target network for stable training
    - Epsilon-greedy exploration
    - Double DQN (uses main network for action selection)
    """
    
    def __init__(
        self,
        game,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        memory_size: int = 10000,
        target_update_freq: int = 100,
        model_path: Optional[str] = None,
        hidden_sizes: list = None
    ):
        """
        Initialize the DQN agent.
        
        Args:
            game: Game environment (FallingObjectsGame instance)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon: Initial epsilon for epsilon-greedy exploration
            epsilon_min: Minimum epsilon value
            epsilon_decay: Epsilon decay rate per episode
            batch_size: Batch size for training
            memory_size: Size of replay buffer
            target_update_freq: Frequency (in steps) to update target network
            model_path: Path to load a pre-trained model from
            hidden_sizes: List of hidden layer sizes for the neural network
        """
        super().__init__(game)
        
        # Get state and action sizes from the game
        # Create a dummy observation to determine state size
        dummy_obs = game.reset()
        self.state_size = len(dummy_obs)
        self.action_size = game.action_space_size
        
        # State normalization: observations range from -1.0 to 2.0
        # Normalize to approximately [-1, 1] range for better training
        # Values: -1.0 (walls), 0.0 (empty), 0.5 (warning 2 steps), 1.0 (warning 1 step), 1.5 (landing), 2.0 (player)
        # We'll normalize by dividing by 2.0 to bring range to [-0.5, 1.0], then scale to [-1, 1]
        self.state_min = -1.0
        self.state_max = 2.0
        self.normalize_state = True
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device (use GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Better default network architecture for large state space (403 dimensions)
        if hidden_sizes is None:
            # Use larger network for better capacity
            hidden_sizes = [256, 256, 128]
        
        # Create main and target networks
        self.dqn = DQNModel(self.state_size, self.action_size, hidden_sizes).to(self.device)
        self.target_dqn = DQNModel(self.state_size, self.action_size, hidden_sizes).to(self.device)
        
        # Initialize target network with same weights as main network
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.target_dqn.eval()  # Target network is always in eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Training step counter (for target network updates)
        self.train_step_count = 0
        
        # Load model if provided
        if model_path:
            self.load(model_path)
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize state to improve training stability.
        
        Args:
            state: Raw state observation
            
        Returns:
            Normalized state
        """
        if self.normalize_state:
            # Normalize to [-1, 1] range
            # Original range: [-1.0, 2.0]
            # Scale to [-1, 1]: (state - min) / (max - min) * 2 - 1
            normalized = (state - self.state_min) / (self.state_max - self.state_min) * 2.0 - 1.0
            return normalized
        return state
    
    def predict(self, observation: np.ndarray) -> int:
        """
        Predict the best action given an observation (no exploration).
        
        Args:
            observation: Current state observation
            
        Returns:
            Action to take
        """
        self.dqn.eval()
        with torch.no_grad():
            # Normalize state before passing to network
            normalized_obs = self._normalize_state(observation)
            state_tensor = torch.FloatTensor(normalized_obs).unsqueeze(0).to(self.device)
            q_values = self.dqn(state_tensor)
            action = q_values.argmax().item()
        self.dqn.train()
        return action
    
    def act(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Choose an action (with exploration if training).
        
        Args:
            observation: Current state observation
            training: Whether we're in training mode (uses exploration)
            
        Returns:
            Action to take
        """
        if training and random.random() < self.epsilon:
            # Random action (exploration)
            return random.randrange(self.action_size)
        else:
            # Greedy action (exploitation)
            return self.predict(observation)
    
    def __call__(self, observation: np.ndarray) -> int:
        """
        Make the agent callable (same as predict).
        
        Args:
            observation: Current state observation
            
        Returns:
            Action to take
        """
        return self.predict(observation)
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step using experience replay.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        # Need enough experiences in buffer
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample a batch from replay buffer
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Normalize states before converting to tensors
        states_normalized = np.array([self._normalize_state(s) for s in states])
        next_states_normalized = np.array([self._normalize_state(s) for s in next_states])
        
        # Convert to tensors
        states = torch.FloatTensor(states_normalized).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states_normalized).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q-values (for the actions we took)
        current_q_values = self.dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q-values using target network
        with torch.no_grad():
            # Use main network to select best action (Double DQN)
            next_actions = self.dqn(next_states).argmax(1)
            # Use target network to evaluate that action
            next_q_values = self.target_dqn(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # Set Q-value to 0 for terminal states
            next_q_values[dones] = 0.0
        
        # Compute target Q-values
        target_q_values = rewards + (self.gamma * next_q_values)
        
        # Clip rewards to prevent exploding Q-values (helps with stability)
        # Rewards are typically in range [-10, 0.1], so clipping to [-10, 1] is safe
        target_q_values = torch.clamp(target_q_values, min=-20.0, max=10.0)
        
        # Compute loss (using Huber loss for more robust training)
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay epsilon (should be called after each episode)."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
    
    def episode_end(self):
        """Called at the end of each episode to decay epsilon."""
        self.decay_epsilon()
    
    def save(self, filepath: str):
        """
        Save the agent model to a file.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'dqn_state_dict': self.dqn.state_dict(),
            'target_dqn_state_dict': self.target_dqn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'train_step_count': self.train_step_count,
            'state_min': self.state_min,
            'state_max': self.state_max,
            'normalize_state': self.normalize_state,
        }, filepath)
    
    def load(self, filepath: str):
        """
        Load the agent model from a file.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.dqn.load_state_dict(checkpoint['dqn_state_dict'])
        self.target_dqn.load_state_dict(checkpoint['target_dqn_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.train_step_count = checkpoint.get('train_step_count', 0)
        self.state_min = checkpoint.get('state_min', self.state_min)
        self.state_max = checkpoint.get('state_max', self.state_max)
        self.normalize_state = checkpoint.get('normalize_state', self.normalize_state)
        
        print(f"Model loaded from {filepath}")
        print(f"Epsilon: {self.epsilon:.3f}")

