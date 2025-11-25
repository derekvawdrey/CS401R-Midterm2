"""
DQN Agent wrapper that integrates with the game environment.
"""

import numpy as np
from typing import Optional
import os

try:
    from .base_agent import BaseAgent
    from .models.dqn_model import DQNAgent
except ImportError:
    from base_agent import BaseAgent
    from models.dqn_model import DQNAgent


class SnakeDQNAgent(BaseAgent):
    """
    DQN Agent for the Snake game.
    Wraps the DQNAgent to work with the game environment.
    """
    
    def __init__(
        self,
        game,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        model_path: Optional[str] = None
    ):
        """
        Initialize the DQN agent for Snake.
        
        Args:
            game: The game environment (MoveBasedSnakeGame instance)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            memory_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency (in steps) to update target network
            model_path: Path to load a saved model from (optional)
        """
        super().__init__(game)
        
        # Get state and action sizes from the game
        # Note: The game returns flattened grid, so state_size = grid_width * grid_height
        sample_state = game.reset()
        state_size = sample_state.shape[0]
        action_size = game.action_space_size
        
        # Create DQN agent
        self.dqn = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            memory_size=memory_size,
            batch_size=batch_size,
            target_update_freq=target_update_freq
        )
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.dqn.load(model_path)
    
    def predict(self, state: np.ndarray, debug: bool = False) -> int:
        """
        Predict the next action given an observation (no exploration).
        
        Args:
            state: Game state observation (flattened grid)
            debug: If True, print Q-values for debugging
            
        Returns:
            Action to take (0-3)
        """
        return self.dqn.predict(state, debug=debug)
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose an action (with exploration if training).
        
        Args:
            state: Game state observation
            training: Whether to use exploration
            
        Returns:
            Action to take
        """
        return self.dqn.act(state, training=training)
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.dqn.remember(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        return self.dqn.train_step_update()
    
    def save(self, filepath: str):
        """Save the model to a file."""
        self.dqn.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load the model from a file."""
        if os.path.exists(filepath):
            print(f"Loading model from {filepath}")
            self.dqn.load(filepath)
        else:
            raise FileNotFoundError(f"Model file not found: {filepath}")
    
    def __call__(self, state: np.ndarray) -> int:
        """Make the agent callable."""
        return self.predict(state)

