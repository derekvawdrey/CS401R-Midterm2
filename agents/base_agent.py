"""
Base agent class for reinforcement learning agents.
All agents should inherit from this class and implement the required methods.
"""

from abc import abstractmethod
from typing import Optional
import numpy as np


class BaseAgent:
    """
    Base class for all RL agents.
    
    All agents must implement:
    - predict(): Get action without exploration
    - act(): Get action (with exploration if training)
    - __call__(): Make agent callable
    
    Optional methods (for training):
    - remember(): Store experience (for agents with replay buffer)
    - train_step(): Perform one training step
    - save(): Save the agent model
    - load(): Load the agent model
    """
    
    def __init__(self, game):
        """
        Initialize the base agent.
        
        Args:
            game: The game environment (FallingObjectsGame instance)
        """
        self.game = game
    
    @abstractmethod
    def predict(self, observation: np.ndarray) -> int:
        """
        Predict the best action given an observation (no exploration).
        
        Args:
            observation: Current state observation
            
        Returns:
            Action to take
        """
        pass
    
    @abstractmethod
    def act(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Choose an action (with exploration if training).
        
        Args:
            observation: Current state observation
            training: Whether we're in training mode (uses exploration)
            
        Returns:
            Action to take
        """
        pass
    
    @abstractmethod
    def __call__(self, observation: np.ndarray) -> int:
        """
        Make the agent callable (same as predict).
        
        Args:
            observation: Current state observation
            
        Returns:
            Action to take
        """
        pass
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """
        Store experience in replay buffer (optional, for agents that use it).
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Default: do nothing (for agents that don't use replay buffer)
        pass
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step (optional, for agents that train online).
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        # Default: do nothing (for agents that don't train online)
        return None
    
    def save(self, filepath: str):
        """
        Save the agent model to a file (optional).
        
        Args:
            filepath: Path to save the model
        """
        # Default: do nothing (for agents that don't support saving)
        pass
    
    def load(self, filepath: str):
        """
        Load the agent model from a file (optional).
        
        Args:
            filepath: Path to load the model from
        """
        # Default: do nothing (for agents that don't support loading)
        pass