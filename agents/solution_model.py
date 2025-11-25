"""
Example solution agent - template for solution implementation.
"""

import numpy as np
import random

try:
    from .base_agent import BaseAgent
except ImportError:
    from base_agent import BaseAgent


class SolutionAgent(BaseAgent):
    """
    Example solution agent class.
    """
    
    def __init__(self, game):
        """
        Initialize the solution agent.
        
        Args:
            game: The game environment (MoveBasedSnakeGame instance)
        """
        super().__init__(game)
        self.action_space_size = game.action_space_size
    
    def predict(self, observation: np.ndarray) -> int:
        """
        Predict the next action given an observation.
        
        Args:
            observation: Game state observation
            
        Returns:
            Action to take
        """
        return random.randint(0, self.action_space_size - 1)
    
    def act(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Choose an action (with exploration if training).
        
        Args:
            observation: Game state observation
            training: Whether we're in training mode
            
        Returns:
            Action to take
        """
        return self.predict(observation)
    
    def __call__(self, observation: np.ndarray) -> int:
        """
        Make the agent callable.
        """
        return self.predict(observation)