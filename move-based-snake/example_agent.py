"""
Example agent interface for the move-based Snake game.
Shows how to create an agent that can be used with the game environment.
"""

from typing import Any
import numpy as np


class ExampleAgent:
    """
    Example agent interface.
    Your agent should implement at least one of: predict(), act(), or be callable.
    """
    
    def __init__(self, action_space_size: int = 6):
        self.action_space_size = action_space_size
    
    def predict(self, observation: np.ndarray) -> int:
        """
        Predict the next action given an observation.
        
        Args:
            observation: Game state observation (flattened grid)
            
        Returns:
            Action to take (0-5)
        """
        # Example: random agent
        import random
        return random.randint(0, self.action_space_size - 1)
    
    def act(self, observation: np.ndarray) -> int:
        """
        Alternative method name - same as predict().
        """
        return self.predict(observation)
    
    def __call__(self, observation: np.ndarray) -> int:
        """
        Make the agent callable.
        """
        return self.predict(observation)


# Example usage with the game:
if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    
    from game import MoveBasedSnakeGame
    
    # Create game and agent
    game = MoveBasedSnakeGame(grid_width=15, grid_height=15, num_monsters=2)
    agent = ExampleAgent(game.action_space_size)
    
    # Run one episode
    obs = game.reset()
    total_reward = 0
    steps = 0
    
    print("Running episode with example agent...")
    while not game.done and steps < 100:
        action = agent.predict(obs)
        obs, reward, done, info = game.step(action)
        total_reward += reward
        steps += 1
    
    print(f"Episode finished: steps={steps}, total_reward={total_reward:.2f}")

