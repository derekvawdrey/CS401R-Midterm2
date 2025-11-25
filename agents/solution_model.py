from base_agent import BaseAgent
import numpy as np
import random

class SolutionAgent(BaseAgent):
    def __init__(self, model, env):
        super().__init__(model, env)

    def train(self, num_episodes: int):
        pass

    def predict(self, observation: np.ndarray) -> int:
        return random.randint(0, self.action_space_size - 1)

    def act(self, observation: np.ndarray) -> int:
        return self.predict(observation)

    def __call__(self, observation: np.ndarray) -> int:
        return self.predict(observation)