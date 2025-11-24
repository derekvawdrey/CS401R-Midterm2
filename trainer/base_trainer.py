from abc import abstractmethod

class BaseTrainer:
    def __init__(self, model, env):
        self.model = model
        self.env = env

    @abstractmethod
    def train(self, num_episodes: int):
        pass

    def evaluate(self, num_episodes: int):
        pass