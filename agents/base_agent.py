from abc import abstractmethod

class BaseAgent:
    def __init__(self, model, env):
        self.model = model
        self.env = env

    @abstractmethod
    def train(self, num_episodes: int):
        pass

    def evaluate(self, num_episodes: int):
        pass

    @abstractmethod
    def predict(self, observation: np.ndarray) -> int:
        pass

    @abstractmethod
    def act(self, observation: np.ndarray) -> int:
        pass

    @abstractmethod
    def __call__(self, observation: np.ndarray) -> int:
        pass