from base_trainer import BaseTrainer

class SolutionModel(BaseTrainer):
    def __init__(self, model, env):
        super().__init__(model, env)

    def train(self, num_episodes: int):
        pass