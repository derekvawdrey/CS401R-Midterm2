from base_trainer import BaseTrainer

class HomeworkModel(BaseTrainer):
    def __init__(self, model, env):
        super().__init__(model, env)

    def train(self, num_episodes: int):
        pass