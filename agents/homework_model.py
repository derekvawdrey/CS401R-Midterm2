from base_agent import BaseAgent

class HomeworkAgent(BaseAgent):
    def __init__(self, model, env):
        super().__init__(model, env)

    def train(self, num_episodes: int):
        pass

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