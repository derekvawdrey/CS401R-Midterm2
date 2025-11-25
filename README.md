# Action-based Snake Game

In a regular Snake game, the snake automatically moves every frame, whether you take an action or not. In this game, movement is action-based: the snake only moves when you perform an action. Each action advances the game state, rather than the snake moving continuously on its own.

But there’s a twist. Instead of apples, there are monsters. These monsters move around the map and try to avoid you. Their behavior can be partly random, but most of the time they actively attempt to stay away from the snake.

## The Goal
Our goal is to train a RL model that will learn to play the game the best it can.

## Running the code

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Player Mode

Run the game in player mode (keyboard controls):

```bash
python main.py --mode player
```

**Controls:**
- Arrow Keys or WASD: Move the snake
- Space: Detach tail segment (convert to wall)
- R: Reset the game
- Close window or ESC: Quit

**Options:**
- `--width N`: Set grid width (default: 20)
- `--height N`: Set grid height (default: 20)
- `--monsters N`: Set number of monsters (default: 3)

### Agent Mode

Run the game with an agent:

```bash
python main.py --mode agent --random-agent
```

**Options:**
- `--random-agent`: Use a random agent for testing
- `--dqn-model PATH`: Load a trained DQN model
- `--no-render`: Disable rendering (faster training)
- `--max-steps N`: Maximum steps per episode (default: 1000)

### Training Agents

Train an agent using the general training script:

```bash
python train_agent.py --agent dqn --episodes 1000
```

**Training Options:**
- `--agent TYPE`: Agent type to train (default: dqn)
- `--episodes N`: Number of training episodes (default: 1000)
- `--width N`: Grid width (default: 20)
- `--height N`: Grid height (default: 20)
- `--monsters N`: Number of monsters (default: 3)
- `--render`: Render the game during training (slower)
- `--load-model PATH`: Load a pre-trained model to continue training
- `--save-model PATH`: Path to save the trained model (default: agent_model.pth)
- `--save-freq N`: Frequency (episodes) to save model (default: 100)
- `--learning-rate FLOAT`: Learning rate (default: 0.001)
- `--gamma FLOAT`: Discount factor (default: 0.99)
- `--epsilon FLOAT`: Initial exploration rate (default: 1.0)
- `--epsilon-min FLOAT`: Minimum exploration rate (default: 0.01)
- `--epsilon-decay FLOAT`: Epsilon decay rate (default: 0.995)
- `--batch-size N`: Batch size for training (default: 64)
- `--memory-size N`: Replay buffer size (default: 10000)
- `--max-steps N`: Maximum steps per episode (default: 1000)

**Example:**
```bash
# Train a DQN agent for 2000 episodes
python train_agent.py --agent dqn --episodes 2000 --save-model my_dqn.pth

# Continue training from a saved model
python train_agent.py --agent dqn --episodes 1000 --load-model my_dqn.pth --save-model my_dqn.pth

# Test a trained model
python main.py --mode agent --dqn-model my_dqn.pth
```

### Creating Your Own Agent

To create your own agent, inherit from `BaseAgent` in `agents/base_agent.py`:

```python
from agents.base_agent import BaseAgent
import numpy as np

class MyAgent(BaseAgent):
    def __init__(self, game):
        super().__init__(game)
        # Initialize your agent here
    
    def predict(self, observation: np.ndarray) -> int:
        """Predict the best action (no exploration)."""
        # Your prediction logic here
        return action
    
    def act(self, observation: np.ndarray, training: bool = True) -> int:
        """Choose an action (with exploration if training)."""
        # Your action selection logic here
        return action
    
    def __call__(self, observation: np.ndarray) -> int:
        """Make the agent callable."""
        return self.predict(observation)
    
    # Optional methods for training:
    # def remember(self, state, action, reward, next_state, done): ...
    # def train_step(self) -> Optional[float]: ...
    # def save(self, filepath: str): ...
    # def load(self, filepath: str): ...
```

Then train it using the general training function:

```python
from agents.trainer import train_any_agent
from game import MoveBasedSnakeGame

game = MoveBasedSnakeGame(grid_width=20, grid_height=20, num_monsters=3)
agent = MyAgent(game)

train_any_agent(
    game=game,
    agent=agent,
    num_episodes=1000,
    model_path="my_agent.pth"
)
```

### Game Details

The game supports 4 actions:
- `0`: UP
- `1`: RIGHT
- `2`: DOWN
- `3`: LEFT

The observation is a flattened grid (width × height) with values:
- `2.0`: Snake head
- `1.0`: Snake body
- `-1.0`: Monster
- `0.0`: Empty cell

Rewards (these can be changed in rewards.py):
- `-10.0`: Collision (game over)
- `5.0`: Eating a monster
- `0.0`: Survival reward per step

## Solution
The solution is implemented in `agents/solution_model.py`