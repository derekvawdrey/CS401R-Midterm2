# Falling Objects Avoidance Game

A reinforcement learning game where you (or an AI agent) play as a monster that must avoid falling objects. Objects fall from the sky, and you get a 2-step warning before they fall. When objects land, they create walls that you cannot walk through. You die if a falling object hits you.

## The Goal
Our goal is to train a RL model that will learn to play the game the best it can - avoiding falling objects and navigating around walls.

## Game Mechanics

- **Player**: You control a monster that can move in 4 directions (UP, RIGHT, DOWN, LEFT)
- **Falling Objects**: Objects appear at random positions on the board
- **Warning System**: Objects show a warning indicator at the landing position 2 steps before they land
- **Walls**: When objects land, they create permanent walls at that position
- **Collision**: You die if you're at a position when an object lands on it, or if you try to walk into a wall

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
- Arrow Keys or WASD: Move the player
- R: Reset the game
- Close window or ESC: Quit

**Options:**
- `--width N`: Set grid width (default: 20)
- `--height N`: Set grid height (default: 20)
- `--fall-prob FLOAT`: Probability of spawning a falling object each step (default: 0.1)
- `--wrap-boundaries`: Enable boundary wrapping (player wraps around screen edges)

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
- `--no-loop`: Run only one episode and exit

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
- `--fall-prob FLOAT`: Probability of spawning a falling object each step (default: 0.1)
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
- `--wrap-boundaries`: Enable boundary wrapping

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
from game import FallingObjectsGame

game = FallingObjectsGame(grid_width=20, grid_height=20, fall_probability=0.1)
agent = MyAgent(game)

train_any_agent(
    game=game,
    agent=agent,
    num_episodes=1000,
    model_path="my_agent.pth"
)
```

### Game Details

The game supports 5 actions:
- `0`: UP
- `1`: RIGHT
- `2`: DOWN
- `3`: LEFT
- `4`: STAY (no movement)

The observation is a flattened grid (width × height) with values:
- `2.0`: Player position
- `1.5`: Object landing this step
- `1.0`: Warning indicator (1 step before landing)
- `0.5`: Warning indicator (2 steps before landing)
- `-1.0`: Wall (landed object)
- `0.0`: Empty cell

Additional observation features (if enabled):
- Danger signals: 3 values indicating danger in left, forward, and right directions relative to player's facing direction

Rewards (these can be changed in `move-based-snake/training_options.py`):
- `-10.0`: Collision (game over - hit by falling object, walked into wall, or hit boundary)
- `0.1`: Survival reward per step

## Solution
The solution is implemented in `agents/solution_model.py`
