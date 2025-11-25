# Action-based Snake Game

In a regular Snake game, the snake automatically moves every frame, whether you take an action or not. In this game, movement is action-based: the snake only moves when you perform an action. Each action advances the game state, rather than the snake moving continuously on its own.

But there’s a twist. Instead of apples, there are monsters. These monsters move around the map and try to avoid you. Their behavior can be partly random, but most of the time they actively attempt to stay away from the snake.

As the snake, you can also detach segments of your body and turn them into walls. This takes the last segment of your body, and converts it into a wall. However, be careful, those walls become permanent obstacles for both you and the monsters, and running into them will still kill you.

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
- `--no-render`: Disable rendering (faster training)
- `--max-steps N`: Maximum steps per episode (default: 1000)

### Using with Your Own Agent

To hook in your own RL agent, modify `main.py` or create your own script:

```python
from move_based_snake.game import MoveBasedSnakeGame

# Create game environment
game = MoveBasedSnakeGame(grid_width=20, grid_height=20, num_monsters=3)

# Reset and get initial observation
obs = game.reset()

# Game loop
while not game.done:
    # Get action from your agent
    action = your_agent.predict(obs)  # or your_agent.act(obs)
    
    # Step the environment
    obs, reward, done, info = game.step(action)
    
    # Train your agent here if needed
    # your_agent.train(obs, reward, done, ...)
```

The game supports 6 actions:
- `0`: UP
- `1`: RIGHT
- `2`: DOWN
- `3`: LEFT
- `4`: DETACH_TAIL (convert last segment to wall)
- `5`: NO_OP (do nothing, monsters still move)

The observation is a flattened grid (width × height) with values:
- `2.0`: Snake head
- `1.0`: Snake body
- `-1.0`: Monster
- `3.0`: Wall
- `0.0`: Empty cell

## Solution
The solution is implemented in the ```trainer/solution_model.py```