"""
Training configuration for the Falling Objects game.

Modify the values below to change the rewards for different game events
and toggle training features.
All values are in float format.
"""

# Reward for colliding with walls, boundaries, or being hit by falling objects (game over)
COLLISION_REWARD = -10.0

# Reward for surviving each step (small positive reward to encourage survival)
# Try values like 0.01, 0.1, or 0.5 to encourage longer episodes
SURVIVAL_REWARD = 0.1

# Whether to include danger signals (left, forward, right) in the observation
ENABLE_DANGER_SIGNALS = True
