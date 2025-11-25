"""
Training configuration for the Snake game.

Modify the values below to change the rewards for different game events
and toggle training features.
All values are in float format.
"""

# Reward for colliding with walls, boundaries, or self (game over)
COLLISION_REWARD = -50.0

# Reward for eating a monster
EAT_MONSTER_REWARD = 5.0

# Reward for surviving each step (small positive reward to encourage survival)
SURVIVAL_REWARD = 0.0

# Whether to include danger signals (left, forward, right) in the observation
ENABLE_DANGER_SIGNALS = True

