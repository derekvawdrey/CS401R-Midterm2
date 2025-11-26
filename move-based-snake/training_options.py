"""
Training configuration for the Falling Objects game.

Modify the values below to change the rewards for different game events
and toggle training features.
All values are in float format.
"""

# Reward for colliding with walls, boundaries, or being hit by falling objects (game over)
# Reduced from -10.0 to make learning easier (agent needs to survive fewer steps to offset)
COLLISION_REWARD = -5.0

# Reward for surviving each step (small positive reward to encourage survival)
# Increased to make survival more valuable relative to collision penalty
SURVIVAL_REWARD = 0.2

# Reward for avoiding danger (being far from falling objects landing soon)
# Positive value rewards moving away from danger
# Increased to give stronger signal for evasive actions
DANGER_AVOIDANCE_REWARD = 0.1

# Whether to include danger signals (left, forward, right) in the observation
ENABLE_DANGER_SIGNALS = True
