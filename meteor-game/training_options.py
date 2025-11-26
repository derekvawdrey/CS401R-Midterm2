"""
Training configuration for the Falling Objects game.

Modify the values below to change the rewards for different game events
and toggle training features.
All values are in float format.
"""

# ============================================================================
# BASIC REWARDS
# ============================================================================

# Reward for colliding with walls, boundaries, or being hit by falling objects (game over)
# Reduced from -10.0 to make learning easier (agent needs to survive fewer steps to offset)
COLLISION_REWARD = -5.0

# Reward for surviving each step (small positive reward to encourage survival)
# Increased significantly to make survival more valuable - agent needs strong incentive to learn the 4-step pattern
SURVIVAL_REWARD = 0.5

# Reward for avoiding danger (being far from falling objects landing soon)
# Positive value rewards moving away from danger
# Increased to give stronger signal for evasive actions
DANGER_AVOIDANCE_REWARD = 0.0

# ============================================================================
# ALTERNATIVE REWARD STRATEGIES
# ============================================================================

# 1. DISTANCE-BASED REWARDS
# Reward for maintaining safe distance from meteors
# Set to 0 to disable, or use a small positive value (e.g., 0.1)
DISTANCE_SAFETY_REWARD = 0.0  # Reward per step for being far from immediate dangers
MIN_SAFE_DISTANCE = 3  # Minimum distance (Manhattan) to be considered "safe"

# 2. PROGRESSIVE SURVIVAL REWARDS
# Increase survival reward over time to encourage longer episodes
# Set to 0 to disable, or use a small value (e.g., 0.001)
PROGRESSIVE_SURVIVAL_BONUS = 0.001  # Additional reward per step = steps * this_value

# 3. EFFICIENCY REWARDS
# Reward for minimal movement (staying still when safe)
# Set to 0 to disable, or use a small positive value (e.g., 0.05)
STAY_SAFE_REWARD = 0.0  # Reward for STAY action when in a safe position
UNNECESSARY_MOVEMENT_PENALTY = 0.0  # Small penalty for moving when safe to stay

# 4. DANGER-LEVEL REWARDS
# Different rewards based on how close danger is
IMMEDIATE_DANGER_AVOIDANCE = 0.0  # Reward for avoiding meteors landing in 1 step
NEAR_DANGER_AVOIDANCE = 0.0  # Reward for avoiding meteors landing in 2 steps
FAR_DANGER_AVOIDANCE = 0.0  # Reward for avoiding meteors landing in 3+ steps

# 5. MULTI-DANGER BONUS
# Bonus for successfully navigating multiple simultaneous dangers
MULTI_DANGER_BONUS = 0.0  # Bonus when avoiding 2+ meteors landing soon
MULTI_DANGER_THRESHOLD = 2  # Number of meteors needed to trigger bonus

# 6. TIME-BASED DECAY
# Survival rewards decay over time to encourage faster learning early on
# Set to 0 to disable, or use a small value (e.g., 0.0001)
SURVIVAL_DECAY_RATE = 0.0  # Survival reward multiplier decreases by this per step

# 7. PATTERN RECOGNITION REWARDS
# Reward for recognizing and avoiding danger patterns
PATTERN_RECOGNITION_BONUS = 0.0  # Bonus for avoiding player-targeted meteors (4-step pattern)

# 8. CENTER_POSITION_REWARD
# Reward for staying near center (encourages strategic positioning)
CENTER_POSITION_REWARD = 0.5  # Reward for being near grid center
CENTER_RADIUS = 5  # Distance from center to receive reward

# 9. BORDER_PENALTY
# Penalty for being near the border rows/columns
# This discourages the agent from staying near the edges of the map
BORDER_PENALTY = -1.0  # Penalty per step for being near border (negative value applies penalty, set to 0 to disable)
BORDER_PENALTY_DISTANCE = 1  # Number of cells away from border to apply penalty (1 = only on border, 2 = border + 1 cell in, etc.)

# ============================================================================
# OBSERVATION SETTINGS
# ============================================================================

# Whether to include danger signals (left, forward, right) in the observation
ENABLE_DANGER_SIGNALS = True

# Radius of local view around player sent to the agent
# This determines the size of the observation window (view_radius cells in each direction)
# Default: 4 gives a 9x9 window (2*4+1 = 9)
VIEW_RADIUS = 3
