"""
Main game environment for the falling objects avoidance game.
Supports both player control and agent control for RL training.
"""

from typing import List, Tuple, Set, Optional, Dict, Any
import random
import numpy as np

try:
    from .player import Player
    from .training_options import COLLISION_REWARD, SURVIVAL_REWARD, ENABLE_DANGER_SIGNALS, DANGER_AVOIDANCE_REWARD
except ImportError:
    from player import Player
    from training_options import COLLISION_REWARD, SURVIVAL_REWARD, ENABLE_DANGER_SIGNALS, DANGER_AVOIDANCE_REWARD

class FallingObjectsGame:
    """
    Meteor explosion avoidance game environment.
    
    Meteors fall from the sky and explode in a 3×3 radius. The player must avoid
    the explosions to survive. Meteors show warnings 2 steps before landing.
    When meteors explode, they create a visible explosion effect that fades over
    2 steps, then disappear (no permanent obstacles).
    
    Actions:
        0: UP
        1: RIGHT
        2: DOWN
        3: LEFT
        4: STAY (no movement)
    """
    
    ACTIONS = {
        0: Player.UP,
        1: Player.RIGHT, 
        2: Player.DOWN,
        3: Player.LEFT,
        4: None,  # STAY - no movement
    }
    
    def __init__(self, grid_width: int = 20, grid_height: int = 20,
                 fall_probability: float = 0.1, warning_steps: int = 2,
                 enable_danger_signals: bool = None, wrap_boundaries: bool = False):
        """
        Initialize the game.
        
        Args:
            grid_width: Width of the game grid
            grid_height: Height of the game grid
            fall_probability: Probability of spawning a new falling object each step
            warning_steps: Number of steps before an object falls (default: 2)
            enable_danger_signals: Whether to include danger signals in observation.
                                  If None, uses ENABLE_DANGER_SIGNALS from training_options
            wrap_boundaries: Whether player wraps around boundaries (default: False)
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.fall_probability = fall_probability
        self.warning_steps = warning_steps
        self.enable_danger_signals = enable_danger_signals if enable_danger_signals is not None else ENABLE_DANGER_SIGNALS
        self.wrap_boundaries = wrap_boundaries
        
        self.player: Optional[Player] = None
        # Falling objects: list of (x, y, steps_until_fall)
        # (x, y): landing position where object will appear
        # steps_until_fall: countdown until object lands (0 = lands this step)
        self.falling_objects: List[Tuple[int, int, int]] = []  # (x, y, steps_until_fall)
        # Walls: set of (x, y) positions that are blocked
        self.walls: Set[Tuple[int, int]] = set()
        
        self.steps = 0
        self.done = False
        self.score = 0
        
        # Track previous position for reward shaping (movement away from danger)
        self.prev_player_pos: Optional[Tuple[int, int]] = None
        self.prev_falling_objects: List[Tuple[int, int, int]] = []
        
        # Track active explosions (for visual effect)
        # Format: {(x, y): steps_remaining} where steps_remaining decreases each step
        self.active_explosions: Dict[Tuple[int, int], int] = {}
        self.explosion_duration: int = 2  # Show explosion for 2 steps
        
        # Action space size (including STAY)
        self.action_space_size = 5
        
    def reset(self) -> np.ndarray:
        """
        Reset the game to initial state.
        
        Returns:
            Initial observation (state representation)
        """
        self.steps = 0
        self.done = False
        self.score = 0
        
        # Initialize player in center
        start_pos = (self.grid_width // 2, self.grid_height // 2)
        self.player = Player(start_pos, Player.RIGHT)
        
        # Clear falling objects and walls
        self.falling_objects = []
        self.walls = set()
        
        # Reset position tracking
        self.prev_player_pos = None
        self.prev_falling_objects = []
        
        # Clear explosions
        self.active_explosions = {}
        
        return self._get_observation()
    
    def _spawn_falling_object(self):
        """
        Spawn a new meteor warning every step. 
        
        20% chance to target the player's current position (encourages evasion),
        80% chance for random position.
        """
        # 20% chance to target the player's position, 80% chance for random position
        target_player = random.random() < 0.2
        
        if target_player and self.player is not None:
            # Target the player's current position
            player_pos = self.player.get_position()
            x, y = player_pos[0], player_pos[1]
        else:
            # Choose a random position on the board
            x = random.randint(0, self.grid_width - 1)
            y = random.randint(0, self.grid_height - 1)
        
        # Don't spawn on top of existing walls or other falling objects
        occupied = self.walls.copy()
        occupied.update({(fx, fy) for fx, fy, _ in self.falling_objects})
        
        # If targeting player but position is occupied, fall back to random
        if (x, y) in occupied and target_player:
            # Fall back to random position
            x = random.randint(0, self.grid_width - 1)
            y = random.randint(0, self.grid_height - 1)
        
        # Try to find an empty position (with more attempts since we need one every step)
        attempts = 0
        max_attempts = 50  # More attempts since we must find a position
        while (x, y) in occupied and attempts < max_attempts:
            x = random.randint(0, self.grid_width - 1)
            y = random.randint(0, self.grid_height - 1)
            attempts += 1
        
        # Add if we found a valid position (or if board is mostly full, add anyway)
        if (x, y) not in occupied or attempts >= max_attempts:
            # Add with warning_steps countdown (object will land at this position)
            self.falling_objects.append((x, y, self.warning_steps))
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Perform an action and advance the game state.
        
        Each step:
        1. Player moves (if action is movement)
        2. New meteor spawns (with probability fall_probability)
        3. Existing meteors countdown (warnings decrease)
        4. Meteors that reach 0 explode in 3×3 radius
        5. Explosion effects fade over 2 steps
        
        Args:
            action: Action to take (0-4: UP, RIGHT, DOWN, LEFT, STAY)
            
        Returns:
            observation: New state observation (flattened grid + danger signals)
            reward: Reward for this step (survival + avoidance bonuses)
            done: Whether the episode is done (player hit by explosion or boundary)
            info: Additional information (steps, score, reason for game over)
        """
        if self.done:
            return self._get_observation(), 0.0, True, {'steps': self.steps, 'score': self.score}
        
        self.steps += 1
        reward = 0.0
        info = {'steps': self.steps, 'score': self.score}
        
        # Handle player action
        if action == 4:  # STAY - no movement
            # Player stays in place, just continue
            pass
        elif 0 <= action <= 3:  # Movement action
            direction = self.ACTIONS[action]
            self.player.set_direction(direction)
            
            # Calculate where player wants to move
            new_pos = (self.player.pos[0] + direction[0], 
                      self.player.pos[1] + direction[1])
            
            # Note: No wall collision check - meteors explode and disappear, no permanent walls
            
            # Try to move
            moved = self.player.move(self.grid_width, self.grid_height, 
                                   wrap_boundaries=self.wrap_boundaries)
            if not moved and not self.wrap_boundaries:
                # Hit boundary
                self.done = True
                reward = COLLISION_REWARD
                info['reason'] = 'boundary_collision'
                return self._get_observation(), reward, True, info
        
        # Update falling objects
        # First, spawn new objects
        self._spawn_falling_object()
        
        # Update existing falling objects
        new_falling_objects = []
        explosion_positions = []  # Track explosion positions for this step
        
        for x, y, steps_until_fall in self.falling_objects:
            if steps_until_fall > 0:
                # Still in warning phase - countdown continues
                new_falling_objects.append((x, y, steps_until_fall - 1))
            else:
                # Meteor lands and explodes this step
                player_pos = self.player.get_position()
                
                # Check if player is at the landing position (direct hit)
                if player_pos[0] == x and player_pos[1] == y:
                    # Player is hit by meteor explosion
                    self.done = True
                    reward = COLLISION_REWARD
                    info['reason'] = 'hit_by_meteor_explosion'
                    return self._get_observation(), reward, True, info
                
                # Meteor explodes - create explosion effect (3x3 radius)
                explosion_positions.append((x, y))
                # Add explosion effect at center and all adjacent cells
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        explosion_x = x + dx
                        explosion_y = y + dy
                        # Check bounds
                        if 0 <= explosion_x < self.grid_width and 0 <= explosion_y < self.grid_height:
                            # Add explosion effect
                            self.active_explosions[(explosion_x, explosion_y)] = self.explosion_duration
                            
                            # Check if player is in explosion radius
                            if player_pos[0] == explosion_x and player_pos[1] == explosion_y:
                                # Player is in explosion radius
                                self.done = True
                                reward = COLLISION_REWARD
                                info['reason'] = 'hit_by_meteor_explosion_radius'
                                return self._get_observation(), reward, True, info
                
                # Meteor explodes and disappears (no wall created)
                # Walls are no longer created - meteors just explode and vanish
        
        self.falling_objects = new_falling_objects
        
        # Update active explosions (decrease duration, remove expired ones)
        expired_explosions = []
        for pos, duration in self.active_explosions.items():
            new_duration = duration - 1
            if new_duration <= 0:
                expired_explosions.append(pos)
            else:
                self.active_explosions[pos] = new_duration
        
        for pos in expired_explosions:
            del self.active_explosions[pos]
        
        # Note: No wall collision check needed anymore since walls aren't created
        
        # Reward shaping: reward moving AWAY from falling objects
        player_pos = self.player.get_position()
        avoidance_reward = 0.0
        
        # Reward for moving away from danger (if we have previous position)
        if self.prev_player_pos is not None:
            for x, y, steps_until_fall in self.falling_objects:
                if steps_until_fall <= 2:  # Object landing soon (within 2 steps)
                    # Calculate distance now and before
                    current_dist = abs(player_pos[0] - x) + abs(player_pos[1] - y)
                    prev_dist = abs(self.prev_player_pos[0] - x) + abs(self.prev_player_pos[1] - y)
                    
                    if current_dist > prev_dist:
                        # Moved AWAY from danger - reward this!
                        distance_gained = current_dist - prev_dist
                        avoidance_reward += DANGER_AVOIDANCE_REWARD * distance_gained
                    elif current_dist < prev_dist:
                        # Moved TOWARD danger - small penalty
                        distance_lost = prev_dist - current_dist
                        avoidance_reward -= DANGER_AVOIDANCE_REWARD * 0.5 * distance_lost
        
        # Also reward being at a safe distance from immediate dangers
        for x, y, steps_until_fall in self.falling_objects:
            if steps_until_fall <= 1:  # Object landing this step or next
                dist = abs(player_pos[0] - x) + abs(player_pos[1] - y)
                if dist >= 3:
                    # Safe distance - small bonus
                    avoidance_reward += DANGER_AVOIDANCE_REWARD * 0.3
        
        reward += avoidance_reward
        
        # Update previous state for next step
        self.prev_player_pos = player_pos
        self.prev_falling_objects = [(x, y, s) for x, y, s in self.falling_objects]
        
        # Small survival reward
        reward += SURVIVAL_REWARD
        self.score = self.steps  # Score is just survival time
        
        info['walls_count'] = len(self.walls)
        info['falling_objects_count'] = len(self.falling_objects)
        
        return self._get_observation(), reward, self.done, info
    
    def _get_relative_directions(self, current_dir: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Get left, forward, and right directions relative to the player's current direction.
        
        Args:
            current_dir: Current direction of the player
            
        Returns:
            Tuple of (left_dir, forward_dir, right_dir)
        """
        # Forward is the current direction
        forward = current_dir
        
        # Calculate left and right by rotating the direction vector
        # For (x, y): left = (y, -x), right = (-y, x)
        left = (current_dir[1], -current_dir[0])
        right = (-current_dir[1], current_dir[0])
        
        return left, forward, right
    
    def _check_danger_at_position(self, pos: Tuple[int, int]) -> bool:
        """
        Check if a position is dangerous (out of bounds, or meteor landing/exploding).
        
        Args:
            pos: Position to check
            
        Returns:
            True if position is dangerous, False otherwise
        """
        # Check boundary collision if not wrapping
        if not self.wrap_boundaries:
            if pos[0] < 0 or pos[0] >= self.grid_width or pos[1] < 0 or pos[1] >= self.grid_height:
                return True
        
        # Check if a meteor will land/explode at this position (this step or next step)
        # Explosion radius: check center and adjacent cells
        for x, y, steps_until_fall in self.falling_objects:
            if steps_until_fall <= 1:  # Meteor lands/explodes this step or next step
                # Check if position is in explosion radius (center + adjacent cells)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        explosion_x = x + dx
                        explosion_y = y + dy
                        if explosion_x == pos[0] and explosion_y == pos[1]:
                            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current game state as an observation.
        
        Observation values:
        - 2.0: Player position
        - 1.5: Meteor exploding this step (center)
        - 1.0: Meteor landing in 1 step (center)
        - 0.8: Explosion danger zone (adjacent to meteor landing in 1 step)
        - 0.5: Meteor landing in 2 steps (warning)
        - -0.5 to -0.8: Active explosion effect (fading)
        - 0.0: Empty cell
        
        If danger signals enabled, also includes 3 values for danger in
        left, forward, and right directions relative to player's facing.
        
        Returns:
            Flattened observation array representing the game state
        """
        # Create a grid representation
        grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        
        if self.player is not None:
            # Player = 2.0
            pos = self.player.get_position()
            if 0 <= pos[0] < self.grid_width and 0 <= pos[1] < self.grid_height:
                grid[pos[1], pos[0]] = 2.0
        
        # Meteors (no walls anymore - meteors explode and disappear):
        # - Warning (2 steps away) = 0.5 at landing position
        # - Warning (1 step away) = 1.0 at landing position  
        # - Landing/exploding this step = 1.5 at landing position
        # - Explosion radius (adjacent cells) = 0.8 to show danger zone
        for x, y, steps_until_fall in self.falling_objects:
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                if steps_until_fall == 2:
                    # 2 steps warning - show at landing position
                    grid[y, x] = max(grid[y, x], 0.5)
                elif steps_until_fall == 1:
                    # 1 step warning - show at landing position and explosion radius
                    grid[y, x] = max(grid[y, x], 1.0)
                    # Show explosion radius (adjacent cells)
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            exp_x, exp_y = x + dx, y + dy
                            if 0 <= exp_x < self.grid_width and 0 <= exp_y < self.grid_height:
                                grid[exp_y, exp_x] = max(grid[exp_y, exp_x], 0.8)
                else:
                    # Landing/exploding this step - show explosion radius
                    grid[y, x] = max(grid[y, x], 1.5)
                    # Show explosion radius (adjacent cells)
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            exp_x, exp_y = x + dx, y + dy
                            if 0 <= exp_x < self.grid_width and 0 <= exp_y < self.grid_height:
                                grid[exp_y, exp_x] = max(grid[exp_y, exp_x], 0.8)
        
        # Show active explosions (bomb effect)
        # Use -0.5 to -0.8 to show explosion effect (negative values for visual distinction)
        for (exp_x, exp_y), duration in self.active_explosions.items():
            if 0 <= exp_x < self.grid_width and 0 <= exp_y < self.grid_height:
                # Intensity decreases as explosion fades (2 -> 1 -> 0)
                # Use negative values to distinguish from meteors
                intensity = -0.5 - (0.3 * (1 - duration / self.explosion_duration))
                grid[exp_y, exp_x] = min(grid[exp_y, exp_x], intensity)  # Use min to ensure explosion shows
        
        grid_flat = grid.flatten()
        observation_parts = [grid_flat]
        
        # Get danger signals (left, forward, right) if enabled
        if self.enable_danger_signals:
            danger_signals = np.zeros(3, dtype=np.float32)
            if self.player is not None:
                head_pos = self.player.get_position()
                current_dir = self.player.direction
                
                # Get relative directions
                left_dir, forward_dir, right_dir = self._get_relative_directions(current_dir)
                
                # Check danger in each direction
                left_pos = (head_pos[0] + left_dir[0], head_pos[1] + left_dir[1])
                forward_pos = (head_pos[0] + forward_dir[0], head_pos[1] + forward_dir[1])
                right_pos = (head_pos[0] + right_dir[0], head_pos[1] + right_dir[1])
                
                danger_signals[0] = 1.0 if self._check_danger_at_position(left_pos) else 0.0
                danger_signals[1] = 1.0 if self._check_danger_at_position(forward_pos) else 0.0
                danger_signals[2] = 1.0 if self._check_danger_at_position(right_pos) else 0.0
            
            observation_parts.append(danger_signals)
        
        # Concatenate all observation parts
        return np.concatenate(observation_parts) if len(observation_parts) > 1 else observation_parts[0]
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get current game state as a dictionary (useful for rendering)."""
        return {
            'player': self.player,
            'falling_objects': self.falling_objects,
            'walls': self.walls,
            'active_explosions': self.active_explosions,
            'grid_width': self.grid_width,
            'grid_height': self.grid_height,
            'steps': self.steps,
            'score': self.score,
            'done': self.done,
            'warning_steps': self.warning_steps
        }
