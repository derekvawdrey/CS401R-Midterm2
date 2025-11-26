"""
Main game environment for the falling objects avoidance game.
Supports both player control and agent control for RL training.
"""

from typing import List, Tuple, Set, Optional, Dict, Any
import random
import numpy as np

try:
    from .player import Player
    from .training_options import (
        COLLISION_REWARD, SURVIVAL_REWARD, ENABLE_DANGER_SIGNALS, DANGER_AVOIDANCE_REWARD,
        DISTANCE_SAFETY_REWARD, MIN_SAFE_DISTANCE, PROGRESSIVE_SURVIVAL_BONUS,
        STAY_SAFE_REWARD, UNNECESSARY_MOVEMENT_PENALTY, IMMEDIATE_DANGER_AVOIDANCE,
        NEAR_DANGER_AVOIDANCE, FAR_DANGER_AVOIDANCE, MULTI_DANGER_BONUS,
        MULTI_DANGER_THRESHOLD, SURVIVAL_DECAY_RATE, PATTERN_RECOGNITION_BONUS,
        CENTER_POSITION_REWARD, CENTER_RADIUS
    )
except ImportError:
    from player import Player
    from training_options import (
        COLLISION_REWARD, SURVIVAL_REWARD, ENABLE_DANGER_SIGNALS, DANGER_AVOIDANCE_REWARD,
        DISTANCE_SAFETY_REWARD, MIN_SAFE_DISTANCE, PROGRESSIVE_SURVIVAL_BONUS,
        STAY_SAFE_REWARD, UNNECESSARY_MOVEMENT_PENALTY, IMMEDIATE_DANGER_AVOIDANCE,
        NEAR_DANGER_AVOIDANCE, FAR_DANGER_AVOIDANCE, MULTI_DANGER_BONUS,
        MULTI_DANGER_THRESHOLD, SURVIVAL_DECAY_RATE, PATTERN_RECOGNITION_BONUS,
        CENTER_POSITION_REWARD, CENTER_RADIUS
    )

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
                 fall_probability: float = 0.1, warning_steps: int = 3,
                 enable_danger_signals: bool = None, wrap_boundaries: bool = False,
                 player_target_probability: float = 0.5, view_radius: int = 4):
        """
        Initialize the game.
        
        Args:
            grid_width: Width of the game grid
            grid_height: Height of the game grid
            fall_probability: Probability of spawning a new falling object each step (not used if spawn_every_n_steps is set)
            warning_steps: Number of steps before an object falls (default: 3)
            enable_danger_signals: Whether to include danger signals in observation.
                                  If None, uses ENABLE_DANGER_SIGNALS from training_options
            wrap_boundaries: Whether player wraps around boundaries (default: False)
            player_target_probability: Probability of spawning a bomb on player every 4 steps (default: 0.5)
            view_radius: Radius of local view around player (default: 4, gives 9x9 window)
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.fall_probability = fall_probability
        self.warning_steps = warning_steps
        self.spawn_every_n_steps = 2  # Spawn a new bomb every 2 steps
        self.player_target_probability = player_target_probability
        self.enable_danger_signals = enable_danger_signals if enable_danger_signals is not None else ENABLE_DANGER_SIGNALS
        self.wrap_boundaries = wrap_boundaries
        self.view_radius = view_radius
        
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
    
    def _is_position_in_danger(self, x: int, y: int) -> bool:
        """
        Check if a position is in danger (in explosion radius of falling or exploding bombs).
        
        Args:
            x: X coordinate to check
            y: Y coordinate to check
            
        Returns:
            True if position is in danger zone, False otherwise
        """
        # Check if position is in active explosion
        if (x, y) in self.active_explosions:
            return True
        
        # Check if position is in explosion radius of any falling bomb
        for fx, fy, steps_until_fall in self.falling_objects:
            # Check explosion radius (3x3) for bombs that will explode soon
            if steps_until_fall <= 1:  # Bomb will explode this step or next
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        explosion_x = fx + dx
                        explosion_y = fy + dy
                        if explosion_x == x and explosion_y == y:
                            return True
        
        return False
    
    def _spawn_falling_object(self):
        """
        Spawn a new meteor warning at a random position.
        """
        # Choose a random position on the board
        x = random.randint(0, self.grid_width - 1)
        y = random.randint(0, self.grid_height - 1)
        
        # Don't spawn on top of existing walls, other falling objects, or active explosions
        occupied = self.walls.copy()
        occupied.update({(fx, fy) for fx, fy, _ in self.falling_objects})
        # Also check active explosions (bombs currently exploding)
        occupied.update(self.active_explosions.keys())
        
        # Try to find an empty position (with more attempts since we need one every step)
        attempts = 0
        max_attempts = 50  # More attempts since we must find a position
        while ((x, y) in occupied or self._is_position_in_danger(x, y)) and attempts < max_attempts:
            x = random.randint(0, self.grid_width - 1)
            y = random.randint(0, self.grid_height - 1)
            attempts += 1
        
        # Add if we found a valid position (or if board is mostly full, add anyway)
        if ((x, y) not in occupied and not self._is_position_in_danger(x, y)) or attempts >= max_attempts:
            # Add with warning_steps countdown (object will land at this position)
            self.falling_objects.append((x, y, self.warning_steps))
    
    def _spawn_on_player(self):
        """
        Spawn a new meteor warning at the player's current position.
        """
        if self.player is not None:
            player_pos = self.player.get_position()
            x, y = player_pos[0], player_pos[1]
            
            # Don't spawn on top of existing walls, other falling objects, or active explosions
            occupied = self.walls.copy()
            occupied.update({(fx, fy) for fx, fy, _ in self.falling_objects})
            # Also check active explosions (bombs currently exploding)
            occupied.update(self.active_explosions.keys())
            
            # Only spawn if position is not already occupied and not in danger
            if (x, y) not in occupied and not self._is_position_in_danger(x, y):
                # Add with warning_steps countdown (object will land at this position)
                self.falling_objects.append((x, y, self.warning_steps))
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Perform an action and advance the game state.
        
        Each step:
        1. Player moves (if action is movement)
        2. Every 4 steps, there's a chance a new meteor spawns on player's position
        3. Every 2 steps, a random meteor spawns at a random position
        4. Existing meteors countdown (warnings decrease)
        5. Meteors that reach 0 explode in 3×3 radius (after 3 warning steps)
        6. Explosion effects fade over 2 steps
        
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
                # Hit boundary - player stays in place, game continues
                pass
        
        # Update falling objects
        # Every 4 steps, there's a chance to spawn a bomb on the player
        if self.steps % 4 == 0:
            if random.random() < self.player_target_probability:
                self._spawn_on_player()
        
        # Also spawn random bombs from the sky periodically
        # Spawn a random bomb every 2 steps (independent of player-targeted bombs)
        if self.steps % 2 == 0:
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
                if steps_until_fall <= 3:  # Object landing soon (within 3 steps)
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
        
        # ====================================================================
        # ALTERNATIVE REWARD STRATEGIES
        # ====================================================================
        
        # 1. DISTANCE-BASED SAFETY REWARD
        if DISTANCE_SAFETY_REWARD > 0:
            immediate_dangers = [obj for obj in self.falling_objects if obj[2] <= 1]
            if immediate_dangers:
                min_dist = min([abs(player_pos[0] - x) + abs(player_pos[1] - y) 
                               for x, y, _ in immediate_dangers])
                if min_dist >= MIN_SAFE_DISTANCE:
                    reward += DISTANCE_SAFETY_REWARD
        
        # 2. PROGRESSIVE SURVIVAL BONUS
        if PROGRESSIVE_SURVIVAL_BONUS > 0:
            reward += self.steps * PROGRESSIVE_SURVIVAL_BONUS
        
        # 3. EFFICIENCY REWARDS (STAY when safe, penalty for unnecessary movement)
        if STAY_SAFE_REWARD > 0 or UNNECESSARY_MOVEMENT_PENALTY > 0:
            immediate_dangers = [obj for obj in self.falling_objects if obj[2] <= 1]
            is_safe = True
            if immediate_dangers:
                min_dist = min([abs(player_pos[0] - x) + abs(player_pos[1] - y) 
                               for x, y, _ in immediate_dangers])
                is_safe = min_dist >= MIN_SAFE_DISTANCE
            
            if action == 4:  # STAY action
                if is_safe and STAY_SAFE_REWARD > 0:
                    reward += STAY_SAFE_REWARD
            elif is_safe and UNNECESSARY_MOVEMENT_PENALTY > 0:
                # Moved when safe to stay
                reward -= UNNECESSARY_MOVEMENT_PENALTY
        
        # 4. DANGER-LEVEL REWARDS (different rewards based on danger proximity)
        if IMMEDIATE_DANGER_AVOIDANCE > 0 or NEAR_DANGER_AVOIDANCE > 0 or FAR_DANGER_AVOIDANCE > 0:
            immediate_count = sum(1 for obj in self.falling_objects if obj[2] == 0)
            near_count = sum(1 for obj in self.falling_objects if obj[2] == 1)
            far_count = sum(1 for obj in self.falling_objects if obj[2] >= 2)
            
            # Check if player avoided immediate dangers
            if immediate_count > 0:
                immediate_dists = [abs(player_pos[0] - x) + abs(player_pos[1] - y) 
                                  for x, y, s in self.falling_objects if s == 0]
                if all(d >= 3 for d in immediate_dists):
                    reward += IMMEDIATE_DANGER_AVOIDANCE * immediate_count
            
            # Check if player avoided near dangers
            if near_count > 0:
                near_dists = [abs(player_pos[0] - x) + abs(player_pos[1] - y) 
                             for x, y, s in self.falling_objects if s == 1]
                if all(d >= 2 for d in near_dists):
                    reward += NEAR_DANGER_AVOIDANCE * near_count
            
            # Check if player avoided far dangers
            if far_count > 0:
                far_dists = [abs(player_pos[0] - x) + abs(player_pos[1] - y) 
                            for x, y, s in self.falling_objects if s >= 2]
                if all(d >= 2 for d in far_dists):
                    reward += FAR_DANGER_AVOIDANCE * far_count
        
        # 5. MULTI-DANGER BONUS (bonus for navigating multiple simultaneous dangers)
        if MULTI_DANGER_BONUS > 0:
            imminent_dangers = [obj for obj in self.falling_objects if obj[2] <= 1]
            if len(imminent_dangers) >= MULTI_DANGER_THRESHOLD:
                # Check if player avoided all of them
                all_safe = all(abs(player_pos[0] - x) + abs(player_pos[1] - y) >= 3 
                              for x, y, _ in imminent_dangers)
                if all_safe:
                    reward += MULTI_DANGER_BONUS * len(imminent_dangers)
        
        # 6. PATTERN RECOGNITION BONUS (bonus for avoiding player-targeted meteors)
        if PATTERN_RECOGNITION_BONUS > 0:
            # Check if this step had a player-targeted meteor (spawned every 4 steps)
            if self.steps % 4 == 0 and self.prev_player_pos is not None:
                # Check if any meteor is at previous player position (player-targeted)
                player_targeted = any(x == self.prev_player_pos[0] and y == self.prev_player_pos[1] 
                                     for x, y, _ in self.falling_objects)
                if player_targeted:
                    # Player moved away from targeted position
                    if player_pos != self.prev_player_pos:
                        reward += PATTERN_RECOGNITION_BONUS
        
        # 7. CENTER POSITION REWARD (reward for staying near center)
        if CENTER_POSITION_REWARD > 0:
            center_x, center_y = self.grid_width // 2, self.grid_height // 2
            dist_from_center = abs(player_pos[0] - center_x) + abs(player_pos[1] - center_y)
            if dist_from_center <= CENTER_RADIUS:
                reward += CENTER_POSITION_REWARD
        
        # Update previous state for next step
        self.prev_player_pos = player_pos
        self.prev_falling_objects = [(x, y, s) for x, y, s in self.falling_objects]
        
        # 8. SURVIVAL REWARD (with optional decay)
        survival_reward = SURVIVAL_REWARD
        if SURVIVAL_DECAY_RATE > 0:
            # Decay survival reward over time
            survival_reward *= max(0.1, 1.0 - (self.steps * SURVIVAL_DECAY_RATE))
        reward += survival_reward
        self.score = self.steps  # Score is just survival time
        
        info['walls_count'] = len(self.walls)
        info['falling_objects_count'] = len(self.falling_objects)
        
        return self._get_observation(), reward, self.done, info
    
    def _get_relative_directions(self, current_dir: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Get left, forward, right, and backward directions relative to the player's current direction.
        
        Args:
            current_dir: Current direction of the player
            
        Returns:
            Tuple of (left_dir, forward_dir, right_dir, backward_dir)
        """
        # Forward is the current direction
        forward = current_dir
        
        # Calculate left and right by rotating the direction vector
        # For (x, y): left = (y, -x), right = (-y, x)
        left = (current_dir[1], -current_dir[0])
        right = (-current_dir[1], current_dir[0])
        
        # Backward is opposite of forward
        backward = (-current_dir[0], -current_dir[1])
        
        return left, forward, right, backward
    
    def _check_danger_at_position(self, pos: Tuple[int, int], check_warning_steps: int = 2) -> bool:
        """
        Check if a position is dangerous (out of bounds, or meteor landing/exploding).
        
        Args:
            pos: Position to check
            check_warning_steps: How many steps ahead to check for meteors (default: 2)
            
        Returns:
            True if position is dangerous, False otherwise
        """
        # Check boundary collision if not wrapping
        if not self.wrap_boundaries:
            if pos[0] < 0 or pos[0] >= self.grid_width or pos[1] < 0 or pos[1] >= self.grid_height:
                return True
        
        # Check if a meteor will land/explode at this position within the warning window
        # Explosion radius: check center and adjacent cells
        for x, y, steps_until_fall in self.falling_objects:
            if steps_until_fall <= check_warning_steps:  # Meteor lands/explodes within warning window
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
        Get current game state as an observation (local view around player).
        
        Observation values:
        - 2.0: Player position (center of local view)
        - 1.5: Meteor exploding this step (center)
        - 1.0: Meteor landing in 1 step (center)
        - 0.8: Explosion danger zone (adjacent to meteor landing in 1 step)
        - 0.5: Meteor landing in 2 steps (warning)
        - 0.3: Meteor landing in 3 steps (early warning)
        - -0.5 to -0.8: Active explosion effect (fading)
        - 0.0: Empty cell
        
        Also includes:
        - 1.0 or 0.0: in_danger flag (1.0 if player is currently in danger zone)
        
        If danger signals enabled, also includes 4 values for danger in
        left, forward, right, and backward directions relative to player's facing.
        
        Returns:
            Flattened observation array representing the local view around player
        """
        if self.player is None:
            # Return empty observation if no player
            view_size = (2 * self.view_radius + 1) ** 2
            return np.zeros(view_size + 1 + (4 if self.enable_danger_signals else 0), dtype=np.float32)
        
        player_pos = self.player.get_position()
        px, py = player_pos[0], player_pos[1]
        
        # Create local view grid (view_radius cells in each direction)
        view_size = 2 * self.view_radius + 1
        local_view = np.zeros((view_size, view_size), dtype=np.float32)
        
        # Process all falling objects and their explosion radii
        for x, y, steps_until_fall in self.falling_objects:
            # Check if meteor center is in local view
            local_x = x - px + self.view_radius
            local_y = y - py + self.view_radius
            if 0 <= local_x < view_size and 0 <= local_y < view_size:
                if steps_until_fall >= 3:
                    local_view[local_y, local_x] = max(local_view[local_y, local_x], 0.3)
                elif steps_until_fall == 2:
                    local_view[local_y, local_x] = max(local_view[local_y, local_x], 0.5)
                elif steps_until_fall == 1:
                    local_view[local_y, local_x] = max(local_view[local_y, local_x], 1.0)
                else:  # steps_until_fall == 0, exploding this step
                    local_view[local_y, local_x] = max(local_view[local_y, local_x], 1.5)
            
            # Show explosion radius for meteors that will explode soon (within 1 step)
            if steps_until_fall <= 1:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        exp_x, exp_y = x + dx, y + dy
                        # Convert to local coordinates
                        local_exp_x = exp_x - px + self.view_radius
                        local_exp_y = exp_y - py + self.view_radius
                        # Check if in bounds of local view and global grid
                        if (0 <= local_exp_x < view_size and 0 <= local_exp_y < view_size and
                            0 <= exp_x < self.grid_width and 0 <= exp_y < self.grid_height):
                            local_view[local_exp_y, local_exp_x] = max(local_view[local_exp_y, local_exp_x], 0.8)
        
        # Process active explosions
        for (exp_x, exp_y), duration in self.active_explosions.items():
            # Convert to local coordinates
            local_x = exp_x - px + self.view_radius
            local_y = exp_y - py + self.view_radius
            # Check if in bounds of local view
            if 0 <= local_x < view_size and 0 <= local_y < view_size:
                # Don't overwrite player position
                if not (local_x == self.view_radius and local_y == self.view_radius):
                    intensity = -0.5 - (0.3 * (1 - duration / self.explosion_duration))
                    local_view[local_y, local_x] = min(local_view[local_y, local_x], intensity)
        
        # Mark player position last (center of view) so it's always visible
        local_view[self.view_radius, self.view_radius] = 2.0
        
        grid_flat = local_view.flatten()
        observation_parts = [grid_flat]
        
        # Add in_danger flag for player's current position
        in_danger = 0.0
        if self.player is not None:
            player_pos = self.player.get_position()
            if self._is_position_in_danger(player_pos[0], player_pos[1]):
                in_danger = 1.0
        observation_parts.append(np.array([in_danger], dtype=np.float32))
        
        # Get danger signals (left, forward, right, backward) if enabled
        if self.enable_danger_signals:
            danger_signals = np.zeros(4, dtype=np.float32)
            if self.player is not None:
                head_pos = self.player.get_position()
                current_dir = self.player.direction
                
                # Get relative directions
                left_dir, forward_dir, right_dir, backward_dir = self._get_relative_directions(current_dir)
                
                # Check danger in each direction (check up to 2 steps ahead for better warning)
                left_pos = (head_pos[0] + left_dir[0], head_pos[1] + left_dir[1])
                forward_pos = (head_pos[0] + forward_dir[0], head_pos[1] + forward_dir[1])
                right_pos = (head_pos[0] + right_dir[0], head_pos[1] + right_dir[1])
                backward_pos = (head_pos[0] + backward_dir[0], head_pos[1] + backward_dir[1])
                
                # Check danger at adjacent cells, looking 2 steps ahead for meteors
                danger_signals[0] = 1.0 if self._check_danger_at_position(left_pos, check_warning_steps=2) else 0.0
                danger_signals[1] = 1.0 if self._check_danger_at_position(forward_pos, check_warning_steps=2) else 0.0
                danger_signals[2] = 1.0 if self._check_danger_at_position(right_pos, check_warning_steps=2) else 0.0
                danger_signals[3] = 1.0 if self._check_danger_at_position(backward_pos, check_warning_steps=2) else 0.0
            
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
