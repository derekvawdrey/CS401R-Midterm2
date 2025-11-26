"""
Main game environment for the falling objects avoidance game.
Supports both player control and agent control for RL training.
"""

from typing import List, Tuple, Set, Optional, Dict, Any
import random
import numpy as np

try:
    from .player import Player
    from .training_options import COLLISION_REWARD, SURVIVAL_REWARD, ENABLE_DANGER_SIGNALS
except ImportError:
    from player import Player
    from training_options import COLLISION_REWARD, SURVIVAL_REWARD, ENABLE_DANGER_SIGNALS

class FallingObjectsGame:
    """
    Falling objects avoidance game environment.
    
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
        
        return self._get_observation()
    
    def _spawn_falling_object(self):
        """Spawn a new falling object warning at a random position on the board every step."""
        # Choose a random position on the board
        x = random.randint(0, self.grid_width - 1)
        y = random.randint(0, self.grid_height - 1)
        
        # Don't spawn on top of existing walls or other falling objects
        occupied = self.walls.copy()
        occupied.update({(fx, fy) for fx, fy, _ in self.falling_objects})
        
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
        
        Args:
            action: Action to take (0-3)
            
        Returns:
            observation: New state observation
            reward: Reward for this step
            done: Whether the episode is done
            info: Additional information
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
            
            # Check if new position is a wall
            if new_pos in self.walls:
                # Hit a wall - game over
                self.done = True
                reward = COLLISION_REWARD
                info['reason'] = 'wall_collision'
                return self._get_observation(), reward, True, info
            
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
        
        for x, y, steps_until_fall in self.falling_objects:
            if steps_until_fall > 0:
                # Still in warning phase - countdown continues
                new_falling_objects.append((x, y, steps_until_fall - 1))
            else:
                # Object lands this step - check if it hits the player or creates a wall
                player_pos = self.player.get_position()
                
                # Check if player is at the landing position
                if player_pos[0] == x and player_pos[1] == y:
                    # Player is hit by falling object
                    self.done = True
                    reward = COLLISION_REWARD
                    info['reason'] = 'hit_by_falling_object'
                    return self._get_observation(), reward, True, info
                
                # Object lands and creates a wall at (x, y)
                self.walls.add((x, y))
        
        self.falling_objects = new_falling_objects
        
        # Check if player is standing on a wall (shouldn't happen, but safety check)
        if self.player.get_position() in self.walls:
            self.done = True
            reward = COLLISION_REWARD
            info['reason'] = 'standing_on_wall'
            return self._get_observation(), reward, True, info
        
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
        Check if a position is dangerous (wall or out of bounds).
        
        Args:
            pos: Position to check
            
        Returns:
            True if position is dangerous, False otherwise
        """
        # Check boundary collision if not wrapping
        if not self.wrap_boundaries:
            if pos[0] < 0 or pos[0] >= self.grid_width or pos[1] < 0 or pos[1] >= self.grid_height:
                return True
        
        # Check if position is a wall
        if pos in self.walls:
            return True
        
        # Check if a falling object will land at this position
        for x, y, steps_until_fall in self.falling_objects:
            if steps_until_fall == 0:
                # Object lands this step - check if it lands at this position
                if x == pos[0] and y == pos[1]:
                    return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current game state as an observation.
        
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
        
        # Walls = -1.0
        for wall_pos in self.walls:
            x, y = wall_pos
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                grid[y, x] = -1.0
        
        # Falling objects: 
        # - Warning (2 steps away) = 0.5 at landing position
        # - Warning (1 step away) = 1.0 at landing position
        # - Landing this step = 1.5 at landing position
        for x, y, steps_until_fall in self.falling_objects:
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                if steps_until_fall == 2:
                    # 2 steps warning
                    grid[y, x] = max(grid[y, x], 0.5)
                elif steps_until_fall == 1:
                    # 1 step warning
                    grid[y, x] = max(grid[y, x], 1.0)
                else:
                    # Landing this step
                    grid[y, x] = max(grid[y, x], 1.5)
        
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
            'grid_width': self.grid_width,
            'grid_height': self.grid_height,
            'steps': self.steps,
            'score': self.score,
            'done': self.done,
            'warning_steps': self.warning_steps
        }
