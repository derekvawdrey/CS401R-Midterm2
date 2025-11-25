"""
Main game environment for the move-based Snake game.
Supports both player control and agent control for RL training.
"""

from typing import List, Tuple, Set, Optional, Dict, Any
import random
import numpy as np

try:
    from .snake import Snake
    from .monsters.base_monster import Monster
except ImportError:
    from snake import Snake
    from monsters.base_monster import Monster

class MoveBasedSnakeGame:
    """
    Move-based Snake game environment.
    
    Actions:
        0: UP
        1: RIGHT
        2: DOWN
        3: LEFT
        4: DETACH_TAIL (convert last segment to wall)
        5: NO_OP (do nothing, but monsters still move)
    """
    
    ACTIONS = {
        0: Snake.UP,
        1: Snake.RIGHT,
        2: Snake.DOWN,
        3: Snake.LEFT,
        4: 'DETACH_TAIL',
        5: 'NO_OP'
    }
    
    def __init__(self, grid_width: int = 20, grid_height: int = 20,
                 num_monsters: int = 3, snake_start_length: int = 3,
                 monster_avoidance_prob: float = 0.8):
        """
        Initialize the game.
        
        Args:
            grid_width: Width of the game grid
            grid_height: Height of the game grid
            num_monsters: Number of monsters in the game
            snake_start_length: Initial length of the snake
            monster_avoidance_prob: Probability monsters use avoidance behavior
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_monsters = num_monsters
        self.snake_start_length = snake_start_length
        self.monster_avoidance_prob = monster_avoidance_prob
        
        self.snake: Optional[Snake] = None
        self.monsters: List[Monster] = []
        self.walls: Set[Tuple[int, int]] = set()
        self.wall_animations: Dict[Tuple[int, int], int] = {}  # Track wall animations: {pos: frame}
        self.steps = 0
        self.done = False
        self.score = 0
        
        # Action space size
        self.action_space_size = 6
        
    def reset(self) -> np.ndarray:
        """
        Reset the game to initial state.
        
        Returns:
            Initial observation (state representation)
        """
        self.steps = 0
        self.done = False
        self.score = 0
        self.walls = set()
        self.wall_animations = {}
        
        # Initialize snake in center
        start_pos = (self.grid_width // 2, self.grid_height // 2)
        self.snake = Snake(start_pos, self.snake_start_length, Snake.RIGHT)
        
        # Initialize monsters at random positions
        self.monsters = []
        occupied = self.snake.get_all_positions()
        
        # Available monster types
        monster_types = [
            "Blinded Grimlock", "Bloodshot Eye", "Brawny Ogre", "Crimson Slaad",
            "Crushing Cyclops", "Death Slime", "Fungal Myconid", "Humongous Ettin",
            "Murky Slaad", "Ochre Jelly", "Ocular Watcher", "Red Cap",
            "Shrieker Mushroom", "Stone Troll", "Swamp Troll"
        ]
        
        for i in range(self.num_monsters):
            pos = self._get_random_empty_position(occupied)
            if pos is not None:
                monster_type = monster_types[i % len(monster_types)]  # Cycle through types
                monster = Monster(pos, self.monster_avoidance_prob, monster_type)
                self.monsters.append(monster)
                occupied.add(pos)
        
        return self._get_observation()
    
    def _spawn_monster(self):
        """Spawn a new monster at a random empty position."""
        # Get all occupied positions
        occupied = self.snake.get_all_positions()
        occupied.update(self.walls)
        occupied.update({m.get_position() for m in self.monsters})
        
        # Available monster types
        monster_types = [
            "Blinded Grimlock", "Bloodshot Eye", "Brawny Ogre", "Crimson Slaad",
            "Crushing Cyclops", "Death Slime", "Fungal Myconid", "Humongous Ettin",
            "Murky Slaad", "Ochre Jelly", "Ocular Watcher", "Red Cap",
            "Shrieker Mushroom", "Stone Troll", "Swamp Troll"
        ]
        
        # Find an empty position
        pos = self._get_random_empty_position(occupied)
        if pos is not None:
            # Randomly select a monster type
            monster_type = random.choice(monster_types)
            monster = Monster(pos, self.monster_avoidance_prob, monster_type)
            self.monsters.append(monster)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Perform an action and advance the game state.
        
        Args:
            action: Action to take (0-5)
            
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
        
        # Handle action
        if action == 4:  # DETACH_TAIL
            wall_pos = self.snake.detach_tail()
            if wall_pos is not None:
                # Start wall animation (frame 0 of 9)
                self.wall_animations[wall_pos] = 0
                # Walls are added immediately but animation plays
                self.walls.add(wall_pos)
                reward = 2.0  # Reward for strategic wall placement
                self.score += 2  # +2 points for placing a wall
            else:
                reward = -0.1  # Penalty for trying to detach when not possible
        elif action == 5:  # NO_OP
            # Do nothing with snake, but monsters still move
            pass
        elif 0 <= action <= 3:  # Movement action
            direction = self.ACTIONS[action]
            self.snake.set_direction(direction)
            old_tail = self.snake.move(grow=False, 
                                      grid_width=self.grid_width, 
                                      grid_height=self.grid_height)
            
            # Check collision with walls/boundaries/self
            if self.snake.check_collision(self.walls, self.grid_width, self.grid_height):
                self.done = True
                reward = -10.0
                info['reason'] = 'collision'
                info['snake_length'] = len(self.snake.body)
                return self._get_observation(), reward, True, info
            
            # Check collision with monsters (eat them!)
            snake_head = self.snake.get_head()
            eaten_monsters = []
            for monster in self.monsters:
                if monster.get_position() == snake_head:
                    eaten_monsters.append(monster)
            
            # Eat monsters - snake grows and monsters are removed
            for monster in eaten_monsters:
                self.monsters.remove(monster)
                # Grow the snake by adding a segment at the tail
                self.snake.grow()
                reward += 5.0  # Reward for eating a monster
                self.score += 1
                
                # Spawn a new monster at a random empty position
                self._spawn_monster()
            
            if eaten_monsters:
                info['monsters_eaten'] = len(eaten_monsters)
        
        # Move monsters (they all move simultaneously based on current state)
        snake_head = self.snake.get_head()
        snake_body = self.snake.get_all_positions()
        monster_positions = {m.get_position() for m in self.monsters}
        
        # Calculate all monster moves first
        monster_moves = []
        for monster in self.monsters:
            direction = monster.choose_direction(
                snake_head, self.walls, snake_body, monster_positions,
                self.grid_width, self.grid_height
            )
            monster_moves.append((monster, direction))
        
        # Execute moves, preventing collisions between monsters
        # Track which positions will be occupied after all moves
        occupied_positions = set()  # Positions that will be occupied after moves
        
        # First pass: calculate all planned positions
        planned_positions = {}
        for monster, direction in monster_moves:
            if direction is not None:
                new_pos = (monster.pos[0] + direction[0], monster.pos[1] + direction[1])
                if new_pos not in planned_positions:
                    planned_positions[new_pos] = []
                planned_positions[new_pos].append(monster)
            else:
                # Monster staying in place
                if monster.pos not in planned_positions:
                    planned_positions[monster.pos] = []
                planned_positions[monster.pos].append(monster)
        
        # Second pass: execute moves, preventing collisions
        for monster, direction in monster_moves:
            if direction is not None:
                new_pos = (monster.pos[0] + direction[0], monster.pos[1] + direction[1])
                
                # Check if this position is already occupied or will be by another monster
                if new_pos in occupied_positions:
                    # Position already taken, monster stays in place
                    occupied_positions.add(monster.pos)  # Current position stays occupied
                    continue
                
                # Check if multiple monsters want this position
                if new_pos in planned_positions and len(planned_positions[new_pos]) > 1:
                    # Multiple monsters want same position - only first one in list moves
                    if planned_positions[new_pos][0] != monster:
                        # Not first monster, stay in place
                        occupied_positions.add(monster.pos)
                        continue
                
                # Safe to move
                occupied_positions.add(new_pos)
                monster.move(direction)
            else:
                # Monster staying in place
                occupied_positions.add(monster.pos)
        
        # Update wall animations
        completed_animations = []
        for wall_pos in list(self.wall_animations.keys()):
            self.wall_animations[wall_pos] += 1
            if self.wall_animations[wall_pos] >= 9:  # Animation complete (9 frames)
                completed_animations.append(wall_pos)
        for wall_pos in completed_animations:
            del self.wall_animations[wall_pos]
        
        # Small survival reward
        reward += 0.01
        
        info['snake_length'] = len(self.snake.body) if self.snake else 0
        
        return self._get_observation(), reward, self.done, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current game state as an observation.
        
        Returns:
            Flattened observation array representing the game state
        """
        # Create a grid representation
        grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        
        if self.snake is not None:
            # Snake body = 1.0, head = 2.0
            for i, pos in enumerate(self.snake.body):
                # Check bounds before accessing grid
                if 0 <= pos[0] < self.grid_width and 0 <= pos[1] < self.grid_height:
                    if i == 0:
                        grid[pos[1], pos[0]] = 2.0  # Head
                    else:
                        grid[pos[1], pos[0]] = 1.0  # Body
        
        # Monsters = -1.0
        for monster in self.monsters:
            pos = monster.get_position()
            if 0 <= pos[0] < self.grid_width and 0 <= pos[1] < self.grid_height:
                grid[pos[1], pos[0]] = -1.0
        
        # Walls = 3.0
        for wall_pos in self.walls:
            if 0 <= wall_pos[0] < self.grid_width and 0 <= wall_pos[1] < self.grid_height:
                grid[wall_pos[1], wall_pos[0]] = 3.0
        
        return grid.flatten()
    
    def _get_random_empty_position(self, occupied: Set[Tuple[int, int]], 
                                   max_attempts: int = 100) -> Optional[Tuple[int, int]]:
        """Get a random position that's not occupied."""
        attempts = 0
        while attempts < max_attempts:
            pos = (random.randint(0, self.grid_width - 1), 
                   random.randint(0, self.grid_height - 1))
            if pos not in occupied:
                return pos
            attempts += 1
        return None
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get current game state as a dictionary (useful for rendering)."""
        return {
            'snake': self.snake,
            'monsters': self.monsters,
            'walls': self.walls,
            'wall_animations': self.wall_animations,
            'grid_width': self.grid_width,
            'grid_height': self.grid_height,
            'steps': self.steps,
            'score': self.score,
            'done': self.done
        }

