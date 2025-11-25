from typing import List, Tuple, Set, Optional
import random
import math

class Monster:
    """Represents a monster that tries to avoid the snake."""
    
    # Directions: UP, RIGHT, DOWN, LEFT
    UP = (0, -1)
    RIGHT = (1, 0)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    DIRECTIONS = [UP, RIGHT, DOWN, LEFT]
    
    def __init__(self, pos: Tuple[int, int], avoidance_prob: float = 0.8,
                 monster_type: str = None):
        """
        Initialize a monster.
        
        Args:
            pos: Initial position
            avoidance_prob: Probability of using avoidance behavior vs random (default 0.8)
            monster_type: Type/name of the monster (for sprite loading)
        """
        self.pos = pos
        self.avoidance_prob = avoidance_prob
        self.monster_type = monster_type
    
    def get_position(self) -> Tuple[int, int]:
        """Get the monster's current position."""
        return self.pos
    
    def distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def choose_direction(self, snake_head: Tuple[int, int], 
                        walls: Set[Tuple[int, int]],
                        snake_body: Set[Tuple[int, int]],
                        monsters: Set[Tuple[int, int]],
                        grid_width: int, grid_height: int) -> Optional[Tuple[int, int]]:
        """
        Choose a direction to move, trying to avoid the snake.
        
        Args:
            snake_head: Position of the snake's head
            walls: Set of wall positions
            snake_body: Set of all snake body positions
            monsters: Set of other monster positions
            grid_width: Width of the grid
            grid_height: Height of the grid
        """
        # Random behavior
        if random.random() > self.avoidance_prob:
            valid_dirs = self._get_valid_directions(walls, snake_body, monsters, 
                                                   grid_width, grid_height)
            return random.choice(valid_dirs) if valid_dirs else None
        
        # Avoidance behavior - move in direction that maximizes distance from snake
        valid_dirs = self._get_valid_directions(walls, snake_body, monsters,
                                               grid_width, grid_height)
        
        if not valid_dirs:
            return None
        
        # Score each direction by distance from snake after moving
        best_dir = None
        max_distance = -1
        
        for direction in valid_dirs:
            new_pos = (self.pos[0] + direction[0], self.pos[1] + direction[1])
            dist = self.distance(new_pos, snake_head)
            if dist > max_distance:
                max_distance = dist
                best_dir = direction
        
        return best_dir if best_dir is not None else random.choice(valid_dirs)
    
    def _get_valid_directions(self, walls: Set[Tuple[int, int]],
                             snake_body: Set[Tuple[int, int]],
                             monsters: Set[Tuple[int, int]],
                             grid_width: int, grid_height: int) -> List[Tuple[int, int]]:
        """Get list of valid directions the monster can move."""
        valid_dirs = []
        
        for direction in self.DIRECTIONS:
            new_pos = (self.pos[0] + direction[0], self.pos[1] + direction[1])
            
            # Check boundaries
            if new_pos[0] < 0 or new_pos[0] >= grid_width or \
               new_pos[1] < 0 or new_pos[1] >= grid_height:
                continue
            
            # Check walls
            if new_pos in walls:
                continue
            
            # Check snake body (monsters can overlap with snake body but prefer not to)
            # Check other monsters (monsters avoid each other)
            if new_pos in monsters:
                continue
            
            valid_dirs.append(direction)
        
        return valid_dirs
    
    def move(self, direction: Optional[Tuple[int, int]]):
        """Move the monster in the given direction."""
        if direction is not None:
            self.pos = (self.pos[0] + direction[0], self.pos[1] + direction[1])

