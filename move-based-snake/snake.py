from typing import List, Tuple, Optional, Set
import random

class Snake:
    """Represents the snake in the game."""
    
    # Directions: UP, RIGHT, DOWN, LEFT
    UP = (0, -1)
    RIGHT = (1, 0)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    
    def __init__(self, start_pos: Tuple[int, int], start_length: int = 3, 
                 start_dir: Tuple[int, int] = RIGHT):
        self.body: List[Tuple[int, int]] = []
        self.direction = start_dir
        self.next_direction = start_dir
        
        # Initialize body
        for i in range(start_length):
            pos = (start_pos[0] - i * start_dir[0], start_pos[1] - i * start_dir[1])
            self.body.append(pos)
    
    def get_head(self) -> Tuple[int, int]:
        """Get the position of the snake's head."""
        return self.body[0]
    
    def get_tail(self) -> Tuple[int, int]:
        """Get the position of the snake's tail."""
        return self.body[-1]
    
    def set_direction(self, direction: Tuple[int, int]):
        """Set the direction for the next move."""
        # Prevent moving directly opposite direction
        if (direction[0] * -1, direction[1] * -1) != self.direction:
            self.next_direction = direction
    
    def move(self, grow: bool = False, grid_width: int = None, grid_height: int = None) -> Optional[Tuple[int, int]]:
        """
        Move the snake one step. Returns the old tail position if not growing.
        Wraps around boundaries if grid dimensions are provided.
        
        Args:
            grow: If True, the snake grows by one segment
            grid_width: Width of the grid (for wrapping)
            grid_height: Height of the grid (for wrapping)
            
        Returns:
            Old tail position if not growing, None otherwise
        """
        self.direction = self.next_direction
        head = self.get_head()
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        # Wrap around boundaries if grid dimensions provided
        if grid_width is not None and grid_height is not None:
            # Wrap coordinates (Python's modulo handles negative numbers correctly)
            new_head = (new_head[0] % grid_width, new_head[1] % grid_height)
        
        old_tail = None
        if not grow:
            old_tail = self.body.pop()
        
        self.body.insert(0, new_head)
        return old_tail
    
    def check_collision(self, walls: Set[Tuple[int, int]], 
                       grid_width: int = None, grid_height: int = None) -> bool:
        """
        Check if the snake has collided with walls or itself.
        Boundaries are NOT checked (snake wraps around).
        
        Args:
            walls: Set of wall positions
            grid_width: Width of the grid (optional, not used for boundaries)
            grid_height: Height of the grid (optional, not used for boundaries)
        
        Returns:
            True if collision detected, False otherwise
        """
        head = self.get_head()
        
        # Note: Boundaries are not checked - snake wraps around
        # (boundary checking removed to enable wrapping)
        
        # Check walls
        if head in walls:
            return True
        
        # Check self-collision
        if head in self.body[1:]:
            return True
        
        return False
    
    def can_detach_tail(self) -> bool:
        """Check if the snake has enough segments to detach the tail."""
        return len(self.body) > 1
    
    def detach_tail(self) -> Optional[Tuple[int, int]]:
        """
        Detach the tail segment and return its position as a wall.
        
        Returns:
            Position of the detached tail segment, or None if not possible
        """
        if not self.can_detach_tail():
            return None
        return self.body.pop()
    
    def grow(self):
        """Grow the snake by adding a segment at the tail (without moving)."""
        if len(self.body) > 0:
            tail = self.body[-1]
            self.body.append(tail)
    
    def get_all_positions(self) -> Set[Tuple[int, int]]:
        """Get all positions occupied by the snake."""
        return set(self.body)

