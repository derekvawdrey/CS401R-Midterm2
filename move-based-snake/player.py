from typing import Tuple

class Player:
    """Represents the player (monster) in the game."""
    
    # Directions: UP, RIGHT, DOWN, LEFT
    UP = (0, -1)
    RIGHT = (1, 0)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    
    def __init__(self, start_pos: Tuple[int, int], start_dir: Tuple[int, int] = RIGHT):
        self.pos = start_pos
        self.direction = start_dir
        self.next_direction = start_dir
    
    def get_position(self) -> Tuple[int, int]:
        """Get the player's current position."""
        return self.pos
    
    def set_direction(self, direction: Tuple[int, int]):
        """Set the direction for the next move."""
        # Allow any direction change (no snake-like restrictions)
        self.next_direction = direction
    
    def move(self, grid_width: int, grid_height: int, 
             wrap_boundaries: bool = False) -> bool:
        """
        Move the player one step.
        
        Args:
            grid_width: Width of the grid
            grid_height: Height of the grid
            wrap_boundaries: If True, wrap around boundaries
            
        Returns:
            True if move was successful, False if out of bounds (when not wrapping)
        """
        self.direction = self.next_direction
        new_pos = (self.pos[0] + self.direction[0], self.pos[1] + self.direction[1])
        
        # Wrap around boundaries or check bounds
        if wrap_boundaries:
            new_pos = (new_pos[0] % grid_width, new_pos[1] % grid_height)
        else:
            # Check boundaries
            if new_pos[0] < 0 or new_pos[0] >= grid_width or \
               new_pos[1] < 0 or new_pos[1] >= grid_height:
                return False
        
        self.pos = new_pos
        return True
    
    def check_collision(self, grid_width: int, grid_height: int,
                       wrap_boundaries: bool = False) -> bool:
        """
        Check if the player is out of bounds (when not wrapping).
        
        Args:
            grid_width: Width of the grid
            grid_height: Height of the grid
            wrap_boundaries: If False, check boundary collisions
            
        Returns:
            True if out of bounds, False otherwise
        """
        if not wrap_boundaries:
            if self.pos[0] < 0 or self.pos[0] >= grid_width or \
               self.pos[1] < 0 or self.pos[1] >= grid_height:
                return True
        return False

