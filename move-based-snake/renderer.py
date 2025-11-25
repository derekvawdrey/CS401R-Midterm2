"""
Renderer for the move-based Snake game using pygame.
"""

import pygame
import os
from typing import Dict, Any, Optional
from pathlib import Path

class GameRenderer:
    """Handles rendering of the game using pygame."""
    
    def __init__(self, grid_width: int, grid_height: int, 
                 cell_size: int = 30, fps: int = 10):
        """
        Initialize the renderer.
        
        Args:
            grid_width: Width of the game grid
            grid_height: Height of the game grid
            cell_size: Size of each cell in pixels
            fps: Frames per second for rendering
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.fps = fps
        
        # Calculate window size
        self.window_width = grid_width * cell_size
        self.window_height = grid_height * cell_size
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Move-Based Snake Game")
        self.clock = pygame.time.Clock()
        
        # Colors
        self.BACKGROUND = (20, 20, 20)
        self.SNAKE_HEAD = (100, 200, 100)
        self.SNAKE_BODY = (50, 150, 50)
        self.MONSTER = (200, 50, 50)
        self.WALL = (100, 100, 100)
        self.GRID_LINES = (40, 40, 40)
        
        # Load assets if available
        # Assets are in the same directory as this file
        self.assets_path = Path(__file__).parent / "assets"
        self.snake_head_img = None
        self.snake_body_img = None
        self.snake_tail_img = None
        self.wall_img = None
        self._load_assets()
    
    def _load_assets(self):
        """Load image assets if they exist."""
        try:
            snake_path = self.assets_path / "snake"
            if (snake_path / "snake_head.png").exists():
                self.snake_head_img = pygame.image.load(snake_path / "snake_head.png")
                self.snake_head_img = pygame.transform.scale(
                    self.snake_head_img, (self.cell_size, self.cell_size)
                )
            if (snake_path / "snake_body.png").exists():
                self.snake_body_img = pygame.image.load(snake_path / "snake_body.png")
                self.snake_body_img = pygame.transform.scale(
                    self.snake_body_img, (self.cell_size, self.cell_size)
                )
            if (snake_path / "snake_tail.png").exists():
                self.snake_tail_img = pygame.image.load(snake_path / "snake_tail.png")
                self.snake_tail_img = pygame.transform.scale(
                    self.snake_tail_img, (self.cell_size, self.cell_size)
                )
            if (snake_path / "snake_wall_animation.png").exists():
                self.wall_img = pygame.image.load(snake_path / "snake_wall_animation.png")
                self.wall_img = pygame.transform.scale(
                    self.wall_img, (self.cell_size, self.cell_size)
                )
        except Exception as e:
            print(f"Warning: Could not load some assets: {e}")
            print("Using colored rectangles instead.")
    
    def render(self, state_dict: Dict[str, Any]):
        """
        Render the current game state.
        
        Args:
            state_dict: Dictionary containing game state
        """
        self.screen.fill(self.BACKGROUND)
        
        # Draw grid lines
        for x in range(0, self.window_width, self.cell_size):
            pygame.draw.line(self.screen, self.GRID_LINES, 
                           (x, 0), (x, self.window_height))
        for y in range(0, self.window_height, self.cell_size):
            pygame.draw.line(self.screen, self.GRID_LINES, 
                           (0, y), (self.window_width, y))
        
        # Draw walls
        for wall_pos in state_dict.get('walls', set()):
            self._draw_cell(wall_pos, self.WALL, self.wall_img)
        
        # Draw monsters
        for monster in state_dict.get('monsters', []):
            pos = monster.get_position()
            self._draw_cell(pos, self.MONSTER)
        
        # Draw snake
        snake = state_dict.get('snake')
        if snake and snake.body:
            # Draw body segments
            for i, pos in enumerate(snake.body):
                if i == 0:
                    # Head
                    self._draw_cell(pos, self.SNAKE_HEAD, self.snake_head_img)
                elif i == len(snake.body) - 1:
                    # Tail
                    self._draw_cell(pos, self.SNAKE_BODY, self.snake_tail_img)
                else:
                    # Body
                    self._draw_cell(pos, self.SNAKE_BODY, self.snake_body_img)
        
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def _draw_cell(self, pos: tuple, color: tuple, image: Optional[pygame.Surface] = None):
        """Draw a cell at the given grid position."""
        x = pos[0] * self.cell_size
        y = pos[1] * self.cell_size
        
        if image:
            self.screen.blit(image, (x, y))
        else:
            pygame.draw.rect(self.screen, color, 
                           (x + 1, y + 1, self.cell_size - 2, self.cell_size - 2))
    
    def handle_events(self) -> Optional[int]:
        """
        Handle pygame events and return action if any.
        
        Returns:
            Action code (0-5) or None if no action
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP or event.key == pygame.K_w:
                    return 0  # UP
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    return 1  # RIGHT
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    return 2  # DOWN
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    return 3  # LEFT
                elif event.key == pygame.K_SPACE:
                    return 4  # DETACH_TAIL
                elif event.key == pygame.K_r:
                    return 'reset'
        return None
    
    def close(self):
        """Close the renderer and pygame."""
        pygame.quit()

