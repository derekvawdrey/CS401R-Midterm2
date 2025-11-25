"""
Renderer for the move-based Snake game using pygame.
Supports animations for wall creation and monsters.
"""

import pygame
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import time

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
        
        # Animation tracking
        self.animation_frame = 0
        self.last_animation_time = time.time()
        self.animation_speed = 0.15  # Seconds per frame
        
        # Load assets if available
        self.assets_path = Path(__file__).parent / "assets"
        self.snake_head_img = None
        self.snake_body_img = None
        self.snake_tail_img = None
        self.wall_animation_frames: List[pygame.Surface] = []
        self.wall_final_img: Optional[pygame.Surface] = None
        self.monster_sprites: Dict[str, List[pygame.Surface]] = {}
        
        self._load_assets()
    
    def _load_assets(self):
        """Load image assets if they exist."""
        try:
            snake_path = self.assets_path / "snake"
            
            # Load snake parts
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
            
            # Load wall animation spritesheet (3x3 grid, 9 frames)
            if (snake_path / "snake_wall_animation.png").exists():
                wall_sheet = pygame.image.load(snake_path / "snake_wall_animation.png")
                wall_sheet = wall_sheet.convert_alpha()  # Ensure alpha channel
                
                # Get actual dimensions and calculate frame size
                sheet_width = wall_sheet.get_width()
                sheet_height = wall_sheet.get_height()
                rows = 3
                cols = 3
                frame_size = min(sheet_width // cols, sheet_height // rows)
                
                # Extract each frame from the spritesheet in the order specified:
                # row 1 col 1, row 1 col 2, row 1 col 3, row 2 col 1, etc.
                for row in range(rows):
                    for col in range(cols):
                        x = col * frame_size
                        y = row * frame_size
                        frame = pygame.Surface((frame_size, frame_size), pygame.SRCALPHA)
                        frame.blit(wall_sheet, (0, 0), (x, y, frame_size, frame_size))
                        frame = pygame.transform.scale(frame, (self.cell_size, self.cell_size))
                        self.wall_animation_frames.append(frame)
                
                # The last frame becomes the final wall sprite
                if self.wall_animation_frames:
                    self.wall_final_img = self.wall_animation_frames[-1].copy()
            
            # Load monster spritesheets
            monsters_path = self.assets_path / "monsters"
            if monsters_path.exists():
                monster_types = [
                    "Blinded Grimlock", "Bloodshot Eye", "Brawny Ogre", "Crimson Slaad",
                    "Crushing Cyclops", "Death Slime", "Fungal Myconid", "Humongous Ettin",
                    "Murky Slaad", "Ochre Jelly", "Ocular Watcher", "Red Cap",
                    "Shrieker Mushroom", "Stone Troll", "Swamp Troll"
                ]
                
                for monster_type in monster_types:
                    monster_dir = monsters_path / monster_type
                    # Try PNG first, then GIF
                    png_file = monster_dir / f"{monster_type.replace(' ', '')}.png"
                    
                    if png_file.exists():
                        try:
                            sprite_sheet = pygame.image.load(png_file)
                            sprite_sheet = sprite_sheet.convert_alpha()  # Ensure alpha channel
                            
                            # Monster spritesheets are 64x16 (4 frames of 16x16)
                            sheet_width = sprite_sheet.get_width()
                            sheet_height = sprite_sheet.get_height()
                            frame_size = min(sheet_height, sheet_width // 4)  # Assume 4 frames
                            num_frames = sheet_width // frame_size
                            
                            frames = []
                            for i in range(num_frames):
                                x = i * frame_size
                                y = 0
                                frame = pygame.Surface((frame_size, frame_size), pygame.SRCALPHA)
                                frame.blit(sprite_sheet, (0, 0), (x, y, frame_size, frame_size))
                                frame = pygame.transform.scale(frame, (self.cell_size, self.cell_size))
                                frames.append(frame)
                            
                            if frames:
                                self.monster_sprites[monster_type] = frames
                        except Exception as e:
                            print(f"Warning: Could not load sprite for {monster_type}: {e}")
                
                if self.monster_sprites:
                    print(f"Loaded {len(self.monster_sprites)} monster sprite sets")
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
        
        # Update animation frame for monsters
        current_time = time.time()
        if current_time - self.last_animation_time >= self.animation_speed:
            self.animation_frame = (self.animation_frame + 1) % 1000  # Cycle through frames
            self.last_animation_time = current_time
        
        # Draw walls (with animations if in progress)
        wall_animations = state_dict.get('wall_animations', {})
        for wall_pos in state_dict.get('walls', set()):
            if wall_pos in wall_animations:
                # Draw animation frame
                frame_index = wall_animations[wall_pos]
                if frame_index < len(self.wall_animation_frames):
                    self._draw_cell(wall_pos, self.WALL, 
                                   self.wall_animation_frames[frame_index])
                else:
                    # Animation complete, draw final wall
                    self._draw_cell(wall_pos, self.WALL, self.wall_final_img)
            else:
                # Regular wall (animation complete)
                self._draw_cell(wall_pos, self.WALL, self.wall_final_img)
        
        # Draw monsters with animation
        for monster in state_dict.get('monsters', []):
            pos = monster.get_position()
            monster_type = getattr(monster, 'monster_type', None)
            
            if monster_type and monster_type in self.monster_sprites:
                frames = self.monster_sprites[monster_type]
                # Cycle through frames based on animation_frame
                frame_index = (self.animation_frame // 2) % len(frames)  # Slow down animation
                monster_img = frames[frame_index]
                self._draw_cell(pos, self.MONSTER, monster_img)
            else:
                # Fallback to colored rectangle
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
