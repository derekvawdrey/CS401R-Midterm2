"""
Renderer for the falling objects avoidance game using pygame.
"""

import pygame
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import time

class GameRenderer:
    """Handles rendering of the game using pygame."""
    
    def __init__(self, grid_width: int, grid_height: int, 
                 cell_size: int = 30, fps: int = 10, limit_fps: bool = True,
                 enable_sound_effects: bool = True):
        """
        Initialize the renderer.
        
        Args:
            grid_width: Width of the game grid
            grid_height: Height of the game grid
            cell_size: Size of each cell in pixels
            fps: Frames per second for rendering
            limit_fps: Whether to limit FPS (False for maximum speed in agent mode)
            enable_sound_effects: Whether to play sound effects (False for agent mode)
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.fps = fps
        self.limit_fps = limit_fps
        self.enable_sound_effects = enable_sound_effects
        
        # Calculate window size
        self.window_width = grid_width * cell_size
        self.window_height = grid_height * cell_size
        
        # Initialize pygame and mixer for sound
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Falling Objects Game")
        self.clock = pygame.time.Clock()
        
        # Sound effects
        self.sound_game_over = None
        self.sound_background = None
        self.background_playing = False
        
        # Colors
        self.BACKGROUND = (20, 20, 20)
        self.PLAYER = (100, 200, 100)  # Green for player
        self.WALL = (100, 100, 100)  # Gray for walls
        self.FALLING_OBJECT = (255, 100, 100)  # Red for falling objects
        self.WARNING_2_STEPS = (255, 200, 100)  # Orange for 2-step warning
        self.WARNING_1_STEP = (255, 150, 50)  # Darker orange for 1-step warning
        self.GRID_LINES = (40, 40, 40)
        
        # Animation tracking
        self.animation_frame = 0
        self.last_animation_time = time.time()
        self.animation_speed = 0.15  # Seconds per frame
        
        # State tracking for sound effects
        self.prev_done = False
        
        # Load assets if available
        self.assets_path = Path(__file__).parent / "assets"
        self.player_img = None
        self.monster_sprites: Dict[str, List[pygame.Surface]] = {}
        
        self._load_assets()
        self._load_sounds()
    
    def _load_assets(self):
        """Load image assets if they exist."""
        try:
            # Try to load a monster sprite for the player
            monsters_path = self.assets_path / "monsters"
            if monsters_path.exists():
                # Try to load the first available monster type
                monster_types = [
                    "Blinded Grimlock", "Bloodshot Eye", "Brawny Ogre", "Crimson Slaad",
                    "Crushing Cyclops", "Death Slime", "Fungal Myconid", "Humongous Ettin",
                    "Murky Slaad", "Ochre Jelly", "Ocular Watcher", "Red Cap",
                    "Shrieker Mushroom", "Stone Troll", "Swamp Troll"
                ]
                
                for monster_type in monster_types:
                    monster_dir = monsters_path / monster_type
                    png_file = monster_dir / f"{monster_type.replace(' ', '')}.png"
                    
                    if png_file.exists():
                        try:
                            sprite_sheet = pygame.image.load(png_file)
                            sprite_sheet = sprite_sheet.convert_alpha()
                            
                            sheet_width = sprite_sheet.get_width()
                            sheet_height = sprite_sheet.get_height()
                            frame_size = min(sheet_height, sheet_width // 4)
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
                                # Use first monster as player sprite
                                if self.player_img is None:
                                    self.player_img = frames[0]
                        except Exception as e:
                            print(f"Warning: Could not load sprite for {monster_type}: {e}")
                
                if self.monster_sprites:
                    print(f"Loaded {len(self.monster_sprites)} monster sprite sets")
        except Exception as e:
            print(f"Warning: Could not load some assets: {e}")
            print("Using colored rectangles instead.")
    
    def _load_sounds(self):
        """Load sound effects and background music."""
        try:
            music_path = self.assets_path / "music"
            
            if (music_path / "game-over.mp3").exists():
                self.sound_game_over = pygame.mixer.Sound(str(music_path / "game-over.mp3"))
            if (music_path / "background.mp3").exists():
                self.sound_background = str(music_path / "background.mp3")
                try:
                    pygame.mixer.music.load(self.sound_background)
                    pygame.mixer.music.set_volume(0.5)
                    pygame.mixer.music.play(-1)
                    self.background_playing = True
                except Exception as e:
                    print(f"Warning: Could not play background music: {e}")
            
        except Exception as e:
            print(f"Warning: Could not load some sound files: {e}")
    
    def play_sound_game_over(self):
        """Play the game over sound effect."""
        if self.sound_game_over:
            try:
                pygame.mixer.music.stop()
                self.background_playing = False
                self.sound_game_over.play()
            except Exception:
                pass
    
    def _check_and_play_sounds(self, state_dict: Dict[str, Any], info: Dict[str, Any] = None):
        """Check game state for events and play appropriate sounds."""
        if not self.enable_sound_effects:
            return
        
        done = state_dict.get('done', False)
        
        # Check for game over
        if done and not self.prev_done:
            self.play_sound_game_over()
        
        self.prev_done = done
    
    def reset_state_tracking(self):
        """Reset state tracking (call when game resets)."""
        self.prev_done = False
        
        # Restart background music if it stopped
        if self.sound_background and not self.background_playing:
            try:
                pygame.mixer.music.load(self.sound_background)
                pygame.mixer.music.set_volume(0.5)
                pygame.mixer.music.play(-1)
                self.background_playing = True
            except Exception:
                pass
    
    def render(self, state_dict: Dict[str, Any], info: Dict[str, Any] = None, 
               skip_sound_check: bool = False):
        """
        Render the current game state.
        
        Args:
            state_dict: Dictionary containing game state
            info: Optional info dictionary from game.step()
            skip_sound_check: If True, skip sound checking for faster rendering
        """
        if not skip_sound_check:
            self._check_and_play_sounds(state_dict, info)
        
        self.screen.fill(self.BACKGROUND)
        
        # Draw grid lines
        for x in range(0, self.window_width, self.cell_size):
            pygame.draw.line(self.screen, self.GRID_LINES, 
                           (x, 0), (x, self.window_height))
        for y in range(0, self.window_height, self.cell_size):
            pygame.draw.line(self.screen, self.GRID_LINES, 
                           (0, y), (self.window_width, y))
        
        # Draw walls
        walls = state_dict.get('walls', set())
        for wall_pos in walls:
            self._draw_cell(wall_pos, self.WALL)
        
        # Draw falling objects and warnings
        falling_objects = state_dict.get('falling_objects', [])
        warning_steps = state_dict.get('warning_steps', 2)
        
        for x, y, steps_until_fall in falling_objects:
            if steps_until_fall == 2:
                # 2 step warning - show at landing position
                self._draw_cell((x, y), self.WARNING_2_STEPS)
            elif steps_until_fall == 1:
                # 1 step warning - show at landing position
                self._draw_cell((x, y), self.WARNING_1_STEP)
            else:
                # Landing this step - show at landing position
                self._draw_cell((x, y), self.FALLING_OBJECT)
        
        # Draw player
        player = state_dict.get('player')
        if player:
            pos = player.get_position()
            self._draw_cell(pos, self.PLAYER, self.player_img)
        
        pygame.display.flip()
        if hasattr(self, 'limit_fps') and self.limit_fps:
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
            Action code (0-3) or None if no action
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
                elif event.key == pygame.K_r:
                    return 'reset'
        return None
    
    def render_game_over(self, score: int, steps: int):
        """
        Render a game over screen.
        
        Args:
            score: Final score
            steps: Number of steps taken
        """
        state_dict = {'done': True}
        self._check_and_play_sounds(state_dict)
        
        # Darken the screen with a semi-transparent overlay
        overlay = pygame.Surface((self.window_width, self.window_height))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # Game Over text
        try:
            font_large = pygame.font.Font(None, 72)
            font_medium = pygame.font.Font(None, 36)
            font_small = pygame.font.Font(None, 24)
        except:
            font_large = pygame.font.SysFont('Arial', 72)
            font_medium = pygame.font.SysFont('Arial', 36)
            font_small = pygame.font.SysFont('Arial', 24)
        
        # Title
        game_over_text = font_large.render("GAME OVER", True, (255, 0, 0))
        game_over_rect = game_over_text.get_rect(center=(self.window_width // 2, self.window_height // 2 - 80))
        self.screen.blit(game_over_text, game_over_rect)
        
        # Stats
        score_text = font_medium.render(f"Score: {score}", True, (255, 255, 255))
        score_rect = score_text.get_rect(center=(self.window_width // 2, self.window_height // 2 - 20))
        self.screen.blit(score_text, score_rect)
        
        steps_text = font_medium.render(f"Steps: {steps}", True, (255, 255, 255))
        steps_rect = steps_text.get_rect(center=(self.window_width // 2, self.window_height // 2 + 20))
        self.screen.blit(steps_text, steps_rect)
        
        # Instructions
        restart_text = font_small.render("Press R to restart or ESC to quit", True, (200, 200, 200))
        restart_rect = restart_text.get_rect(center=(self.window_width // 2, self.window_height // 2 + 80))
        self.screen.blit(restart_text, restart_rect)
        
        pygame.display.flip()
    
    def close(self):
        """Close the renderer and pygame."""
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        pygame.quit()
