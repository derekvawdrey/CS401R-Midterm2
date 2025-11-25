#!/usr/bin/env python3
"""
Main entry point for the move-based Snake game.
Supports both player mode and agent mode.
"""

import argparse
import sys
import os
from typing import Optional

# Add the move-based-snake directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'move-based-snake'))

from game import MoveBasedSnakeGame
from renderer import GameRenderer


def play_player_mode(game: MoveBasedSnakeGame, renderer: GameRenderer):
    """Run the game in player mode (keyboard controls)."""
    state = game.reset()
    renderer.reset_state_tracking()  # Initialize state tracking
    
    print("Controls:")
    print("  Arrow Keys or WASD: Move")
    print("  Space: Detach tail (convert to wall)")
    print("  R: Reset game")
    print("  ESC or Close window: Quit")
    
    running = True
    game_over = False
    
    while running:
        action = renderer.handle_events()
        
        if action == 'quit':
            running = False
            break
        elif action == 'reset':
            state = game.reset()
            renderer.reset_state_tracking()  # Reset sound tracking
            game_over = False
            print("Game reset!")
            continue
        
        if not game_over:
            # Game is still running
            if action is not None:
                # Perform action
                state, reward, done, info = game.step(action)
                
                if done:
                    game_over = True
                    # Check for sounds one more time (including game over sound)
                    renderer._check_and_play_sounds(game.get_state_dict())
                    # Render game over screen
                    renderer.render_game_over(
                        score=info.get('score', 0),
                        steps=info.get('steps', 0),
                        snake_length=info.get('snake_length', 0)
                    )
                else:
                    # Render normal game state (pass info for sound detection)
                    renderer.render(game.get_state_dict(), info)
                    # Small status update every 50 steps
                    if info.get('steps', 0) % 50 == 0:
                        print(f"Steps: {info['steps']}, Snake Length: {info['snake_length']}")
            else:
                # No action, render current state
                renderer.render(game.get_state_dict())
        else:
            # Game is over, show game over screen
            # Keep rendering game over screen until reset or quit
            renderer.render_game_over(
                score=game.score,
                steps=game.steps,
                snake_length=len(game.snake.body) if game.snake else 0
            )


def play_agent_mode(game: MoveBasedSnakeGame, renderer: Optional[GameRenderer], 
                    agent, max_steps: int = 1000, render: bool = True, auto_restart: bool = True):
    """
    Run the game in agent mode.
    
    Args:
        game: Game environment
        renderer: Renderer (None if rendering disabled)
        agent: Agent that provides actions (should have a predict/act method)
        max_steps: Maximum steps per episode
        render: Whether to render the game
        auto_restart: Whether to automatically restart when game ends
    """
    episode_count = 0
    
    while True:  # Loop for multiple episodes
        state = game.reset()
        if renderer:
            renderer.reset_state_tracking()
        
        total_reward = 0
        step_count = 0
        
        if render and renderer:
            renderer.render(game.get_state_dict())
        
        while not game.done and step_count < max_steps:
            # Get action from agent
            if hasattr(agent, 'predict'):
                action = agent.predict(state)
            elif hasattr(agent, 'act'):
                action = agent.act(state)
            elif callable(agent):
                action = agent(state)
            else:
                raise ValueError("Agent must have a predict(), act() method, or be callable")
            
            # Perform action
            state, reward, done, info = game.step(action)
            total_reward += reward
            step_count += 1
            
            # Render if enabled (but limit event checking for speed)
            if render and renderer:
                # Only check events occasionally (every 100 steps) to avoid slowdown
                if step_count % 100 == 0:
                    render_event = renderer.handle_events()
                    if render_event == 'quit':
                        return {
                            'total_reward': total_reward,
                            'steps': step_count,
                            'score': game.score,
                            'snake_length': len(game.snake.body) if game.snake else 0,
                            'episode': episode_count
                        }
                # Skip sound checking every step for speed (sounds still work)
                skip_sound = (step_count % 10 != 0)
                renderer.render(game.get_state_dict(), info, skip_sound_check=skip_sound)
        
        episode_count += 1
        result = {
            'total_reward': total_reward,
            'steps': step_count,
            'score': game.score,
            'snake_length': len(game.snake.body) if game.snake else 0,
            'episode': episode_count
        }
        
        if not auto_restart:
            return result
        
        # Auto-restart for next episode - continue loop immediately
        # No delay in agent mode for maximum speed
        # Continue to next episode (loop will restart at the top)


class RandomAgent:
    """Simple random agent for testing."""
    
    def __init__(self, action_space_size: int = 6):
        self.action_space_size = action_space_size
    
    def predict(self, state):
        import random
        return random.randint(0, self.action_space_size - 1)


def main():
    parser = argparse.ArgumentParser(description='Move-based Snake Game')
    parser.add_argument('--mode', choices=['player', 'agent'], default='player',
                       help='Game mode: player (keyboard) or agent (RL)')
    parser.add_argument('--width', type=int, default=20,
                       help='Grid width (default: 20)')
    parser.add_argument('--height', type=int, default=20,
                       help='Grid height (default: 20)')
    parser.add_argument('--monsters', type=int, default=3,
                       help='Number of monsters (default: 3)')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering (for agent mode)')
    parser.add_argument('--random-agent', action='store_true',
                       help='Use random agent for testing')
    parser.add_argument('--dqn-model', type=str, default=None,
                       help='Path to a trained DQN model to use')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum steps per episode (default: 1000)')
    
    args = parser.parse_args()
    
    # Create game
    game = MoveBasedSnakeGame(
        grid_width=args.width,
        grid_height=args.height,
        num_monsters=args.monsters
    )
    
    # Create renderer if needed
    renderer = None
    if not args.no_render:
        renderer = GameRenderer(
            grid_width=args.width,
            grid_height=args.height,
            cell_size=30,
            fps=10 if args.mode == 'player' else 60,  # Higher FPS for agent mode
            limit_fps=(args.mode == 'player'),  # Only limit FPS in player mode
            enable_sound_effects=(args.mode == 'player')  # Only sound effects in player mode
        )
    
    # Run game
    if args.mode == 'player':
        try:
            play_player_mode(game, renderer)
        except KeyboardInterrupt:
            print("\nGame interrupted.")
        finally:
            if renderer:
                renderer.close()
    else:  # agent mode
        # Create agent
        if args.random_agent:
            agent = RandomAgent(game.action_space_size)
        elif args.dqn_model:
            # Load DQN agent
            try:
                from agents.dqn_agent import SnakeDQNAgent
                print(f"Loading DQN agent from {args.dqn_model}")
                agent = SnakeDQNAgent(game, model_path=args.dqn_model)
                print("DQN agent loaded successfully!")
            except Exception as e:
                print(f"Error loading DQN agent: {e}")
                print("Falling back to random agent.")
                agent = RandomAgent(game.action_space_size)
        else:
            # Try to load agent from trainer
            try:
                # User can provide their own agent here
                print("No agent provided. Using random agent. Use --dqn-model to load a trained DQN.")
                agent = RandomAgent(game.action_space_size)
            except Exception as e:
                print(f"Error loading agent: {e}. Using random agent.")
                agent = RandomAgent(game.action_space_size)
        
        # Run episodes with auto-restart
        try:
            while True:
                result = play_agent_mode(
                    game, renderer, agent,
                    max_steps=args.max_steps,
                    render=not args.no_render,
                    auto_restart=True
                )
                
                # If auto_restart is True, play_agent_mode will loop internally
                # This break will only happen if user quits or auto_restart=False
                if result:
                    print(f"Episode {result.get('episode', 1)} finished:")
                    print(f"  Total Reward: {result['total_reward']:.2f}")
                    print(f"  Steps: {result['steps']}")
                    print(f"  Score: {result['score']}")
                    print(f"  Snake Length: {result['snake_length']}")
                    break
        except KeyboardInterrupt:
            print("\nAgent mode interrupted.")
        finally:
            if renderer:
                renderer.close()


if __name__ == '__main__':
    main()

