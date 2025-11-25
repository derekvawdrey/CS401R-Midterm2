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
    
    print("Controls:")
    print("  Arrow Keys or WASD: Move")
    print("  Space: Detach tail (convert to wall)")
    print("  R: Reset game")
    print("  ESC or Close window: Quit")
    
    running = True
    while running:
        action = renderer.handle_events()
        
        if action == 'quit':
            running = False
            break
        elif action == 'reset':
            state = game.reset()
            print("Game reset!")
            continue
        elif action is not None:
            # Perform action
            state, reward, done, info = game.step(action)
            
            if done:
                print(f"Game Over! Score: {info.get('score', 0)}, Steps: {info.get('steps', 0)}")
                print("Press R to reset or close window to quit.")
            else:
                # Small status update every 50 steps
                if info.get('steps', 0) % 50 == 0:
                    print(f"Steps: {info['steps']}, Snake Length: {info['snake_length']}")
        else:
            # No action, just render current state
            pass
        
        # Render current state
        renderer.render(game.get_state_dict())


def play_agent_mode(game: MoveBasedSnakeGame, renderer: Optional[GameRenderer], 
                    agent, max_steps: int = 1000, render: bool = True):
    """
    Run the game in agent mode.
    
    Args:
        game: Game environment
        renderer: Renderer (None if rendering disabled)
        agent: Agent that provides actions (should have a predict/act method)
        max_steps: Maximum steps per episode
        render: Whether to render the game
    """
    state = game.reset()
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
        
        # Render if enabled
        if render and renderer:
            render_event = renderer.handle_events()
            if render_event == 'quit':
                break
            renderer.render(game.get_state_dict())
    
    return {
        'total_reward': total_reward,
        'steps': step_count,
        'score': game.score,
        'snake_length': len(game.snake.body) if game.snake else 0
    }


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
            fps=10 if args.mode == 'player' else 5
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
        else:
            # Try to load agent from trainer
            try:
                # User can provide their own agent here
                print("No agent provided. Using random agent. Implement your agent in trainer/")
                agent = RandomAgent(game.action_space_size)
            except Exception as e:
                print(f"Error loading agent: {e}. Using random agent.")
                agent = RandomAgent(game.action_space_size)
        
        # Run episode
        result = play_agent_mode(
            game, renderer, agent,
            max_steps=args.max_steps,
            render=not args.no_render
        )
        
        print(f"Episode finished:")
        print(f"  Total Reward: {result['total_reward']:.2f}")
        print(f"  Steps: {result['steps']}")
        print(f"  Score: {result['score']}")
        print(f"  Snake Length: {result['snake_length']}")
        
        if renderer:
            renderer.close()


if __name__ == '__main__':
    main()

