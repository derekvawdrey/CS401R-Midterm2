#!/usr/bin/env python3
"""
Main entry point for the falling objects avoidance game.
Supports both player mode and agent mode.
"""

import argparse
import sys
import os
from typing import Optional

# Add the move-based-snake directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'move-based-snake'))

from game import FallingObjectsGame
from renderer import GameRenderer


def play_player_mode(game: FallingObjectsGame, renderer: GameRenderer):
    """Run the game in player mode (keyboard controls)."""
    state = game.reset()
    renderer.reset_state_tracking()
    
    print("Controls:")
    print("  Arrow Keys or WASD: Move")
    print("  Space: Stay still (no movement)")
    print("  R: Reset game")
    print("  ESC or Close window: Quit")
    print("\nAvoid exploding meteors! You get a 2-step warning before meteors land.")
    print("Meteors explode in a 3x3 radius (center + 8 adjacent cells).")
    print("Explosions are visible for 2 steps - stay clear of the red zones!")
    print("The board stays clear - meteors explode and disappear (no permanent walls).")
    
    running = True
    game_over = False
    
    while running:
        action = renderer.handle_events()
        
        if action == 'quit':
            running = False
            break
        elif action == 'reset':
            state = game.reset()
            renderer.reset_state_tracking()
            game_over = False
            print("Game reset!")
            continue
        
        if not game_over:
            if action is not None:
                state, reward, done, info = game.step(action)
                
                if done:
                    game_over = True
                    renderer._check_and_play_sounds(game.get_state_dict())
                    renderer.render_game_over(
                        score=info.get('score', 0),
                        steps=info.get('steps', 0)
                    )
                else:
                    renderer.render(game.get_state_dict(), info)
                    if info.get('steps', 0) % 50 == 0:
                        print(f"Steps: {info['steps']}, Walls: {info.get('walls_count', 0)}")
            else:
                renderer.render(game.get_state_dict())
        else:
            renderer.render_game_over(
                score=game.score,
                steps=game.steps
            )


def play_agent_mode(game: FallingObjectsGame, renderer: Optional[GameRenderer], 
                    agent, max_steps: int = 1000, render: bool = True, auto_restart: bool = True,
                    debug: bool = False):
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
    
    while True:
        state = game.reset()
        if renderer:
            renderer.reset_state_tracking()
        
        total_reward = 0
        step_count = 0
        
        if render and renderer:
            renderer.render(game.get_state_dict())
        
        while not game.done and step_count < max_steps:
            if hasattr(agent, 'predict'):
                action = agent.predict(state, debug=(debug and step_count < 10))
            elif hasattr(agent, 'act'):
                action = agent.act(state)
            elif callable(agent):
                action = agent(state)
            else:
                raise ValueError("Agent must have a predict(), act() method, or be callable")
            
            if debug and step_count < 10:
                action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
                print(f"Step {step_count}: Action = {action} ({action_names[action] if 0 <= action < 4 else 'INVALID'})")
            
            state, reward, done, info = game.step(action)
            total_reward += reward
            step_count += 1
            
            if render and renderer:
                if step_count % 100 == 0:
                    render_event = renderer.handle_events()
                    if render_event == 'quit':
                        return {
                            'total_reward': total_reward,
                            'steps': step_count,
                            'score': game.score,
                            'episode': episode_count
                        }
                skip_sound = (step_count % 10 != 0)
                renderer.render(game.get_state_dict(), info, skip_sound_check=skip_sound)
        
        episode_count += 1
        result = {
            'total_reward': total_reward,
            'steps': step_count,
            'score': game.score,
            'episode': episode_count
        }
        
        if not auto_restart:
            return result


class RandomAgent:
    """Simple random agent for testing."""
    
    def __init__(self, action_space_size: int = 5):
        self.action_space_size = action_space_size
    
    def predict(self, state):
        import random
        return random.randint(0, self.action_space_size - 1)


def main():
    parser = argparse.ArgumentParser(description='Falling Objects Avoidance Game')
    parser.add_argument('--mode', choices=['player', 'agent'], default='player',
                       help='Game mode: player (keyboard) or agent (RL)')
    parser.add_argument('--width', type=int, default=20,
                       help='Grid width (default: 20)')
    parser.add_argument('--height', type=int, default=20,
                       help='Grid height (default: 20)')
    parser.add_argument('--fall-prob', type=float, default=0.1,
                       help='Probability of spawning a falling object each step (default: 0.1)')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering (for agent mode)')
    parser.add_argument('--random-agent', action='store_true',
                       help='Use random agent for testing')
    parser.add_argument('--dqn-model', type=str, default=None,
                       help='Path to a trained DQN model to use')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum steps per episode (default: 1000)')
    parser.add_argument('--no-danger-signals', action='store_true',
                       help='Disable danger signals in observation')
    parser.add_argument('--enable-danger-signals', action='store_true',
                       help='Enable danger signals in observation (overrides training_options.py)')
    parser.add_argument('--debug-agent', action='store_true',
                       help='Print Q-values and actions for debugging')
    parser.add_argument('--no-loop', action='store_true',
                       help='Run only one episode and exit (disable auto-restart)')
    parser.add_argument('--wrap-boundaries', action='store_true',
                       help='Enable boundary wrapping (player wraps around screen edges)')
    
    args = parser.parse_args()
    
    enable_danger = None
    if args.enable_danger_signals:
        enable_danger = True
    elif args.no_danger_signals:
        enable_danger = False
    
    # Create game
    game = FallingObjectsGame(
        grid_width=args.width,
        grid_height=args.height,
        fall_probability=args.fall_prob,
        enable_danger_signals=enable_danger,
        wrap_boundaries=args.wrap_boundaries
    )
    
    # Create renderer if needed
    renderer = None
    if not args.no_render:
        renderer = GameRenderer(
            grid_width=args.width,
            grid_height=args.height,
            cell_size=30,
            fps=10 if args.mode == 'player' else 60,
            limit_fps=(args.mode == 'player'),
            enable_sound_effects=(args.mode == 'player')
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
        if args.random_agent:
            agent = RandomAgent(game.action_space_size)
        elif args.dqn_model:
            try:
                from agents.dqn_agent import DQNAgent
                print(f"Loading DQN agent from {args.dqn_model}")
                agent = DQNAgent(game, model_path=args.dqn_model)
                print("DQN agent loaded successfully!")
            except Exception as e:
                print(f"Error loading DQN agent: {e}")
                print("Falling back to random agent.")
                agent = RandomAgent(game.action_space_size)
        else:
            print("No agent provided. Using random agent. Use --dqn-model to load a trained DQN.")
            agent = RandomAgent(game.action_space_size)
        
        try:
            if args.no_loop:
                result = play_agent_mode(
                    game, renderer, agent,
                    max_steps=args.max_steps,
                    render=not args.no_render,
                    auto_restart=False,
                    debug=args.debug_agent
                )
                if result:
                    print(f"Episode finished:")
                    print(f"  Total Reward: {result['total_reward']:.2f}")
                    print(f"  Steps: {result['steps']}")
                    print(f"  Score: {result['score']}")
            else:
                while True:
                    result = play_agent_mode(
                        game, renderer, agent,
                        max_steps=args.max_steps,
                        render=not args.no_render,
                        auto_restart=True,
                        debug=args.debug_agent
                    )
                    if result:
                        print(f"Episode {result.get('episode', 1)} finished:")
                        print(f"  Total Reward: {result['total_reward']:.2f}")
                        print(f"  Steps: {result['steps']}")
                        print(f"  Score: {result['score']}")
                        break
        except KeyboardInterrupt:
            print("\nAgent mode interrupted.")
        finally:
            if renderer:
                renderer.close()


if __name__ == '__main__':
    main()
