#!/usr/bin/env python3
"""
General training script for the Falling Objects avoidance game.
Supports DQN and can be extended for other agent types.
"""

import argparse
import sys
import os
from typing import Optional

# Add the move-based-snake directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'move-based-snake'))

from game import FallingObjectsGame
from renderer import GameRenderer
from agents.trainer import train_any_agent
from agents.dqn_agent import DQNAgent


def main():
    parser = argparse.ArgumentParser(description='Train an agent for the Falling Objects avoidance game')
    parser.add_argument('--agent', type=str, default='dqn',
                       choices=['dqn'],
                       help='Agent type to train (default: dqn)')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes (default: 1000)')
    parser.add_argument('--width', type=int, default=20,
                       help='Grid width (default: 20)')
    parser.add_argument('--height', type=int, default=20,
                       help='Grid height (default: 20)')
    parser.add_argument('--fall-prob', type=float, default=0.1,
                       help='Probability of spawning a falling object each step (default: 0.1)')
    parser.add_argument('--render', action='store_true',
                       help='Render the game during training (slower)')
    parser.add_argument('--load-model', type=str, default=None,
                       help='Path to load a pre-trained model from')
    parser.add_argument('--save-model', type=str, default='agent_model.pth',
                       help='Path to save the trained model (default: agent_model.pth)')
    parser.add_argument('--save-freq', type=int, default=100,
                       help='Frequency (episodes) to save model (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor (default: 0.99)')
    parser.add_argument('--epsilon', type=float, default=1.0,
                       help='Initial epsilon (default: 1.0)')
    parser.add_argument('--epsilon-min', type=float, default=0.01,
                       help='Minimum epsilon (default: 0.01)')
    parser.add_argument('--epsilon-decay', type=float, default=0.9998,
                       help='Epsilon decay rate (default: 0.995)')  
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training (default: 64)')
    parser.add_argument('--memory-size', type=int, default=20000,
                       help='Replay buffer size (default: 20000)')
    parser.add_argument('--target-update-freq', type=int, default=100,
                       help='Frequency (in steps) to update target network (default: 100)')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum steps per episode (default: 1000)')
    parser.add_argument('--no-danger-signals', action='store_true',
                       help='Disable danger signals in observation')
    parser.add_argument('--enable-danger-signals', action='store_true',
                       help='Enable danger signals in observation (overrides training_options.py)')
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
    if args.render:
        renderer = GameRenderer(
            grid_width=args.width,
            grid_height=args.height,
            cell_size=30,
            fps=60,
            limit_fps=False,
            enable_sound_effects=False
        )
    
    # Create agent based on type
    if args.agent == 'dqn':
        agent = DQNAgent(
            game=game,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            batch_size=args.batch_size,
            memory_size=args.memory_size,
            target_update_freq=args.target_update_freq,
            model_path=args.load_model
        )
        if args.load_model and hasattr(agent, 'dqn'):
            print(f"Initial epsilon: {agent.dqn.epsilon:.3f}")
    else:
        raise ValueError(f"Unknown agent type: {args.agent}")
    
    # Train using general training function
    try:
        train_any_agent(
            game=game,
            agent=agent,
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            render=args.render,
            renderer=renderer,
            save_freq=args.save_freq,
            model_path=args.save_model,
            verbose=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        try:
            agent.save(args.save_model)
        except Exception as e:
            print(f"Warning: Could not save model: {e}")
    finally:
        if renderer:
            renderer.close()


if __name__ == '__main__':
    main()
