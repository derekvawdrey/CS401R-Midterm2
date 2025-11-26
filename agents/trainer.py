"""
General training function that works with any agent implementing BaseAgent interface.
"""

from typing import Optional, Dict, Any
import numpy as np

try:
    from .base_agent import BaseAgent
except ImportError:
    from base_agent import BaseAgent


def train_any_agent(
    game,
    agent: BaseAgent,
    num_episodes: int = 1000,
    max_steps_per_episode: int = 1000,
    render: bool = False,
    renderer = None,
    save_freq: int = 100,
    model_path: str = "agent_model.pth",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train any agent that implements the BaseAgent interface.
    
    This function works with any agent that has:
    - act(state, training=True): Choose action with exploration
    - remember(state, action, reward, next_state, done): Store experience (optional)
    - train_step(): Perform training step (optional)
    - save(filepath): Save model (optional)
    
    Args:
        game: Game environment (FallingObjectsGame instance)
        agent: Agent instance (must inherit from BaseAgent)
        num_episodes: Number of episodes to train
        max_steps_per_episode: Maximum steps per episode
        render: Whether to render the game
        renderer: Renderer instance (if rendering)
        save_freq: Frequency (in episodes) to save the model
        model_path: Path to save the model
        verbose: Whether to print training progress
        
    Returns:
        Dictionary with training statistics
    """
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    episode_losses = []
    
    if verbose:
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Agent type: {type(agent).__name__}")
    
    for episode in range(num_episodes):
        state = game.reset()
        total_reward = 0.0
        steps = 0
        total_loss = 0.0
        loss_count = 0
        
        if render and renderer:
            renderer.reset_state_tracking()
            renderer.render(game.get_state_dict())
        
        while not game.done and steps < max_steps_per_episode:
            # Choose action (with exploration during training)
            action = agent.act(state, training=True)
            
            # Take step in environment
            next_state, reward, done, info = game.step(action)
            
            # Store experience (if agent supports it)
            agent.remember(state, action, reward, next_state, done)
            
            # Train the agent (if agent supports online training)
            loss = agent.train_step()
            if loss is not None:
                total_loss += loss
                loss_count += 1
            
            # Update state
            state = next_state
            total_reward += reward
            steps += 1
            
            # Render if enabled
            if render and renderer:
                if steps % 10 == 0:  # Render every 10 steps for speed
                    render_event = renderer.handle_events()
                    if render_event == 'quit':
                        if verbose:
                            print("\nTraining interrupted by user.")
                        return {
                            'episode_rewards': episode_rewards,
                            'episode_lengths': episode_lengths,
                            'episode_scores': episode_scores,
                            'episode_losses': episode_losses,
                            'episodes_completed': episode + 1
                        }
                    renderer.render(game.get_state_dict(), info, skip_sound_check=True)
        
        # Record episode statistics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_scores.append(game.score)
        avg_loss = total_loss / loss_count if loss_count > 0 else None
        episode_losses.append(avg_loss)
        
        # Call episode_end callback (for epsilon decay, etc.)
        agent.episode_end()
        
        # Print progress
        if verbose and (episode + 1) % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / 10
            avg_length = sum(episode_lengths[-10:]) / 10
            avg_score = sum(episode_scores[-10:]) / 10
            loss_str = f" | Avg Loss: {sum([l for l in episode_losses[-10:] if l is not None]) / max(1, len([l for l in episode_losses[-10:] if l is not None])):.4f}" if any(l is not None for l in episode_losses[-10:]) else ""
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Avg Score: {avg_score:.2f}{loss_str}")
        
        # Save model periodically (if agent supports it)
        if (episode + 1) % save_freq == 0:
            try:
                agent.save(model_path)
                if verbose:
                    print(f"Model saved at episode {episode + 1}")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not save model at episode {episode + 1}: {e}")
    
    # Final save
    try:
        agent.save(model_path)
        if verbose:
            print(f"\nTraining complete! Model saved to {model_path}")
    except Exception as e:
        if verbose:
            print(f"Warning: Could not save final model: {e}")
    
    if verbose:
        final_avg_reward = sum(episode_rewards[-100:]) / min(100, len(episode_rewards))
        print(f"Average reward (last 100 episodes): {final_avg_reward:.2f}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_scores': episode_scores,
        'episode_losses': episode_losses,
        'episodes_completed': num_episodes
    }

