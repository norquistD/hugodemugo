"""
Training Orchestration and Main Entry Point

Main entry point that coordinates training and evaluation. Orchestrates interaction
between Agent (from agent.py), SnakeGame (from snake_game.py), and Renderer (from
renderer.py). Reads configuration from configs.py. Manages training loops, progress
tracking, and Q-table persistence. Supports both training mode (with learning) and
evaluation mode (testing without updates).
"""

import itertools
import os
from typing import Optional, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from agent import Agent
from snake_game import SnakeGame
from renderer import Renderer
from configs import init_config, get_config


def _update_progress_bar(
    pbar: tqdm,
    game_count: int,
    score: int,
    reason: Optional[str],
    high_score: int,
    epsilon: float,
    avg_reward: float,
) -> None:
    """Update progress bar postfix with current game statistics."""
    cfg = get_config()
    postfix = {
        "Score": score,
        "Reason": reason,
        "High": high_score,
        "Epsilon": f"{epsilon:.4f}",
        "AvgReward": f"{avg_reward:.3f}",
    }
    if cfg.MAX_STEPS is not None:
        postfix["Games"] = game_count
    pbar.set_postfix(postfix)


def plot_rewards_and_scores(rewards: list[float], scores: list[int], state: str) -> None:
    """
    Plot reward and score per game with moving averages as two separate subplots on the same image.
    
    Args:
        rewards: List of rewards for each game
        scores: List of scores for each game
        state: State representation name (e.g., "naive", "basic")
    """
    if not rewards or not scores:
        return
    
    if len(rewards) != len(scores):
        return
    
    # Calculate moving averages with window size 100
    window_size = 100
    reward_moving_avg = []
    score_moving_avg = []
    
    for i in range(len(rewards)):
        start_idx = max(0, i - window_size + 1)
        window_rewards = rewards[start_idx:i + 1]
        window_scores = scores[start_idx:i + 1]
        reward_moving_avg.append(sum(window_rewards) / len(window_rewards))
        score_moving_avg.append(sum(window_scores) / len(window_scores))
    
    # Create figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    games = list(range(1, len(rewards) + 1))
    
    # Plot rewards on top subplot
    ax1.plot(games, rewards, alpha=0.3, color='blue', label='Reward per Game', linewidth=0.5)
    ax1.plot(games, reward_moving_avg, color='red', label=f'Moving Average (window={window_size})', linewidth=2)
    ax1.set_xlabel('Game Number')
    ax1.set_ylabel('Reward')
    ax1.set_title(f'Reward per Game - {state.capitalize()} State Representation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot scores on bottom subplot
    ax2.plot(games, scores, alpha=0.3, color='green', label='Score per Game', linewidth=0.5)
    ax2.plot(games, score_moving_avg, color='orange', label=f'Moving Average (window={window_size})', linewidth=2)
    ax2.set_xlabel('Game Number')
    ax2.set_ylabel('Score')
    ax2.set_title(f'Score per Game - {state.capitalize()} State Representation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to file
    filename = f"{state}/reward_and_score_plots.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Reward and score plots saved to {filename}")


def run_episode(agent: Agent, renderer: Renderer, update_qvalues: bool = True) -> Tuple[int, Optional[str], int, float]:
    """
    Run a single episode and return results (Gym convention).

    Args:
        agent: The agent to use
        renderer: The renderer for visualization
        update_qvalues: If True, update Q-values after episode (training mode).
                       If False, skip Q-value updates (evaluation mode).

    Returns:
        Tuple of (score, reason, steps, total_reward)
    """
    game = SnakeGame()
    agent.reset()

    while not game.done:
        # Handle events (quit, etc.)
        if not renderer.handle_events():
            raise SystemExit

        # Get observation and action from agent
        observation = game.get_observation()
        action = agent.act(
            observation["direction"],
            observation["snake"],
            observation["food"],
        )

        # Execute step (Gym-style: returns observation, reward, done, info)
        observation, reward, done, info = game.step(action)

        # Render if visualization enabled
        renderer.render(observation)

        if done:
            break

    # Update Q-values at end of episode (only in training mode)
    if update_qvalues:
        total_reward = agent.update(game.reason)
    else:
        # In evaluation mode, calculate reward for display but don't update Q-values
        total_reward = 0.0  # We don't need the actual reward in eval mode

    return game.get_score(), game.reason, game.steps, total_reward


def train() -> None:
    """Main training loop."""
    cfg = get_config()

    # Initialize agent and renderer
    agent = Agent(cfg.DISPLAY_WIDTH, cfg.DISPLAY_HEIGHT, cfg.BLOCK_SIZE)
    renderer = Renderer()

    # Training state
    high_score = 0
    total_steps = 0
    recent_rewards = []  # Track rewards for last 100 games
    all_rewards = []  # Track all rewards for plotting
    all_scores = []  # Track all scores for plotting

    # Setup progress bar
    if cfg.MAX_STEPS is not None:
        pbar = tqdm(
            total=cfg.MAX_STEPS,
            desc="Training",
            unit=" step",
            dynamic_ncols=True,
            mininterval=0.1,
            maxinterval=1.0,
            smoothing=0.1,
        )
    else:
        pbar = tqdm(
            itertools.count(start=1),
            desc="Training",
            unit=" game",
            dynamic_ncols=True,
            mininterval=0.1,
            maxinterval=1.0,
            smoothing=0.1,
        )

    # Training loop
    try:
        for game_count in itertools.count(start=1):
            # Update epsilon
            agent.epsilon = max(0, cfg.EPSILON * (0.8 - (total_steps / cfg.MAX_STEPS)**4))

            # Run one episode
            score, reason, game_steps, total_reward = run_episode(agent, renderer)
            total_steps += game_steps

            # Track rewards
            recent_rewards.append(total_reward)
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            all_rewards.append(total_reward)  # Track all rewards for plotting
            all_scores.append(score)  # Track all scores for plotting

            # Update progress bar
            if cfg.MAX_STEPS is not None:
                pbar.update(game_steps)
            else:
                pbar.update(1)
            _update_progress_bar(
                pbar, game_count, score, reason, high_score, agent.epsilon, avg_reward
            )

            # Check if we've reached maximum steps
            if cfg.MAX_STEPS is not None and total_steps >= cfg.MAX_STEPS:
                pbar.write(f"Reached maximum steps: {cfg.MAX_STEPS}")
                break

            # Save high score
            if score > high_score:
                high_score = score
                if not os.path.exists(cfg.HIGHSCORES_DIR):
                    os.makedirs(cfg.HIGHSCORES_DIR)
                agent.save_qvalues(
                    path=os.path.join(
                        cfg.HIGHSCORES_DIR, f"highscore{high_score}.json"
                    )
                )
                _update_progress_bar(
                    pbar, game_count, score, reason, high_score, agent.epsilon, avg_reward
                )

            # Periodic save
            if game_count % cfg.QVALUES_SAVE_INTERVAL == 0:
                # pbar.write("Save Qvals")
                agent.save_qvalues()
    except KeyboardInterrupt:
        pbar.write("Training interrupted by user. Saving Q-values...")
    finally:
        # Final save
        agent.save_qvalues()
        pbar.close()
        # Generate combined plot
        if all_rewards and all_scores:
            plot_rewards_and_scores(all_rewards, all_scores, cfg.STATE_REPRESENTATION)


def evaluate() -> None:
    """Evaluation mode: test agent without training (epsilon=0, no Q-table updates)."""
    cfg = get_config()

    # Initialize agent and renderer
    agent = Agent(cfg.DISPLAY_WIDTH, cfg.DISPLAY_HEIGHT, cfg.BLOCK_SIZE)
    renderer = Renderer()

    # Set epsilon to 0 for pure exploitation (no exploration)
    agent.epsilon = 0.0

    # Evaluation statistics
    scores = []
    reasons = []
    total_steps = 0

    print("Running evaluation mode (epsilon=0, no Q-table updates)...")

    # Setup progress bar
    if cfg.MAX_STEPS is not None:
        pbar = tqdm(
            total=cfg.MAX_STEPS,
            desc="Evaluation",
            unit=" step",
            dynamic_ncols=True,
            mininterval=0.1,
            maxinterval=1.0,
            smoothing=0.1,
        )
    else:
        pbar = tqdm(
            itertools.count(start=1),
            desc="Evaluation",
            unit=" episode",
            dynamic_ncols=True,
            mininterval=0.1,
            maxinterval=1.0,
            smoothing=0.1,
        )

    try:
        for _ in itertools.count(start=1):
            # Run one episode without Q-value updates
            score, reason, game_steps, _ = run_episode(agent, renderer, update_qvalues=False)
            total_steps += game_steps

            # Track statistics
            scores.append(score)
            reasons.append(reason)

            # Update progress bar
            if cfg.MAX_STEPS is not None:
                pbar.update(game_steps)
            else:
                pbar.update(1)
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            pbar.set_postfix({
                "Score": score,
                "Avg": f"{avg_score:.1f}",
                "Max": max_score,
                "Reason": reason,
            })

            # Check if we've reached maximum steps
            if cfg.MAX_STEPS is not None and total_steps >= cfg.MAX_STEPS:
                pbar.write(f"Reached maximum steps: {cfg.MAX_STEPS}")
                break
    except KeyboardInterrupt:
        pbar.write("Evaluation interrupted by user.")
    finally:
        pbar.close()

        # Print final statistics
        if scores:
            print(f"Episodes: {len(scores)}")
            print(f"Average Score: {sum(scores) / len(scores):.2f}")
            print(f"Max Score: {max(scores)}")
            print(f"Min Score: {min(scores)}")
            print(f"Total Steps: {total_steps}")
            print(f"Average Steps per Episode: {total_steps / len(scores):.1f}")

if __name__ == "__main__":
    # Initialize configuration from command-line arguments
    init_config()
    cfg = get_config()
    
    # Check if evaluation mode is enabled
    if cfg.EVALS_MODE:
        evaluate()
    else:
        train()
