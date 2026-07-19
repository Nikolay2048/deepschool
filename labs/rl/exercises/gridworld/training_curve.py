"""Plot GridWorld Q-learning training curves.

This file intentionally contains complete plotting/measurement code. The RL
algorithm itself lives in q_learning.py.

Run:
    python labs/rl/exercises/gridworld/training_curve.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from gridworld import ACTIONS, GridWorldEnv
from q_learning import QLearningAgent


def rolling_average(values: list[float], window: int) -> list[float]:
    """Compute rolling average using a trailing window."""
    if window <= 0:
        raise ValueError("window must be positive")

    averages: list[float] = []

    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        averages.append(sum(chunk) / len(chunk))

    return averages


def train_with_history(
    episodes: int = 500,
    max_steps: int = 50,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Train Q-learning agent and collect per-episode metrics."""
    env = GridWorldEnv()
    agent = QLearningAgent(actions=ACTIONS, seed=seed)

    episode_rewards: list[float] = []
    episode_steps: list[float] = []
    episode_successes: list[float] = []

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0
        success = 0.0

        for step_idx in range(1, max_steps + 1):
            action = agent.choose_action(state)
            result = env.step(action)

            agent.update(
                state=state,
                action=action,
                reward=result.reward,
                next_state=result.state,
                done=result.done,
            )

            total_reward += result.reward
            state = result.state

            if result.done:
                success = 1.0
                episode_steps.append(float(step_idx))
                break
        else:
            episode_steps.append(float(max_steps))

        episode_rewards.append(total_reward)
        episode_successes.append(success)

    return {
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
        "episode_successes": episode_successes,
    }


def plot_training_curves(
    episodes: int = 500,
    max_steps: int = 50,
    window: int = 20,
    seed: int = 42,
) -> Path:
    history = train_with_history(
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,
    )

    output_path = Path("outputs") / "rl" / "gridworld-training-curves.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = list(range(1, episodes + 1))

    reward_avg = rolling_average(history["episode_rewards"], window=window)
    success_avg = rolling_average(history["episode_successes"], window=window)
    steps_avg = rolling_average(history["episode_steps"], window=window)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(x, reward_avg)
    axes[0].set_ylabel("Reward")
    axes[0].set_title(f"GridWorld Q-learning training curves, rolling window={window}")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, success_avg)
    axes[1].set_ylabel("Success rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, steps_avg)
    axes[2].set_ylabel("Steps")
    axes[2].set_xlabel("Episode")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def main() -> None:
    output_path = plot_training_curves()
    print(f"Saved GridWorld training curves to {output_path}")


if __name__ == "__main__":
    main()

