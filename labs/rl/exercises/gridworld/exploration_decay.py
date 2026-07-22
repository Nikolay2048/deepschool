"""Compare constant epsilon with decaying epsilon in GridWorld Q-learning.

Run:
    python labs/rl/exercises/gridworld/exploration_decay.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from gridworld import ACTIONS, GridWorldEnv
from q_learning import QLearningAgent, render_policy
from training_curve import rolling_average


def linear_decay(
    episode_idx: int,
    episodes: int,
    start: float,
    end: float,
) -> float:
    """Linearly decay from start to end over all episodes."""
    if episodes <= 1:
        return end

    progress = episode_idx / (episodes - 1)
    return start + progress * (end - start)


def train_with_epsilon_schedule(
    schedule_name: str,
    episodes: int = 500,
    max_steps: int = 50,
    seed: int = 42,
    constant_epsilon: float = 0.2,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
) -> dict[str, list[float] | QLearningAgent]:
    """Train Q-learning with either constant or decaying epsilon."""
    env = GridWorldEnv()
    agent = QLearningAgent(actions=ACTIONS, epsilon=constant_epsilon, seed=seed)

    episode_rewards: list[float] = []
    episode_steps: list[float] = []
    episode_successes: list[float] = []
    epsilons: list[float] = []

    for episode_idx in range(episodes):
        if schedule_name == "constant":
            epsilon = constant_epsilon
        elif schedule_name == "decay":
            epsilon = linear_decay(
                episode_idx=episode_idx,
                episodes=episodes,
                start=epsilon_start,
                end=epsilon_end,
            )
        else:
            raise ValueError(f"Unknown schedule_name: {schedule_name}")

        agent.epsilon = epsilon
        epsilons.append(epsilon)

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
        "agent": agent,
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
        "episode_successes": episode_successes,
        "epsilons": epsilons,
    }


def plot_comparison(
    episodes: int = 500,
    max_steps: int = 50,
    window: int = 20,
    seed: int = 42,
) -> Path:
    constant = train_with_epsilon_schedule(
        schedule_name="constant",
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,
    )
    decay = train_with_epsilon_schedule(
        schedule_name="decay",
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,
    )

    output_path = Path("outputs") / "rl" / "gridworld-exploration-decay.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = list(range(1, episodes + 1))

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    for label, history in [("constant epsilon", constant), ("decaying epsilon", decay)]:
        rewards = rolling_average(history["episode_rewards"], window=window)
        successes = rolling_average(history["episode_successes"], window=window)
        steps = rolling_average(history["episode_steps"], window=window)

        axes[0].plot(x, rewards, label=label)
        axes[1].plot(x, successes, label=label)
        axes[2].plot(x, steps, label=label)
        axes[3].plot(x, history["epsilons"], label=label)

    axes[0].set_title(f"GridWorld exploration decay comparison, rolling window={window}")
    axes[0].set_ylabel("Reward")
    axes[1].set_ylabel("Success rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("Steps")
    axes[3].set_ylabel("Epsilon")
    axes[3].set_xlabel("Episode")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    print("Final greedy policies")
    print("---------------------")
    env = GridWorldEnv()
    print("Constant epsilon:")
    print(render_policy(env, constant["agent"]))
    print()
    print("Decaying epsilon:")
    print(render_policy(env, decay["agent"]))
    print()

    return output_path


def main() -> None:
    output_path = plot_comparison()
    print(f"Saved exploration decay comparison to {output_path}")


if __name__ == "__main__":
    main()

