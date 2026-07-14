"""Plot learning curves for epsilon-greedy bandit agents.

Run:
    python labs/rl/exercises/bandit/learning_curves.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from bandit import BanditEnv, EpsilonGreedyAgent


def run_reward_history(
    epsilon: float,
    k: int = 10,
    steps: int = 1000,
    seed: int = 42,
) -> list[float]:
    """Run one bandit experiment and return reward at each step."""
    env = BanditEnv(k=k, seed=seed)
    agent = EpsilonGreedyAgent(k=k, epsilon=epsilon, seed=seed + 1)

    rewards: list[float] = []

    for _ in range(steps):
        action = agent.choose_action()
        reward = env.step(action)
        agent.update(action, reward)
        rewards.append(reward)
    return rewards


def cumulative_average(values: list[float]) -> list[float]:
    """Return cumulative average for each prefix of values."""
    averages: list[float] = []
    total = 0.0

    for idx, value in enumerate(values, start=1):
        total += value
        averages.append(total / idx)

    return averages


def plot_learning_curves(
    epsilons: list[float],
    steps: int = 1000,
    seed: int = 42,
) -> Path:
    output_path = Path("outputs") / "rl" / "bandit-learning-curves.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for epsilon in epsilons:
        rewards = run_reward_history(epsilon=epsilon, steps=steps, seed=seed)
        averages = cumulative_average(rewards)
        plt.plot(averages, label=f"epsilon={epsilon}")

    plt.title("Bandit learning curves")
    plt.xlabel("Step")
    plt.ylabel("Cumulative average reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def main() -> None:
    epsilons = [0.0, 0.01, 0.1, 0.3, 1.0]
    output_path = plot_learning_curves(epsilons=epsilons)
    print(f"Saved learning curves to {output_path}")


if __name__ == "__main__":
    main()

