"""First RL exercise: multi-armed bandit with epsilon-greedy agent.

Run:
    python labs/rl/exercises/01-bandit/bandit.py
"""

from __future__ import annotations

import random


class BanditEnv:
    """K independent slot machines with hidden mean rewards."""

    def __init__(self, k: int, reward_std: float = 1.0, seed: int | None = None):
        if k <= 0:
            raise ValueError("k must be positive")

        self.k = k
        self.reward_std = reward_std
        self.rng = random.Random(seed)

        # Hidden true values. The agent cannot read these during learning.
        self.true_values = [self.rng.gauss(0.0, 1.0) for _ in range(k)]

    def step(self, action: int) -> float:
        """Pull one arm and receive a noisy reward."""
        if not 0 <= action < self.k:
            raise ValueError(f"Action must be in [0, {self.k})")
        true_mean = self.true_values[action]
        return self.rng.gauss(true_mean, self.reward_std)


class EpsilonGreedyAgent:
    """Agent that sometimes explores and otherwise picks the best known arm."""

    def __init__(self, k: int, epsilon: float, seed: int | None = None):
        if k <= 0:
            raise ValueError("k must be positive")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")

        self.k = k
        self.epsilon = epsilon
        self.rng = random.Random(seed)

        # Q[action] is the current estimate of this action's average reward.
        self.q_values = [0.0 for _ in range(k)]
        self.action_counts = [0 for _ in range(k)]

    def choose_action(self) -> int:
        """Choose an action using epsilon-greedy strategy."""

        if self.rng.random() < self.epsilon:
            choice = self.rng.randrange(self.k)
        else:
            choice = self.q_values.index(max(self.q_values))

        self.action_counts[choice] += 1
        return choice

    def update(self, action: int, reward: float) -> None:
        """
        Update estimated value for the selected action.
        Q_new = Q_old + (reward - Q_old) / N
        """
        self.q_values[action] = self.q_values[action] + (reward - self.q_values[action]) / self.action_counts[action]


def run_experiment(
        k: int = 10,
        steps: int = 1000,
        epsilon: float = 0.1,
        seed: int = 42,
) -> None:
    env = BanditEnv(k=k, seed=seed)
    agent = EpsilonGreedyAgent(k=k, epsilon=epsilon, seed=seed + 1)

    total_reward = 0.0

    for _ in range(steps):
        action = agent.choose_action()
        reward = env.step(action)
        agent.update(action, reward)
        total_reward += reward

    print("True action values:")
    print([round(value, 3) for value in env.true_values])

    print("\nEstimated action values:")
    print([round(value, 3) for value in agent.q_values])

    print("\nAction counts:")
    print(agent.action_counts)

    print("\nTotal reward:")
    print(round(total_reward, 3))


if __name__ == "__main__":
    run_experiment()
