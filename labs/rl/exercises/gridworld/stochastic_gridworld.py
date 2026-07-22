"""Stochastic GridWorld experiment.

The base GridWorldEnv is deterministic: intended action is always executed.
StochasticGridWorldEnv adds slip probability: sometimes the environment replaces
the intended action with a random action.

Run:
    python labs/rl/exercises/gridworld/stochastic_gridworld.py
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt

from gridworld import ACTIONS, Action, GridWorldEnv, StepResult
from q_learning import QLearningAgent, render_policy
from training_curve import rolling_average


class StochasticGridWorldEnv(GridWorldEnv):
    """GridWorld where intended action sometimes slips into a random action."""

    def __init__(
        self,
        slip_probability: float = 0.2,
        seed: int = 42,
        **kwargs,
    ):
        if not 0.0 <= slip_probability <= 1.0:
            raise ValueError("slip_probability must be in [0, 1]")

        super().__init__(**kwargs)
        self.slip_probability = slip_probability
        self.rng = random.Random(seed)

    def step(self, action: Action) -> StepResult:
        if self.rng.random() < self.slip_probability:
            actual_action = self.rng.choice(ACTIONS)
        else:
            actual_action = action

        return super().step(actual_action)


EnvFactory = Callable[[int], GridWorldEnv]


def train_agent_in_env(
    env_factory: EnvFactory,
    episodes: int = 500,
    max_steps: int = 50,
    seed: int = 42,
) -> tuple[QLearningAgent, dict[str, list[float]]]:
    """Train Q-learning agent in provided environment factory."""
    agent = QLearningAgent(actions=ACTIONS, epsilon=0.2, seed=seed)

    episode_rewards: list[float] = []
    episode_steps: list[float] = []
    episode_successes: list[float] = []

    for episode_idx in range(episodes):
        env = env_factory(seed + episode_idx)
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

    return agent, {
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
        "episode_successes": episode_successes,
    }


def evaluate_agent(
    agent: QLearningAgent,
    env_factory: EnvFactory,
    episodes: int = 100,
    max_steps: int = 50,
    seed: int = 10_000,
) -> dict[str, float]:
    """Evaluate greedy policy in environment factory."""
    successes = 0
    total_steps = 0
    total_reward = 0.0

    for episode_idx in range(episodes):
        env = env_factory(seed + episode_idx)
        state = env.reset()
        episode_reward = 0.0

        for step_idx in range(1, max_steps + 1):
            action = agent.best_action(state)
            result = env.step(action)
            episode_reward += result.reward
            state = result.state

            if result.done:
                successes += 1
                total_steps += step_idx
                total_reward += episode_reward
                break
        else:
            total_steps += max_steps
            total_reward += episode_reward

    return {
        "success_rate": successes / episodes,
        "average_steps": total_steps / episodes,
        "average_reward": total_reward / episodes,
    }


def deterministic_env_factory(seed: int) -> GridWorldEnv:
    return GridWorldEnv()


def stochastic_env_factory(seed: int) -> GridWorldEnv:
    return StochasticGridWorldEnv(slip_probability=0.2, seed=seed)


def plot_training_comparison(
    deterministic_history: dict[str, list[float]],
    stochastic_history: dict[str, list[float]],
    window: int = 20,
) -> Path:
    output_path = Path("outputs") / "rl" / "gridworld-stochastic-comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    episodes = len(deterministic_history["episode_rewards"])
    x = list(range(1, episodes + 1))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for label, history in [
        ("deterministic", deterministic_history),
        ("stochastic slip=0.2", stochastic_history),
    ]:
        rewards = rolling_average(history["episode_rewards"], window=window)
        successes = rolling_average(history["episode_successes"], window=window)
        steps = rolling_average(history["episode_steps"], window=window)

        axes[0].plot(x, rewards, label=label)
        axes[1].plot(x, successes, label=label)
        axes[2].plot(x, steps, label=label)

    axes[0].set_title(f"Deterministic vs stochastic GridWorld, rolling window={window}")
    axes[0].set_ylabel("Reward")
    axes[1].set_ylabel("Success rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("Steps")
    axes[2].set_xlabel("Episode")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(
        f"{name:<13} | "
        f"{metrics['success_rate']:>12.3f} | "
        f"{metrics['average_steps']:>13.3f} | "
        f"{metrics['average_reward']:>14.3f}"
    )


def main() -> None:
    deterministic_agent, deterministic_history = train_agent_in_env(
        deterministic_env_factory,
    )
    stochastic_agent, stochastic_history = train_agent_in_env(
        stochastic_env_factory,
    )

    print("Learned policies")
    print("----------------")
    print("Deterministic:")
    print(render_policy(GridWorldEnv(), deterministic_agent))
    print()
    print("Stochastic:")
    print(render_policy(GridWorldEnv(), stochastic_agent))
    print()

    deterministic_metrics = evaluate_agent(
        deterministic_agent,
        deterministic_env_factory,
    )
    stochastic_metrics = evaluate_agent(
        stochastic_agent,
        stochastic_env_factory,
    )

    print("Evaluation")
    print("----------")
    print("env           | success_rate | average_steps | average_reward")
    print("-" * 64)
    print_metrics("deterministic", deterministic_metrics)
    print_metrics("stochastic", stochastic_metrics)
    print()

    output_path = plot_training_comparison(
        deterministic_history,
        stochastic_history,
    )
    print(f"Saved stochastic comparison to {output_path}")


if __name__ == "__main__":
    main()

