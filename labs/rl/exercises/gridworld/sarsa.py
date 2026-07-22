"""SARSA vs Q-learning comparison for GridWorld.

Run:
    python labs/rl/exercises/gridworld/sarsa.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from gridworld import ACTIONS, Action, GridWorldEnv, State
from q_learning import QLearningAgent, render_policy
from stochastic_gridworld import EnvFactory, StochasticGridWorldEnv, evaluate_agent
from training_curve import rolling_average


class SarsaAgent(QLearningAgent):
    """Tabular SARSA agent.

    Reuses epsilon-greedy action selection and Q-table storage from
    QLearningAgent, but updates with Q(next_state, next_action).
    """

    def update_sarsa(
            self,
            state: State,
            action: Action,
            reward: float,
            next_state: State,
            next_action: Action | None,
            done: bool,
    ) -> None:
        """Apply one SARSA update."""
        q_values = self.get_q_values(state)
        old_q = q_values[action]

        if done:
            target = reward
        else:
            if next_action is None:
                raise ValueError("next_action is required when done=False")

            next_q_values = self.get_q_values(next_state)
            target = reward + self.gamma * next_q_values[next_action]

        q_values[action] = old_q + self.alpha * (target - old_q)


def deterministic_env_factory(seed: int) -> GridWorldEnv:
    return GridWorldEnv()


def stochastic_env_factory(seed: int) -> GridWorldEnv:
    return StochasticGridWorldEnv(slip_probability=0.2, seed=seed)


def train_sarsa_in_env(
        env_factory: EnvFactory,
        episodes: int = 500,
        max_steps: int = 50,
        seed: int = 42,
) -> tuple[SarsaAgent, dict[str, list[float]]]:
    """Train SARSA agent in provided environment factory."""
    agent = SarsaAgent(actions=ACTIONS, epsilon=0.2, seed=seed)

    episode_rewards: list[float] = []
    episode_steps: list[float] = []
    episode_successes: list[float] = []

    for episode_idx in range(episodes):
        env = env_factory(seed + episode_idx)
        state = env.reset()
        action = agent.choose_action(state)

        total_reward = 0.0
        success = 0.0

        for step_idx in range(1, max_steps + 1):
            result = env.step(action)
            total_reward += result.reward

            if result.done:
                next_action = None
            else:
                next_action = agent.choose_action(result.state)

            agent.update_sarsa(
                state=state,
                action=action,
                reward=result.reward,
                next_state=result.state,
                next_action=next_action,
                done=result.done,
            )

            if result.done:
                success = 1.0
                episode_steps.append(float(step_idx))
                break

            state = result.state
            action = next_action
        else:
            episode_steps.append(float(max_steps))

        episode_rewards.append(total_reward)
        episode_successes.append(success)

    return agent, {
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
        "episode_successes": episode_successes,
    }


def train_q_learning_in_env(
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


def plot_algorithm_comparison(
        q_history: dict[str, list[float]],
        sarsa_history: dict[str, list[float]],
        env_name: str,
        window: int = 20,
) -> Path:
    output_path = Path("outputs") / "rl" / f"gridworld-sarsa-vs-q-learning-{env_name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    episodes = len(q_history["episode_rewards"])
    x = list(range(1, episodes + 1))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for label, history in [("Q-learning", q_history), ("SARSA", sarsa_history)]:
        rewards = rolling_average(history["episode_rewards"], window=window)
        successes = rolling_average(history["episode_successes"], window=window)
        steps = rolling_average(history["episode_steps"], window=window)

        axes[0].plot(x, rewards, label=label)
        axes[1].plot(x, successes, label=label)
        axes[2].plot(x, steps, label=label)

    axes[0].set_title(f"SARSA vs Q-learning on {env_name} GridWorld")
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
        f"{name:<10} | "
        f"{metrics['success_rate']:>12.3f} | "
        f"{metrics['average_steps']:>13.3f} | "
        f"{metrics['average_reward']:>14.3f}"
    )


def run_comparison(env_name: str, env_factory: EnvFactory) -> Path:
    q_agent, q_history = train_q_learning_in_env(env_factory)
    sarsa_agent, sarsa_history = train_sarsa_in_env(env_factory)

    print(f"{env_name.title()} GridWorld")
    print("-" * (len(env_name) + 10))
    print("Q-learning policy:")
    print(render_policy(GridWorldEnv(), q_agent))
    print()
    print("SARSA policy:")
    print(render_policy(GridWorldEnv(), sarsa_agent))
    print()

    q_metrics = evaluate_agent(q_agent, env_factory)
    sarsa_metrics = evaluate_agent(sarsa_agent, env_factory)

    print("algorithm  | success_rate | average_steps | average_reward")
    print("-" * 61)
    print_metrics("Q-learning", q_metrics)
    print_metrics("SARSA", sarsa_metrics)
    print()

    return plot_algorithm_comparison(
        q_history,
        sarsa_history,
        env_name=env_name,
    )


def main() -> None:
    deterministic_path = run_comparison("deterministic", deterministic_env_factory)
    stochastic_path = run_comparison("stochastic", stochastic_env_factory)

    print(f"Saved deterministic comparison to {deterministic_path}")
    print(f"Saved stochastic comparison to {stochastic_path}")


if __name__ == "__main__":
    main()
