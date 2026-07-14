"""Compare epsilon values for the multi-armed bandit exercise.

Run:
    python labs/rl/exercises/bandit/compare_epsilon.py
"""

from __future__ import annotations

from bandit import BanditEnv, EpsilonGreedyAgent


def run_single_experiment(
    epsilon: float,
    k: int = 10,
    steps: int = 1000,
    seed: int = 42,
) -> dict[str, float | int]:
    env = BanditEnv(k=k, seed=seed)
    agent = EpsilonGreedyAgent(k=k, epsilon=epsilon, seed=seed + 1)

    total_reward = 0.0
    best_action = env.true_values.index(max(env.true_values))

    for _ in range(steps):
        action = agent.choose_action()
        reward = env.step(action)
        agent.update(action, reward)
        total_reward += reward

    best_action_count = agent.action_counts[best_action]

    return {
        "epsilon": epsilon,
        "total_reward": total_reward,
        "average_reward": total_reward / steps,
        "best_action": best_action,
        "best_action_count": best_action_count,
        "best_action_rate": best_action_count / steps,
    }


def print_results(results: list[dict[str, float | int]]) -> None:
    print("epsilon | total_reward | avg_reward | best_action | best_count | best_rate")
    print("-" * 76)

    for row in results:
        print(
            f"{row['epsilon']:>7.2f} | "
            f"{row['total_reward']:>12.3f} | "
            f"{row['average_reward']:>10.3f} | "
            f"{row['best_action']:>11} | "
            f"{row['best_action_count']:>10} | "
            f"{row['best_action_rate']:>9.3f}"
        )


def main() -> None:
    epsilons = [0.0, 0.01, 0.1, 0.3, 1.0]

    results = []

    for epsilon in epsilons:
        result = run_single_experiment(epsilon=epsilon)
        results.append(result)

    print_results(results)


if __name__ == "__main__":
    main()
