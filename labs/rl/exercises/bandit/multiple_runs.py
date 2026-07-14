"""Compare epsilon values across multiple independent bandit runs.

Run:
    python labs/rl/exercises/bandit/multiple_runs.py
"""

from __future__ import annotations

import statistics

from compare_epsilon import run_single_experiment


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def summarize_epsilon(
        epsilon: float,
        runs: int,
        steps: int,
        base_seed: int = 1000,
) -> dict[str, float]:
    total_rewards: list[float] = []
    average_rewards: list[float] = []
    best_action_rates: list[float] = []

    for run_idx in range(runs):
        seed = base_seed + run_idx

        single_run = run_single_experiment(epsilon=epsilon, seed=seed, steps=steps)
        total_rewards.append(single_run["total_reward"])
        average_rewards.append(single_run["average_reward"])
        best_action_rates.append(single_run["best_action_rate"])

    return {
        "epsilon": epsilon,
        "mean_total_reward": mean(total_rewards),
        "std_total_reward": statistics.stdev(total_rewards),
        "mean_average_reward": mean(average_rewards),
        "mean_best_action_rate": mean(best_action_rates),
    }


def print_summary(rows: list[dict[str, float]]) -> None:
    print("epsilon | mean_total | std_total | mean_avg | mean_best_rate")
    print("-" * 68)

    for row in rows:
        print(
            f"{row['epsilon']:>7.2f} | "
            f"{row['mean_total_reward']:>10.3f} | "
            f"{row['std_total_reward']:>9.3f} | "
            f"{row['mean_average_reward']:>8.3f} | "
            f"{row['mean_best_action_rate']:>14.3f}"
        )


def main() -> None:
    epsilons = [0.0, 0.01, 0.1, 0.3, 1.0]
    runs = 50
    steps = 1000

    rows = []

    for epsilon in epsilons:
        rows.append(summarize_epsilon(epsilon, runs, steps, base_seed=0))

    print_summary(rows)


if __name__ == "__main__":
    main()
