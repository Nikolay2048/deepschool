"""Evaluate learned Q-learning policy against GridWorld metrics.

Run:
    python labs/rl/exercises/gridworld/evaluate_q_learning.py
"""

from __future__ import annotations

from gridworld import GridWorldEnv
from q_learning import QLearningAgent, render_policy, train_agent
from random_policy import evaluate_random_policy


def run_greedy_episode(
    agent: QLearningAgent,
    max_steps: int,
) -> dict[str, float | int | bool]:
    """Run one episode using greedy actions from learned Q-table."""
    env = GridWorldEnv()
    state = env.reset()
    total_reward = 0.0

    for step_idx in range(1, max_steps + 1):
        action = agent.best_action(state)
        result = env.step(action)
        total_reward += result.reward
        state = result.state

        if result.done:
            return {
                "success": True,
                "steps": step_idx,
                "total_reward": total_reward,
            }

    return {
        "success": False,
        "steps": max_steps,
        "total_reward": total_reward,
    }


def evaluate_greedy_policy(
    agent: QLearningAgent,
    episodes: int = 100,
    max_steps: int = 50,
) -> dict[str, float]:
    """Evaluate learned greedy policy over many episodes."""
    successes = 0
    total_steps = 0
    total_reward = 0.0

    for _ in range(episodes):
        episode = run_greedy_episode(agent, max_steps)
        total_steps += episode["steps"]
        total_reward += episode["total_reward"]
        successes += episode["success"]

    return {
        "episodes": episodes,
        "success_rate": successes / episodes,
        "average_steps": total_steps / episodes,
        "average_reward": total_reward / episodes,
    }


def print_comparison(
    random_results: dict[str, float],
    learned_results: dict[str, float],
) -> None:
    print("Policy comparison")
    print("-----------------")
    print("policy   | success_rate | average_steps | average_reward")
    print("-" * 60)
    print(
        f"random   | "
        f"{random_results['success_rate']:>12.3f} | "
        f"{random_results['average_steps']:>13.3f} | "
        f"{random_results['average_reward']:>14.3f}"
    )
    print(
        f"learned  | "
        f"{learned_results['success_rate']:>12.3f} | "
        f"{learned_results['average_steps']:>13.3f} | "
        f"{learned_results['average_reward']:>14.3f}"
    )


def main() -> None:
    episodes = 100
    max_steps = 50

    agent = train_agent(episodes=500, max_steps=max_steps, seed=42)
    env = GridWorldEnv()

    print("Learned greedy policy")
    print("---------------------")
    print(render_policy(env, agent))
    print()

    random_results = evaluate_random_policy(
        episodes=episodes,
        max_steps=max_steps,
        seed=42,
    )
    learned_results = evaluate_greedy_policy(
        agent=agent,
        episodes=episodes,
        max_steps=max_steps,
    )

    print_comparison(random_results, learned_results)


if __name__ == "__main__":
    main()
