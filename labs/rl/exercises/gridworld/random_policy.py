"""Random policy baseline for GridWorld.

Run:
    python labs/rl/exercises/gridworld/random_policy.py
"""

from __future__ import annotations

import random

from gridworld import ACTIONS, GridWorldEnv


def run_episode(
        env: GridWorldEnv,
        rng: random.Random,
        max_steps: int,
) -> dict[str, float | int | bool]:
    """Run one random-policy episode."""
    env.reset()

    total_reward = 0.0

    for step_idx in range(1, max_steps + 1):
        # TODO: choose random action from ACTIONS
        action = rng.choice(ACTIONS)
        result = env.step(action)
        total_reward += result.reward
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


def evaluate_random_policy(
        episodes: int = 100,
        max_steps: int = 50,
        seed: int = 42,
) -> dict[str, float]:
    """Evaluate random policy over many episodes."""
    rng = random.Random(seed)
    env = GridWorldEnv()

    successes = 0
    total_steps = 0
    total_reward = 0.0

    for _ in range(episodes):
        episode = run_episode(env, rng, max_steps)
        total_steps += episode["steps"]
        total_reward += episode["total_reward"]
        successes += episode["success"]

    return {
        "episodes": episodes,
        "success_rate": successes / episodes,
        "average_steps": total_steps / episodes,
        "average_reward": total_reward / episodes,
    }


def main() -> None:
    results = evaluate_random_policy()

    print("Random policy baseline")
    print("----------------------")
    print(f"episodes:       {results['episodes']:.0f}")
    print(f"success_rate:   {results['success_rate']:.3f}")
    print(f"average_steps:  {results['average_steps']:.3f}")
    print(f"average_reward: {results['average_reward']:.3f}")


if __name__ == "__main__":
    main()
