"""Visualize learned GridWorld value function.

Run:
    python labs/rl/exercises/gridworld/value_heatmap.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gridworld import GridWorldEnv
from q_learning import ARROWS, QLearningAgent, render_policy, train_agent


def state_value(agent: QLearningAgent, state: tuple[int, int]) -> float:
    """Compute V(state) = max_a Q(state, action)."""
    q_values = agent.get_q_values(state)
    return max(q_values.values())


def build_value_grid(env: GridWorldEnv, agent: QLearningAgent) -> np.ndarray:
    """Build 2D array of values for heatmap."""
    values = np.full((env.rows, env.cols), np.nan)

    for row in range(env.rows):
        for col in range(env.cols):
            state = (row, col)

            if state in env.walls:
                continue
            if state == env.goal:
                values[row, col] = env.goal_reward
            else:
                values[row, col] = state_value(agent, state)

    return values


def plot_value_heatmap(agent: QLearningAgent) -> Path:
    env = GridWorldEnv()
    values = build_value_grid(env, agent)

    output_path = Path("outputs") / "rl" / "gridworld-value-heatmap.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad(color="#d0d0d0")

    fig, ax = plt.subplots(figsize=(6, 6))
    image = ax.imshow(values, cmap=cmap)

    for row in range(env.rows):
        for col in range(env.cols):
            state = (row, col)

            if state in env.walls:
                label = "#"
            elif state == env.goal:
                label = "G"
            else:
                action = agent.best_action(state)
                label = f"{ARROWS[action]}\n{values[row, col]:.2f}"

            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                color="white" if state not in env.walls else "black",
                fontsize=12,
                fontweight="bold",
            )

    ax.set_title("GridWorld learned value function")
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="V(state)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def main() -> None:
    agent = train_agent(episodes=500, max_steps=50, seed=42)
    env = GridWorldEnv()

    print("Learned greedy policy")
    print("---------------------")
    print(render_policy(env, agent))
    print()

    output_path = plot_value_heatmap(agent)
    print(f"Saved value heatmap to {output_path}")


if __name__ == "__main__":
    main()

