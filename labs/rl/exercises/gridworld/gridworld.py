"""A tiny GridWorld environment.

Run:
    python labs/rl/exercises/gridworld/gridworld.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass

Action = str
State = tuple[int, int]

UP: Action = "up"
DOWN: Action = "down"
LEFT: Action = "left"
RIGHT: Action = "right"

ACTIONS: list[Action] = [UP, DOWN, LEFT, RIGHT]


@dataclass(frozen=True)
class StepResult:
    state: State
    reward: float
    done: bool


class GridWorldEnv:
    """Small deterministic grid world with walls and one goal."""

    def __init__(
            self,
            rows: int = 4,
            cols: int = 4,
            start: State = (0, 0),
            goal: State = (3, 3),
            walls: set[State] | None = None,
            step_reward: float = -0.01,
            wall_reward: float = -0.05,
            goal_reward: float = 1.0,
    ):
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")

        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.walls = walls or {(1, 1), (2, 1)}
        self.step_reward = step_reward
        self.wall_reward = wall_reward
        self.goal_reward = goal_reward

        self._validate_cell(start, "start")
        self._validate_cell(goal, "goal")
        for wall in self.walls:
            self._validate_cell(wall, "wall")

        if start == goal:
            raise ValueError("start and goal must be different")
        if start in self.walls:
            raise ValueError("start cannot be inside a wall")
        if goal in self.walls:
            raise ValueError("goal cannot be inside a wall")

        self.state = self.start

    def _validate_cell(self, cell: State, name: str) -> None:
        row, col = cell
        if not 0 <= row < self.rows or not 0 <= col < self.cols:
            raise ValueError(f"{name} cell {cell} is outside the grid")

    def reset(self) -> State:
        self.state = self.start
        return self.state

    def step(self, action: Action) -> StepResult:
        """Apply action and return next state, reward and done flag."""
        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action}")

        row, col = self.state

        # Compute candidate next position.
        if action == UP:
            next_state = (row - 1, col)
        elif action == DOWN:
            next_state = (row + 1, col)
        elif action == LEFT:
            next_state = (row, col - 1)
        else:  # RIGHT
            next_state = (row, col + 1)

        # If next_state is outside the grid, stay in place.
        next_row, next_col = next_state

        if not (0 <= next_row < self.rows and 0 <= next_col < self.cols):
            return StepResult(
                state=self.state,
                reward=self.wall_reward,
                done=False,
            )

        # If next_state is a wall, stay in place.
        if next_state in self.walls:
            return StepResult(
                state=self.state,
                reward=self.wall_reward,
                done=False,
            )

        # Move agent.
        self.state = next_state

        # Check whether goal was reached.
        if self.state == self.goal:
            return StepResult(
                state=self.state,
                reward=self.goal_reward,
                done=True,
            )

        return StepResult(
            state=self.state,
            reward=self.step_reward,
            done=False,
        )

    def render(self) -> str:
        lines = []

        for row in range(self.rows):
            cells = []
            for col in range(self.cols):
                cell = (row, col)

                if cell == self.state:
                    cells.append("A")
                elif cell == self.goal:
                    cells.append("G")
                elif cell in self.walls:
                    cells.append("#")
                else:
                    cells.append(".")

            lines.append(" ".join(cells))

        return "\n".join(lines)


def run_random_episode(seed: int = 42, max_steps: int = 20) -> None:
    rng = random.Random(seed)
    env = GridWorldEnv()

    state = env.reset()
    print("Initial state:", state)
    print(env.render())
    print()

    for step_idx in range(1, max_steps + 1):
        action = rng.choice(ACTIONS)
        result = env.step(action)

        print(
            f"step={step_idx:02d} "
            f"action={action:<5} "
            f"state={result.state} "
            f"reward={result.reward:.2f} "
            f"done={result.done}"
        )
        print(env.render())
        print()

        if result.done:
            break


if __name__ == "__main__":
    run_random_episode()
