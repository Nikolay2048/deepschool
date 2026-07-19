"""Q-learning agent for GridWorld.

Run:
    python labs/rl/exercises/gridworld/q_learning.py
"""

from __future__ import annotations

import random

from gridworld import ACTIONS, DOWN, LEFT, RIGHT, UP, Action, GridWorldEnv, State

ARROWS: dict[Action, str] = {
    UP: "^",
    DOWN: "v",
    LEFT: "<",
    RIGHT: ">",
}


class QLearningAgent:
    """Tabular Q-learning agent."""

    def __init__(
            self,
            actions: list[Action],
            alpha: float = 0.2,
            gamma: float = 0.95,
            epsilon: float = 0.2,
            seed: int = 42,
    ):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")

        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.q_table: dict[State, dict[Action, float]] = {}

    def get_q_values(self, state: State) -> dict[Action, float]:
        """Return Q-values for state, creating zero values when needed."""
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
        return self.q_table[state]

    def choose_action(self, state: State) -> Action:
        """Choose action using epsilon-greedy strategy."""
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        return self.best_action(state)

    def best_action(self, state: State) -> Action:
        """Return action with the highest Q-value for state."""
        q_values = self.get_q_values(state)
        return max(q_values, key=q_values.get)

    def update(
            self,
            state: State,
            action: Action,
            reward: float,
            next_state: State,
            done: bool,
    ) -> None:
        """Apply one Q-learning update."""
        q_values = self.get_q_values(state)
        old_q = q_values[action]

        if done:
            target = reward
        else:
            next_q_values = self.get_q_values(next_state)
            next_best_q = max(next_q_values.values())
            target = reward + self.gamma * next_best_q
        new_q = old_q + self.alpha * (target - old_q)
        q_values[action] = new_q


def train_agent(
        episodes: int = 500,
        max_steps: int = 50,
        seed: int = 42,
) -> QLearningAgent:
    env = GridWorldEnv()
    agent = QLearningAgent(actions=ACTIONS, seed=seed)

    for _ in range(episodes):
        state = env.reset()

        for _ in range(max_steps):
            action = agent.choose_action(state)
            result = env.step(action)

            agent.update(
                state=state,
                action=action,
                reward=result.reward,
                next_state=result.state,
                done=result.done,
            )

            state = result.state

            if result.done:
                break

    return agent


def render_policy(env: GridWorldEnv, agent: QLearningAgent) -> str:
    """Render greedy policy as arrows."""
    lines = []

    for row in range(env.rows):
        cells = []
        for col in range(env.cols):
            state = (row, col)

            if state == env.goal:
                cells.append("G")
            elif state in env.walls:
                cells.append("#")
            else:
                action = agent.best_action(state)
                cells.append(ARROWS[action])

        lines.append(" ".join(cells))

    return "\n".join(lines)


def run_greedy_episode(
        agent: QLearningAgent,
        max_steps: int = 20,
) -> None:
    """Run one episode with greedy actions and print the path."""
    env = GridWorldEnv()
    state = env.reset()

    print("Greedy episode")
    print(env.render())
    print()

    for step_idx in range(1, max_steps + 1):
        action = agent.best_action(state)
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

        state = result.state
        if result.done:
            break


def main() -> None:
    env = GridWorldEnv()
    agent = train_agent()

    print("Learned greedy policy")
    print("---------------------")
    print(render_policy(env, agent))
    print()

    run_greedy_episode(agent)


if __name__ == "__main__":
    main()
