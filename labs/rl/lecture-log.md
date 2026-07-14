# RL Lecture Log

Журнал лекций и практики. Нужен, чтобы быстро видеть, что уже пройдено и какой следующий шаг.

| # | Статус | Материал | Практика | Ключевые идеи |
|---|--------|----------|----------|---------------|
| 01 | В процессе | `lectures/01-intro.md` | `exercises/bandit/bandit.py` | agent, environment, action, reward, policy, exploration vs exploitation, multi-armed bandit |
| 02 | В процессе | `lectures/02-exploration-vs-exploitation.md` | `exercises/bandit/compare_epsilon.py` | epsilon-greedy, exploration, exploitation, total reward, average reward, best action rate, learning curve |
| 03 | Новая | `lectures/03-multiple-runs.md` | `exercises/bandit/multiple_runs.py` | independent runs, seed, mean, standard deviation, stability |
| 04 | Новая | `lectures/04-learning-curves.md` | `exercises/bandit/learning_curves.py` | cumulative average reward, learning curve, noisy reward, visualization |
| 05 | Новая | `lectures/05-gridworld-and-mdp.md` | `exercises/gridworld/gridworld.py` | GridWorld, MDP, state, transition, terminal state |

## Следующее

После GridWorld environment:

1. Проверить ручные переходы среды.
2. Добавить random policy baseline.
3. Перейти к value function и Q-table.
