# RL Lecture Log

Журнал лекций и практики. Нужен, чтобы быстро видеть, что уже пройдено и какой следующий шаг.

| # | Статус | Материал | Практика | Ключевые идеи |
|---|--------|----------|----------|---------------|
| 01 | В процессе | `lectures/01-intro.md` | `exercises/bandit/bandit.py` | agent, environment, action, reward, policy, exploration vs exploitation, multi-armed bandit |
| 02 | В процессе | `lectures/02-exploration-vs-exploitation.md` | `exercises/bandit/compare_epsilon.py` | epsilon-greedy, exploration, exploitation, total reward, average reward, best action rate, learning curve |
| 03 | Новая | `lectures/03-multiple-runs.md` | `exercises/bandit/multiple_runs.py` | independent runs, seed, mean, standard deviation, stability |
| 04 | Новая | `lectures/04-learning-curves.md` | `exercises/bandit/learning_curves.py` | cumulative average reward, learning curve, noisy reward, visualization |
| 05 | Новая | `lectures/05-gridworld-and-mdp.md` | `exercises/gridworld/gridworld.py` | GridWorld, MDP, state, transition, terminal state |
| 06 | Новая | `lectures/06-random-policy-baseline.md` | `exercises/gridworld/random_policy.py` | random policy, baseline, success rate, episode length |
| 07 | Новая | `lectures/07-q-learning.md` | `exercises/gridworld/q_learning.py` | Q-table, Q-learning, alpha, gamma, greedy policy |
| 08 | Новая | `lectures/08-evaluating-policy.md` | `exercises/gridworld/evaluate_q_learning.py` | training vs evaluation, learned policy, policy comparison |
| 09 | Новая | `lectures/09-gridworld-training-curve.md` | `exercises/gridworld/training_curve.py` | training curve, rolling average, reward, success rate |
| 10 | Новая | `lectures/10-value-heatmap.md` | `exercises/gridworld/value_heatmap.py` | value function, V(state), heatmap, policy arrows |

## Следующее

После value heatmap:

1. Перейти к exploration decay.
2. Сделать stochastic GridWorld.
3. Подготовить переход к Gymnasium/FrozenLake.
