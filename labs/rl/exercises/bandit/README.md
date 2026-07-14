# Bandit Exercises

Рабочая папка для всех упражнений по multi-armed bandit.

## Файлы

- `bandit.py` - базовая среда `BanditEnv` и `EpsilonGreedyAgent`.
- `compare_epsilon.py` - сравнение нескольких значений `epsilon` на одном run.
- `multiple_runs.py` - сравнение `epsilon` по нескольким независимым runs.
- `learning_curves.py` - график cumulative average reward для разных `epsilon`.

## Порядок работы

1. Дописать и понять `bandit.py`.
2. Запустить сравнение `epsilon` в `compare_epsilon.py`.
3. Усреднить результаты по нескольким runs в `multiple_runs.py`.
4. Построить learning curves в `learning_curves.py`.

## Запуск

```powershell
python labs/rl/exercises/bandit/bandit.py
python labs/rl/exercises/bandit/compare_epsilon.py
python labs/rl/exercises/bandit/multiple_runs.py
python labs/rl/exercises/bandit/learning_curves.py
```
