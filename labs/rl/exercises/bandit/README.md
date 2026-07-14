# Bandit Exercises

Рабочая папка для всех упражнений по multi-armed bandit.

## Файлы

- `bandit.py` - базовая среда `BanditEnv` и `EpsilonGreedyAgent`.
- `compare_epsilon.py` - сравнение нескольких значений `epsilon` на одном run.
- `multiple_runs.py` - сравнение `epsilon` по нескольким независимым runs.

## Порядок работы

1. Дописать и понять `bandit.py`.
2. Запустить сравнение `epsilon` в `compare_epsilon.py`.
3. Усреднить результаты по нескольким runs в `multiple_runs.py`.

## Запуск

```powershell
python labs/rl/exercises/bandit/bandit.py
python labs/rl/exercises/bandit/compare_epsilon.py
python labs/rl/exercises/bandit/multiple_runs.py
```

