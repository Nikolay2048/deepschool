# Задание 02. Сравнение epsilon

Цель: увидеть, как exploration/exploitation влияет на результат bandit-агента.

## Что нужно сделать

Допиши `compare_epsilon.py`.

Нужно сравнить несколько значений:

```python
epsilons = [0.0, 0.01, 0.1, 0.3, 1.0]
```

Для каждого `epsilon` запусти bandit-эксперимент и выведи:

- `epsilon`;
- `total_reward`;
- `average_reward`;
- `best_action`;
- `best_action_count`;
- `best_action_rate`.

## Что важно заметить

После запуска попробуй ответить:

1. Какой `epsilon` дал лучший `total_reward`?
2. Что произошло при `epsilon = 0.0`?
3. Что произошло при `epsilon = 1.0`?
4. Почему `best_action_rate` полезнее, чем просто `total_reward`?

## Запуск

```powershell
python labs/rl/exercises/02-epsilon-comparison/compare_epsilon.py
```

## Подсказка

Можно импортировать классы из первого задания не через обычный import, потому что папка `01-bandit` начинается с цифры и содержит дефис.

В стартовом коде уже есть функция `load_bandit_module()`.

