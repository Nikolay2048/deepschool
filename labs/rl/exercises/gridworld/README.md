# GridWorld

Первая среда со состояниями.

В bandit-задаче было:

```text
action -> reward
```

В GridWorld будет:

```text
state -> action -> reward -> next_state
```

## Файлы

- `gridworld.py` - простая среда GridWorld.
- `random_policy.py` - baseline без обучения: агент выбирает случайные действия.
- `q_learning.py` - табличный Q-learning агент для GridWorld.
- `evaluate_q_learning.py` - сравнение random policy и learned greedy policy.
- `training_curve.py` - графики процесса обучения Q-learning по эпизодам.

## Что нужно сделать

Допиши TODO в `gridworld.py`:

1. Реализуй движение по действиям `up`, `down`, `left`, `right`.
2. Запрети выход за границы поля.
3. Запрети проход сквозь стены.
4. Верни правильный `reward`.
5. Верни `done = True`, когда агент дошел до цели.

## Запуск

```powershell
python labs/rl/exercises/gridworld/gridworld.py
python labs/rl/exercises/gridworld/random_policy.py
python labs/rl/exercises/gridworld/q_learning.py
python labs/rl/exercises/gridworld/evaluate_q_learning.py
python labs/rl/exercises/gridworld/training_curve.py
```

## Ожидаемая идея

Пока здесь нет обучения. Мы сначала строим среду.

Агент будет делать случайные действия, а ты увидишь, как меняются:

- `state`;
- `action`;
- `reward`;
- `next_state`;
- `done`.

Следующий шаг - измерить random policy baseline: как часто случайный агент доходит до цели и какую среднюю награду получает.

После baseline переходим к Q-learning: агент будет учиться выбирать действия по Q-table.

После Q-learning сравниваем обученную greedy policy с random policy по тем же метрикам.

Training curve показывает, как reward, success rate и длина эпизода менялись во время обучения.
