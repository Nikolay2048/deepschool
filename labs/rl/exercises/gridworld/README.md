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
```

## Ожидаемая идея

Пока здесь нет обучения. Мы сначала строим среду.

Агент будет делать случайные действия, а ты увидишь, как меняются:

- `state`;
- `action`;
- `reward`;
- `next_state`;
- `done`.

