# Лекция 12. Stochastic GridWorld

До сих пор GridWorld был deterministic.

Это значит:

```text
если агент выбрал right, он точно пошел right
```

В реальных задачах так бывает не всегда.

Робот может проскользнуть. Среда может быть шумной. Пользователь может отреагировать не так, как ожидалось. Действие может иметь вероятностный результат.

## Что такое stochastic transition

Stochastic transition - это переход, где одно и то же действие не всегда приводит к одному и тому же следующему состоянию.

Например:

```text
агент выбирает up
с вероятностью 0.8 реально идет up
с вероятностью 0.2 действие заменяется случайным
```

В коде это можно представить так:

```python
if rng.random() < slip_probability:
    actual_action = rng.choice(ACTIONS)
else:
    actual_action = intended_action
```

## Intended action и actual action

Важно различать:

- `intended_action` - что хотел сделать агент;
- `actual_action` - что реально произошло в среде.

Q-learning обновляет Q-value для action, который выбрал агент.

Но reward и next_state приходят из среды после stochastic transition.

## Почему задача становится сложнее

В deterministic GridWorld оптимальный путь стабилен.

В stochastic GridWorld даже хорошая policy иногда может:

- удариться в стену;
- сделать лишний шаг;
- не дойти до цели за `max_steps`;
- получить меньший reward.

Поэтому evaluation может стать хуже, хотя агент выучил разумную стратегию.

## Что изменится на графиках

Обычно stochastic среда дает:

- более шумный training reward;
- более длинные episodes;
- иногда меньший success rate;
- менее резкие и менее красивые learning curves.

Это нормально.

## Главная идея

Deterministic среда полезна для понимания алгоритма.

Stochastic среда ближе к реальности:

```text
агент выбирает действие
но среда не обязана выполнить его идеально
```

Следующий большой шаг после этого - сравнить Q-learning и SARSA, потому что в stochastic задачах различие between off-policy и on-policy становится важнее.

