# Лекция 13. SARSA vs Q-learning

Q-learning и SARSA очень похожи.

Оба учат Q-table:

```text
Q(state, action)
```

Оба используют идею:

```text
старую Q-оценку двигаем к target
```

Но target считается по-разному.

## Q-learning

Q-learning смотрит на лучшее возможное действие в следующем состоянии:

```text
target = reward + gamma * max_a Q(next_state, a)
```

То есть он обновляется так, будто дальше агент выберет greedy action.

Даже если во время training policy иногда делает exploration, target все равно использует лучший action.

Поэтому Q-learning называют **off-policy**.

Он учит greedy policy, даже когда поведение во время сбора опыта epsilon-greedy.

## SARSA

SARSA использует действие, которое policy реально выбрала в следующем состоянии:

```text
target = reward + gamma * Q(next_state, next_action)
```

Где `next_action` выбран той же epsilon-greedy policy.

Название SARSA идет от цепочки:

```text
State, Action, Reward, next State, next Action
```

То есть:

```text
S A R S A
```

## Главное отличие

Q-learning:

```text
что было бы, если дальше выбрать лучшее действие?
```

SARSA:

```text
что будет, если дальше действовать так, как реально действует моя текущая policy?
```

## On-policy и off-policy

### On-policy

Алгоритм учит ту же policy, которой реально действует.

SARSA - on-policy.

Если policy исследует и иногда делает случайные действия, SARSA учитывает это в обучении.

### Off-policy

Алгоритм может действовать одной policy, а учить другую.

Q-learning - off-policy.

Он может собирать опыт epsilon-greedy поведением, но учить greedy policy.

## Почему SARSA может быть осторожнее

В рискованных или stochastic средах exploration может привести к плохим последствиям.

SARSA учитывает, что агент иногда будет исследовать.

Поэтому learned policy может быть более осторожной.

Q-learning смотрит на лучший будущий action и может быть более оптимистичным.

## Формулы рядом

Q-learning:

```text
target = reward + gamma * max_a Q(next_state, a)
Q(state, action) += alpha * (target - Q(state, action))
```

SARSA:

```text
target = reward + gamma * Q(next_state, next_action)
Q(state, action) += alpha * (target - Q(state, action))
```

Если `done=True`, в обоих случаях:

```text
target = reward
```

Потому что будущего уже нет.

## Что нужно понять

- Q-learning использует лучший будущий Q.
- SARSA использует Q следующего action, который реально выбрала policy.
- Q-learning - off-policy.
- SARSA - on-policy.
- В deterministic простой среде они могут давать похожий результат.
- В stochastic или рискованных средах различие становится важнее.

