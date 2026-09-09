# Transformers — учебный трек

Практический трек: от текста и embeddings до собственного маленького decoder-only Transformer.

## Занятия

| № | Тема | Статус |
|---:|---|---|
| 01 | [Tokens, token IDs и embeddings](01-tokens-and-embeddings.ipynb) | Пройдено; повторяем по ходу |
| 02 | [Learned positional embeddings и broadcasting](02-positional-embeddings.ipynb) | Пройдено; повторяем по ходу |
| 03 | [Query, Key, Value и single-head self-attention](03-single-head-self-attention.ipynb) | Практика выполнена, ответы разобраны с ментором |
| 04 | [Causal mask](04-causal-mask.ipynb) и затем упаковка attention в nn.Module | Начат: остановка A |
| 05 | Multi-head attention | Запланировано |
| 06 | Transformer block | Запланировано |
| 07 | Tiny decoder-only language model | Запланировано |

Фундамент PyTorch и математики повторяется по мере необходимости внутри занятий.

Scaling и softmax уже реализованы в уроке 03. Подробный конспект: [Self-attention](../../notes/interview-prep/08-self-attention.md), включая размерности, чтение индексов и обучение проекций. Готовые ответы не заменяют повторную самостоятельную проверку.

Лекция к уроку 04: [Causal mask](../../notes/interview-prep/08b-causal-mask.md). После сборки модели: отдельные этапы обучения и validation, генерации, экспериментов и оформления портфолио. Размер модели выберем после проверки оборудования.
