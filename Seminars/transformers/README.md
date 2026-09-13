# Transformers — учебный трек

Практический трек: от текста и embeddings до собственного маленького decoder-only Transformer.

## Занятия

| № | Тема | Статус |
|---:|---|---|
| 01 | [Tokens, token IDs и embeddings](01-tokens-and-embeddings.ipynb) | Пройдено; повторяем по ходу |
| 02 | [Learned positional embeddings и broadcasting](02-positional-embeddings.ipynb) | Пройдено; повторяем по ходу |
| 03 | [Query, Key, Value и single-head self-attention](03-single-head-self-attention.ipynb) | Практика выполнена, ответы разобраны с ментором |
| 04 | [Causal mask](04-causal-mask.ipynb) | Практика выполнена, ответы разобраны |
| 04b | [Attention как nn.Module](04b-attention-module.ipynb) | Выполнено: эталон, причинность, градиенты, SGD и state_dict проверены |
| 05 | [Multi-head attention: отдельные головы](05-multi-head-attention.ipynb) | Выполнено: формы, причинность и градиенты проверены; ответы разобраны |
| 05b | [Оси голов: reshape и transpose](05b-head-axes.ipynb) | Выполнено: группировка и обратная сборка проверены |
| 05c | [Векторизованный attention](05c-vectorized-attention.ipynb) | Выполнено: Q/K/V, forward, причинность и градиенты совпали с эталоном |
| 06 | Transformer block | Запланировано |
| 07 | Tiny decoder-only language model | Запланировано |

Фундамент PyTorch и математики повторяется по мере необходимости внутри занятий.

Формат проверки ответов: сохраняем исходный раздел «Мои ответы». Исправления и недостающие объяснения добавляем ниже отдельным разделом «Разбор ментора», с указанием, что верно и что нужно уточнить. Ответы ученика не заменяем готовыми без отдельной просьбы.

Лекция к уроку 05: [Multi-head attention](../../notes/interview-prep/08d-multi-head-attention.md).

Лекция к уроку 04b: [Свой модуль PyTorch](../../notes/interview-prep/08c-attention-module.md).

Scaling и softmax уже реализованы в уроке 03. Подробный конспект: [Self-attention](../../notes/interview-prep/08-self-attention.md), включая размерности, чтение индексов и обучение проекций. Готовые ответы не заменяют повторную самостоятельную проверку.

Лекция к уроку 04: [Causal mask](../../notes/interview-prep/08b-causal-mask.md). После сборки модели: отдельные этапы обучения и validation, генерации, экспериментов и оформления портфолио. Размер модели выберем после проверки оборудования.
