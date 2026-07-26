# Карта подготовки и прогресс

## Цель

Подготовиться к собеседованиям на ML Engineer и LLM Engineer, попробовать элементы Research Engineering и собрать портфолио, которое показывает не только использование библиотек, но и понимание моделей, качество экспериментов и production-мышление.

Режим занятий: 60–90 минут в будни, более длинные проектные сессии на выходных.

## Стартовая диагностика

### Сильные стороны

- Уверенный практический Python и инженерный опыт на Java.
- Опыт автоматизации тестирования: декомпозиция, воспроизводимость, проверки и поиск граничных случаев.
- Базовое понимание классического ML.
- Практический опыт с RAG и агентскими системами.
- Хорошая интуиция в задачах на loss и градиентный спуск.

### Зоны роста

- Математика и механика backpropagation.
- PyTorch: autograd, режимы модели и полный training loop.
- Точная оценка ML- и RAG-систем.
- Python internals: iterable/iterator/generator, closures, GIL и concurrency.
- Строгая терминология для технического собеседования.
- Deep Learning: оптимизация, нормализация, архитектуры и Transformers.

## Учебные треки

### Deep Learning

- [x] Производная MSE для одномерной линейной модели.
- [x] Один шаг gradient descent вручную.
- [x] Chain rule для последовательного вычислительного графа.
- [ ] Autograd и накопление градиентов в PyTorch.
- [ ] Линейная регрессия на PyTorch без `nn.Linear`.
- [ ] `nn.Module`, `Dataset`, `DataLoader` и training loop.
- [ ] Активации, инициализация и нормализация.
- [ ] MLP и CNN.
- [ ] Attention и Transformer.

### Classical ML

- [x] Интуиция overfitting и bias–variance trade-off.
- [x] Назначение train/validation/test.
- [ ] Cross-validation и корректный model selection.
- [ ] Метрики классификации и выбор порога.
- [ ] Data leakage и корректные pipelines.
- [ ] Калибровка вероятностей и анализ ошибок.

### LLM Engineering

- [x] Назначение embeddings, retrieval, reranking и generation.
- [x] Базовые факторы выбора chunk size.
- [ ] Retrieval-метрики: Recall@k, MRR, MAP, nDCG.
- [ ] Faithfulness, correctness и citation accuracy.
- [ ] Evaluation dataset и regression testing для RAG.
- [ ] Fine-tuning, LoRA и QLoRA.
- [ ] Serving, latency, batching, caching и стоимость.
- [ ] Agent vs deterministic workflow.

### Python и инженерная подготовка

- [x] Mutable default arguments.
- [x] Базовые различия iterable, iterator и generator.
- [x] Late binding в замыканиях Python.
- [ ] GIL, threading, multiprocessing и asyncio.
- [ ] Типизация, протоколы, context managers и decorators.
- [ ] SQL и алгоритмические задачи для интервью.
- [ ] Docker, API, очереди, мониторинг и CI/CD.

## Будущие проекты портфолио

1. **Deep Learning from first principles** — autograd или ключевые компоненты нейросети с численной проверкой градиентов и тестами.
2. **Production-grade RAG** — retrieval, reranking, evaluation dataset, метрики, tracing и анализ ошибок.
3. **AI Quality Platform** — regression-тестирование LLM-приложений, faithfulness/correctness checks, latency и cost monitoring.

Выбор окончательного направления будет основан на том, какие задачи окажутся наиболее интересными на практике: моделирование, исследования, LLM-продукты или ML-инфраструктура.

## Журнал

### 2026-07-23 — первичная диагностика

- Проверены основы Python, ML, Deep Learning и RAG.
- Обнаружены приоритеты: backpropagation, PyTorch, evaluation и concurrency.

### 2026-07-26 — chain rule

- Вручную вычислены градиенты линейной модели с MSE.
- Разобран составной граф `w, b → z → a → L`.
- Правильно вычислены градиенты `dL/dw = -12` и `dL/db = -6` для задачи из конспекта.
