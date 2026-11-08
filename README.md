# Universal AutoResearch Agent

> **Личный вклад · [Ruzik707](https://github.com/Ruzik707)**
> Портфолио-версия командного проекта Альфа Хакатона. Я развивал отдельную реализацию `ruzik`: воспроизводимую оценку ML-экспериментов, контроль ресурсов, правила решения для целевых метрик и внешние benchmark-инструменты. Разработка велась с использованием AI-помощников.

Из обновлённой версии Ruzik в итоговый командный агент адаптированы **точный подбор порога для binary F1** и **календарные признаки**. Также в карте источников отмечены ранний Ridge и локальная сетка alpha; подходы к группировке дубликатов, резервному ответу и ресурсам имеют несколько командных источников. [Подтверждение переноса](universal_agent/docs/reviews/ruzik_update_85f4988.md) · [Происхождение решений](universal_agent/docs/PROVENANCE.md).

Итоговый `universal_agent` — командная интеграция; я не приписываю себе единоличную разработку всего пакета. **Участники: Ruzik707, Егор Апрельский, Олег Солонинкин и Александр (линия Rakovskiy).** [Вклад участников](CONTRIBUTORS.md).

[Разбор проекта для изучения](docs/portfolio/LEARNING_GUIDE.md) · [Промпт для ChatGPT](docs/portfolio/CHATGPT_STUDY_PROMPT.md) · [Черновик для резюме](docs/portfolio/RESUME_DRAFT.md)

---

Агент для single-target табличной регрессии и классификации: читает task.md и CSV, строит общий план валидации, проверяет гипотезы LLM, выбирает модель и атомарно сохраняет submission.csv рядом с task.md.

Подробные [условия](universal_agent/docs/HACKATHON_REQUIREMENTS.md), [аудит четырёх агентов](universal_agent/docs/AGENT_REVIEW.md), [происхождение решений](universal_agent/docs/PROVENANCE.md), [проверки и ограничения](universal_agent/docs/VALIDATION.md), [сравнение пяти агентов на внешнем контроле](universal_agent/docs/AGENT_COMPARISON.md).

## Установка и запуск

Для распакованного платформенного ZIP (это отдельный формат поставки) entrypoint и requirements лежат в корне архива:

```bash
python -m pip install -r requirements-mvp.txt
python solution.py /path/to/task.md
```

Подробная документация доступна по ссылкам на репозиторий выше и в ZIP не дублируется.

Для работы с этим Git-репозиторием команды выполняются из его корня. Python 3.12. Runtime-зависимости — подмножество закреплённого организаторами `requirements.txt`:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r universal_agent/requirements-mvp.txt
.venv/bin/python universal_agent/solution.py /path/to/task.md --run-dir runs/my-run
```

Без ключа и внешнего endpoint выбирается native Ollama, http://127.0.0.1:11434, qwen3.5:9b. Ollama и модель должны быть установлены и запущены заранее. Приложение не загружает модели и не устанавливает пакеты во время решения.

```bash
export AUTOML_PROVIDER=ollama
export OLLAMA_MODEL=qwen3.5:9b
.venv/bin/python universal_agent/solution.py /path/to/task.md
```

На платформе используются выданные организаторами переменные baseline:

```bash
export AUTOML_PROVIDER=openai
export OPENAI_MODEL='имя выданной модели'
export OPENAI_BASE_URL='адрес выданного endpoint с /v1'
export OPENAI_API_KEY='выданный ключ'
.venv/bin/python universal_agent/solution.py /path/to/task.md
```

Без AUTOML_PROVIDER внешний OPENAI_BASE_URL или непустой OPENAI_API_KEY выбирают OpenAI-compatible режим. Правила кейса запрещают другие модели через собственные API-ключи; наличие Ollama на платформе не предполагается. `.env.example` — пример, `.env` автоматически не загружается.

Короткая проверка без LLM:

```bash
.venv/bin/python universal_agent/solution.py /path/to/task.md --offline \
  --cv-folds 2 --family-trials 1 --max-trials 1 --n-jobs 1 \
  --time-budget 60 --trial-timeout 30 --run-dir runs/smoke
```

## Исследование

- Входы: train.csv, test.csv, необязательные data_dictionary.csv и sample_submission.csv. Target определяется словарём/разницей схем; неоднозначность вызывает ошибку. Поддерживаются относительные ссылки и CSV рядом с task.md. ID исключаются из модели, сохраняются в исходном порядке, включая ведущие нули.
- Метрики: RMSE, MSE, MAE, MAPE, RMSLE, R²; Macro/Weighted/Binary F1, Accuracy, Balanced Accuracy, ROC-AUC, LogLoss. Вероятностный вывод проверяется по контракту задачи/шаблону. Неоднозначная probability schema явно отклоняется.
- По умолчанию 5 OOF folds; подтверждённый future-only временной признак включает forward CV. Идентичные строки признаков в auto-режиме объединяются в одну группу. `--validation-strategy iid` отключает автоматический temporal/group выбор. Все возможные скрытые сущности этим не распознаются.
- При CV>1 стартовый список: LightGBM, Linear/Ridge, ExtraTrees, XGBoost, CatBoost при доступных пакетах; `--family-trials` ограничивает его длину. Ridge остаётся в списке при установленных бустерах; дорогой CatBoost перенесён в конец. Warmup ограничен долей research-окна, каждый trial — меньшим из заданного timeout, 1/6 общего бюджета и оставшегося research-времени минус2с. Inner early stopping видит только train-часть внешнего fold; выбранное число итераций переносится в refit. Кандидаты оцениваются хостом на одном плане.
- LLM меняет валидированные рецепты: параметры, веса, fit options и `preprocessing.feature_ops`. Признаки: drop, to_nan, signed log1p, missing flags, frequency encoding, group statistics, ratio/diff/product, row statistics, category combinations, quantile bins, text statistics, date parts. Частоты, группы, bins и словарь слов обучаются только на fit-строках fold и сохраняются с моделью. Текстовый словарь ограничен первыми 10000 fit-строками и 10000 символами на строку; на упорядоченных данных это может смещать vocabulary.
- Пока выполняется HTTP-запрос LLM, основной поток продолжает ограниченный воспроизводимый recipe search; состояние и журнал меняет только основной поток. Сначала проверяются мягкая балансировка F1 и одна fit-only цепочка freq/missing, затем локальные параметры лучших измеренных семейств. При ошибке провайдера или отсутствии успешных LLM-кандидатов выполняется до четырёх дополнительных trials в оставшемся бюджете. `--search-trials N` явно включает поиск, также с --offline. Все host-гипотезы журналируются как `origin=search` и не считаются работой LLM. Пакет из двух LLM-гипотез допустим и при одном worker: очередь исполняется последовательно. Отдельный cancellation token закрывает ожидающий HTTP при остановке исследования.
- Постоянный fallback (mean/mode/prior) пишется до дорогой диагностики, каждый успешный лидер обновляет файл. При timeout final refit остаётся валидный прогноз. `completed` означает успешный LLM candidate, а не улучшение скрытой метрики; `degraded`, `offline`, `deadline` отражают режим.
- Малые смеси, Macro F1 multipliers и точный бинарный F1 threshold оцениваются через cross-fitted/forward meta assessment. Threshold использует вероятность явно объявленного положительного класса (или прежнее правило 1/последний сортированный класс), включает ties через `>=` и меняет только метки. При отсутствии независимого выигрыша правило отклоняется; ROC-AUC/LogLoss не получают порог. Модели уже выбраны по OOF: это не nested CV и не независимый тест.
- `--experiment-mode code` включает свободные модельные гипотезы с AST-проверкой и train-only preflight. Это дополнительный режим: проверка AST не является системной изоляцией. Границы файлов и сети обеспечивает контейнер.

Операция `{"op":"date_parts","cols":["observed_date"]}` добавляет UTC epoch-days, месяц и день недели, сохраняя исходную колонку. На fit требуется >95% корректных непустых ISO8601-дат (`YYYY-MM-DD` или строка `YYYYMMDD`, необязательное время). Невалидные значения на transform дают NaN; числа и неоднозначные/частичные даты не превращаются в догаданные даты. Операция доступна LLM в каталоге; её выигрыш проверяется обычным CV, автоматически ко всем строковым колонкам она не применяется. Источники и решения по остальным изменениям донора: [повторный аудит Ruzik](universal_agent/docs/reviews/ruzik_update_85f4988.md).

## Бюджет и артефакты

| Параметр | По умолчанию |
|---|---:|
| --time-budget | 10800 секунд, максимум 10800 |
| --trial-timeout / --llm-timeout | 1800 / 180 секунд |
| --cv-folds / --family-trials | 5 / 4 |
| --parallel-trials / --n-jobs | 2 / 4; суммарно <=8 |
| --memory-limit-mb | 12000 на worker; сумма <=28000 |
| --max-trials / --max-steps | без фиксированного предела, в бюджете времени |
| --max-tokens / --response-tokens | 2000000 / 6144 |

Workers ограничены временем, RSS дерева и объёмом вывода; при остановке завершается process group. На Linux/macOS CLI timer покрывает подготовку и диагностику. Финальный admission учитывает measured fit с safety factor, отдельные8с на запуск worker и write overhead от5с; для принятого ансамбля пересчитывается стоимость его активных участников. При нехватке полного окна сохраняется checkpoint. Это консервативная оценка, а фактическую остановку обеспечивает supervisor/CLI timer. RSS polling дополняет лимит памяти контейнера. Локальная служба Ollama отдельно расходует CPU/RAM.

Run-dir содержит task/config/profile, CV plan/fingerprints, events.jsonl, agent.log, decisions.jsonl, recipe/code trials, OOF predictions, измеренные метрики/ошибки/параметры/время, postprocessing, final model, report.json/report.md. Секреты не передаются worker и редактируются в журнале. Данные и результаты исключены из Git/ZIP.

```bash
tail -F runs/my-run/agent.log
.venv/bin/python universal_agent/scripts/verify_run.py runs/my-run
.venv/bin/python universal_agent/solution.py /path/to/task.md --run-dir runs/my-run --resume
```

`--resume --finalize-only` использует сохранённые trials без новых LLM запросов. Должны совпадать входы и схема валидации; используйте ту же версию кода. Общий накопленный бюджет продолжает учитываться. Совместимость обновлённого кода с прежним resume не гарантируется.

## Тесты и поставка

```bash
uv pip install --python .venv/bin/python -r universal_agent/requirements-dev.txt
PYTHONPATH=universal_agent .venv/bin/python -m pytest -q universal_agent/tests
.venv/bin/python universal_agent/scripts/package_solution.py
docker build -t universal-agent ./universal_agent
```

ZIP: universal_agent/dist/solution.zip; корень содержит solution.py и полный runtime-пакет. Сборка детерминирована, использует allowlist, запрещает symlinks. Тест извлекает реальный ZIP и запускает его из постороннего каталога. Docker: Python 3.12 и libgomp1; фактический статус проверки указан в VALIDATION.md.

Single-target CSV не включает ranking, изображения, multi-output и неизвестные форматы. Без скрытых ответов public/private score не вычисляется.
