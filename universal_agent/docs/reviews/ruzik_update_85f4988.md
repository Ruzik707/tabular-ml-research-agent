# Повторный аудит обновлённого Ruzik

Проверено 23.10.2026. Источник: ветка `ruzik`, зафиксированная на [85f4988ba11c70ebe2242c8722b32f2d4bea5c7a](https://github.com/aprelskiyegor/alfa_hackathon/tree/85f4988ba11c70ebe2242c8722b32f2d4bea5c7a). Сравнение с опубликованным universal_agent на `c1673e2`, а также с ранее проверенным локальным snapshot Ruzik. Исходный [отчёт](ruzik.md) относится к старому snapshot; его замечания нельзя автоматически приписывать новой ветке.

Просмотрены новые production-модули, тесты, изменения `639aa2d`/`85f4988`, план generalization и результаты V5. Два независимых прохода отдельно проверили validation/runtime и features/models/decisions. Donor `test_generalization.py` + `test_research.py`: **16 passed**. Ниже решения по переносу; доказательства интеграции — в [VALIDATION.md](../VALIDATION.md).

## Что перенесено

| Идея и источник | Пробел в universal_agent | Адаптация |
|---|---|---|
| Точный binary F1 cutoff: [decisions.py:19–35](https://github.com/aprelskiyegor/alfa_hackathon/blob/85f4988ba11c70ebe2242c8722b32f2d4bea5c7a/ruzik_agent/decisions.py#L19) | Binary F1 учитывал positive class при оценке, но postprocessing выбирал метки через argmax | Сортировка вероятности positive class, кумулятивные TP, проверка концов групп одинаковых значений: O(n log n). Каждый cutoff учится без оцениваемого OOF fold; temporal использует только прошлые folds. Принятие только по assessed score, не по calibration score на всех OOF |
| UTC-дата → epoch-days/month/weekday: [models.py:98–116](https://github.com/aprelskiyegor/alfa_hackathon/blob/85f4988ba11c70ebe2242c8722b32f2d4bea5c7a/ruzik_agent/models.py#L98) | В feature DSL отсутствовали календарные признаки | Явная `date_parts` вместо эвристики имени колонки. Fit требует >95% корректных непустых ISO-дат; исходная колонка сохраняется. UTC, стабильная схема, NaN для невалидного transform, без повторного fit по test |

Порог — отдельное правило меток: `probability[:, positive_index] >= binary_threshold`. Он не изменяет `predict_proba`, не применяется к ROC AUC/LogLoss, не совмещается с class multipliers. Positive class передаётся в host artifacts, final manifest, сериализованную модель и краткий журнал. Совместимость default argmax для старых manifest сохранена; одна OOF fold не позволяет принять новое правило. Если в meta-train нет positive labels, threshold не подбирается из отложенных строк.

При адаптации отдельно исправлены пограничные случаи: строковая/числовая идентичность меток, positive class на первом столбце, пороги 0/1 и ties; для дат — переполнение на минимальном pandas timestamp, отрицательные epoch-дни и числовые значения, которые pandas мог бы принять за наносекунды 1970 года. ISO задаётся явно: полная дата `YYYY-MM-DD` или строка `YYYYMMDD`, необязательное время; частичные и неоднозначные даты не дополняются догадками. Датасеты, донорский entrypoint и зависимости не копировались.

## Что стало лучше у донора и почему не переносится целиком

| Изменение | Плюсы | Ограничение и решение |
|---|---|---|
| Past-only `target_encode` / `target_lag`, `features.py:39–104`; keys из `search.py:16–41` | История только из train, `allow_exact_matches=False` исключает текущий период. Ключи теперь выбираются без y; старое замечание о target-guided key selection к этому коду неприменимо | Не является доказанно плохой идеей. Отложено как отдельная гипотеза для подтверждённой временной регрессии: нужны вложенный отбор, равные периоды/duplicates/unseen keys и проверка переноса на внешний holdout. В текущий unsupervised fit-only DSL не добавляется скрытый доступ к y |
| `log1p`/`sqrt`/gamma и residual offset, `models.py` | Расширяют набор regression objectives и форму зависимости | В donor inverse `expm1` может вернуть отрицательный прогноз при неотрицательной цели; остаточная модель зависит от корректности history. Не копируются без отдельного контракта и оценки конкретной метрики |
| Ранний Linear на небольших невременных таблицах, `search.py:strong_candidates` | В V5 помогает линейным synthetic задачам; Linear уже доступен universal через recipes | В четырёх trials отнял слот balanced LightGBM и ухудшил Titanic. Не вводится безусловная замена стартовых семейств по одному короткому screening |
| Metric-aware class multipliers, `decisions.py:36–46` | Macro/Weighted F1 и Accuracy оптимизируются своими функциями; probability metrics не меняются | Macro F1 уже покрыт universal. Точечный выигрыш inner calibration ещё не доказывает outer gain. Balanced Accuracy не входит в donor `LABEL_METRICS`; общий fallback helper считает для неё weighted F1, поэтому не переносится как BA optimizer |
| Inner split / group / temporal validation, `validation.py` | Train-only calibration/stopping, группировка дубликатов, будущие периоды исключены | IID folds — повторные holdout-ы, строки могут пересекаться. Universal уже использует disjoint OOF и более строгую группировку точных duplicates; замены нет |
| Runtime/ensemble, `runtime.py`, `worker.py` | Supervisor, checkpoint, budget estimates, итоговый refit | Подготовка/profile/hash не охвачены общим deadline, fallback появляется после dummy fit; ансамбль regression-only с отбором по первому fold. Universal уже покрывает эти задачи более полно |
| Explicit positive class / pooled и fold metrics | Новый donor исправляет прежний sorted-only positive class; сообщает разные агрегаты и число repeated records | Universal уже хранит explicit positive class и считает pooled disjoint OOF по единому контракту; repeated-holdout selection protocol не переносится |
| External benchmark, `scripts/benchmark_suite.py`, `compare_benchmarks.py` | Внешние labels не передаются агенту, оценивается финальный CSV, фиксируются hashes/splits/budgets и несколько seeds; источник данных отделён от inference | Полезная методология для следующей оценки, не runtime-функция. Comparator не сравнивает `reference_family`, значит допускает несопоставимые reference baseline. Harness не копировался целиком и чужие числа не выдаются за качество universal |

V5: 14 наборов × 2 seeds, короткие 35 секунд/4 trials; автор честно сообщает ухудшения и отсутствие locked/private проверки. Три метрики одной linear synthetic структуры не являются тремя независимыми подтверждениями. Перенесённые операции расширяют возможности универсального агента; гарантированный выигрыш на скрытых данных этим не установлен.

## Карта реализации и тестов

- Binary F1: `automl_agent/postprocessing.py` → `ensemble.py` → `agent.py` → `final_ensemble.py` / `prediction.py`; отображение в `audit.py`. Проверки: `tests/test_binary_threshold.py`, `test_final_ensemble.py`, `test_research_audit.py` и прежние probability/OOF tests.
- Date parts: `automl_agent/features.py`; каталог автоматически попадает в `recipes.py`, transformer включается в самостоятельный source кандидата. Проверки: `tests/test_date_features.py`, включая pipeline, cloudpickle/joblib reload, одинаковый результат при изменении transform batch и лимиты ширины.
- Полный host-путь принятого cutoff проверен с настоящим supervised final worker и повторным чтением модели/CSV. Новые runtime-зависимости не добавлены; ZIP allowlist прежний.
