# Автоматическая разметка: запуск и возобновление

Из корня проекта, с зависимостями backend и локального E5. Используйте Python 3.12/3.13 и закреплённые зависимости. Ключи и DATABASE_URL не передаются в аргументах и не публикуются.

В текущем локальном checkout подготовлено окружение `tmp/audit-venv` с Python 3.12.15 и версиями numpy/onnxruntime/tokenizers из проекта. Для него команды ниже запускаются как `tmp/audit-venv/bin/python ...`; системный Python и прежняя `.venv` не заменялись.

```bash
export PYTHONPATH=backend:backend/src:knowledge/src
```

## Подготовка без платных вызовов

Исходный пилот должен содержать неизменённые manifest/train/validation/test. Новый train создаётся отдельно:

```bash
python -m alphich_kb.relevance_ml build-matched-train \
  --pilot-plan data/private/relevance-ml/api-pilot-v1/manifest.json \
  --companies-csv data/raw/df_Uid_data_hack.csv \
  --exclude-pair-ids /private/attempted-pairs.json \
  --output-dir data/private/relevance-ml/matched-full-cohort-v1

python -m alphich_kb.relevance_ml mine-candidates \
  --input data/private/relevance-ml/matched-full-cohort-v1/train.jsonl \
  --pilot-plan data/private/relevance-ml/matched-full-cohort-v1/manifest.json \
  --seen-batch-dir /private/previous-batch \
  --exclude-pair-ids /private/attempted-pairs.json \
  --env-file .env --prompt-variant business-only \
  --limit 40 --output-dir data/private/relevance-ml/full-cohort-queue-v8
```

`--seen-batch-dir` повторяется для разных прошлых партий. Он исключает успешные и ошибочные попытки, включая исторические версии промптов. Отсутствие локального файла не доказывает, что вызова не было: перед запуском сверяйте общий журнал. `--exclude-pair-ids` — JSON-массив известных ID попыток, не набор меток.

Второй/следующий запуск подготовки требует нового каталога. Нет необходимости переписывать frozen train для изменения порядка: очередь содержит исходные offsets и хеши.

## GLM: проверка и платный запуск

Платный запуск выполняется в серверном окружении существующего `analysis-worker`, с общей PostgreSQL. Пример ниже предполагает уже загруженное окружение контейнера; `--env-file .env` для него не нужен.

```bash
python -m alphich_kb.relevance_ml judge \
  --input data/private/relevance-ml/matched-full-cohort-v1/train.jsonl \
  --pilot-plan data/private/relevance-ml/matched-full-cohort-v1/manifest.json \
  --offsets-file data/private/relevance-ml/full-cohort-queue-v8/queue.json \
  --output data/private/relevance-ml/full-glm-v8-40 \
  --batch-id full-cohort-v8-glm-40 --model-id z-ai/glm-5.3-flash \
  --prompt-variant business-only --limit 40 --dry-run
```

После бесплатной проверки та же команда запускается с `--live --shared-ledger --continue-settled-validation-errors --max-cost-rub 8 --max-seconds 1200` вместо `--dry-run`. Цена остановки 8 рублей проверяется между вызовами: следующий ответ может превысить её. Общий дневной предел сохраняется в Provider и PostgreSQL.

## Qwen: слепая проверка тех же входов

Та же команда, очередь и диапазон; заменить модель на `qwen/qwen3.6-35b-a3b`, batch ID на `full-cohort-v8-qwen-40`, каталог на `full-qwen-v8-40`. Два запуска выполняются последовательно: существующий receipt-учёт резервирует доступный бюджет, параллельные процессы могут блокировать друг друга. Первая метка не попадает в запрос второй модели.

Незавершённый ответ, HTTP-ошибка и неизвестный исход не создают отрицательный label. Возобновление использует **тот же** каталог, ID партии, модель, вариант промпта, очередь и диапазон. Новое имя batch не является способом повторить ошибочную попытку. При `budget_exceeded` без попытки исходный вызов остаётся доступным для последующего запуска.

## Объединение и QA без ручного этапа

```bash
python -m alphich_kb.relevance_ml export-labels \
  --batch-dir data/private/relevance-ml/full-glm-v8-40 \
  --output data/private/relevance-ml/full-glm-v8-labels.jsonl

python -m alphich_kb.relevance_ml export-labels \
  --batch-dir data/private/relevance-ml/full-qwen-v8-40 \
  --output data/private/relevance-ml/full-qwen-v8-labels.jsonl

python -m alphich_kb.relevance_ml machine-consensus \
  --primary data/private/relevance-ml/full-glm-v8-labels.jsonl \
  --verifier data/private/relevance-ml/full-qwen-v8-labels.jsonl \
  --policy binary --output-dir data/private/relevance-ml/full-silver-v8
```

`silver.jsonl` — пригодные для бинарного weak supervision примеры. `abstained.json` — разногласия/отсутствующие ответы. `report.json` — покрытие и классы. Точный балл остаётся предложением основного учителя; сохраняются оба ответа и интервал. Для строгого согласия точного балла и всех ограничений используйте `--policy exact`.

Temporal eligibility у business-only контракта всегда `unknown`: эту проверку не проводили. Такие метки не разрешают публикацию старой новости. Истинность механизма не доказывается совпадением двух моделей.

## Обучение CPU

Только когда в silver есть достаточно разных положительных и отрицательных сюжетов:

```bash
python -m alphich_kb.relevance_ml train \
  --input /private/approved-case-v9-silver.jsonl \
  --allow-teacher --output-dir data/private/relevance-ml/model-silver-v9
```

Это разбивает новый silver по сюжетам и оценивает согласие с машинными метками. Старый validation/test пилота сюда не добавляется. Не передавайте исходный `--pilot-plan` как план другого train. Не используйте `--min-recall`: этот режим требует отдельного независимого human validation. При единственном классе или пустом разделе обучение должно остановиться. `model.json` остаётся `shadow_only`, вероятности не откалиброваны, а точность на реальной ленте не измерена.

В примере `/private/approved-case-v9-silver.jsonl` заменяется путём из поля `silver` итогового автоматического отчёта. Диагностические v6/v7/v8 метки не смешиваются с case-calibrated v9: старые критерии оказались слишком консервативными. QA и модель отклонят такое смешение.

## Текущий VPS

Изолированный рабочий каталог этого запуска: `/home/alphich/alphich-auto-labeling-20261024-v6`. Суффикс каталога исторический; версии промптов определяются `batch.json`, а версии исходников — сохранёнными source provenance файлами. Данные лежат в его `data/private/relevance-ml`. Production-каталог `/home/alphich/alphich` используется только как источник Compose-окружения и общего журнала; код сайта этим запуском не обновлялся.

Команда контейнера использует `docker compose run --rm --no-deps`, код смонтирован в `/work:ro`, приватные выходы — отдельным writable mount. Для получения статистики достаточно читать `audit/summary-*.json`; полный `.env` и `docker compose config` печатать не нужно.

## Новые первоисточники и критерии самого кейса

`build-matched-train --articles-json /private/articles.json` принимает JSON-массив полных статей с `article_id`, `source_version`, `cluster_id`, `title`, `text`, `url`, `published_at`. Даты должны быть не позже `as_of` исходного плана. Статьи, совпавшие с frozen holdout, исключаются с причиной в новом manifest. Исходный пилот не меняется. В этом запуске отдельно собраны 14 релизов официального RSS ЦБ, включая решение от 23 октября; это новые источники, а не синтетические новости.

`--prompt-variant case-calibrated` (v9) использует общий с runtime текст `personalization/criteria.py`. Он уточняет различие **значимости для управленческого решения** и **доказанного изменения денежных платежей**. Неизвестная сумма долга не отменяет полезность решения ЦБ для компании с подтверждённым кредитным продуктом. При этом условия договора, фактический эффект, география операций и специальная применимость норм не выдумываются. Старые метки не повышаются автоматически и сохраняют прежние версии.

## Полностью автоматическая партия

```bash
python -m alphich_kb.relevance_ml automatic-job \
  --input data/private/relevance-ml/fresh-cbr-train-v8/train.jsonl \
  --pilot-plan data/private/relevance-ml/fresh-cbr-train-v8/manifest.json \
  --offsets-file data/private/relevance-ml/fresh-cbr-queue-v8/queue.json \
  --output-dir data/private/relevance-ml/case-job-v9 \
  --batch-prefix case-v9-overnight --limit 24 \
  --prompt-variant case-calibrated --max-cost-rub 5 --max-seconds 900 \
  --train-shadow
```

По умолчанию это бесплатный preflight обеих моделей. Для оплачиваемого выполнения добавьте `--live --shared-ledger`. Последовательно выполняются две партии с постоянными ID, экспорт и автоматический бинарный QA. Любая неизвестная стоимость, резерв или остановка первой модели предотвращает запуск второй. Повтор той же команды сохраняет оплаченные identity; производные экспорт/отчёты создаются в новых каталогах. При `--train-shadow` обучение запускается только при наличии хотя бы трёх разных сюжетов каждого класса; иначе отчёт честно сохраняет `not_trained`. Это минимальная техническая предпосылка для разделения по сюжетам, не доказательство достаточного объёма данных.

Для долгих ответов Qwen в отдельном labeling-контейнере можно задать `LLM_TIMEOUT_SECONDS=120`; это не меняет timeout production. Повтор неизвестного исхода не является способом обойти остановку.

## Продолжение в текущем бюджетном дне

По просьбе пользователя разметка возобновлена 24 октября, таймер на 03:05 МСК отключён. Для отдельного labeling-контейнера включён `LLM_GATEWAY_BUDGET_GUARD=1`. Обычный production Provider не меняет свою политику.

Перед каждым новым запросом адаптер проверяет `/key/info`: действующий ключ, конечные неотрицательные значения, дневной лимит не выше 500 ₽ по явно настроенному `LLM_RECEIPT_RUB_PER_UNIT`, окно `1d` с окончанием в ближайшую полночь UTC. Ошибка проверки запрещает платный запрос. Общий PostgreSQL сохраняется; другой ключ или отдельный журнал расходов не создаются.

Только явно указанные через `LLM_GATEWAY_RELEASE_CALL_IDS` старые timeout/network-вызовы могут передать удерживаемый резерв контролю лимита на самом шлюзе. Их `cost_micros` остаётся `NULL`, статус остаётся `failed`, повторный запуск того же вызова запрещён. Активные вызовы освобождать нельзя. Основание освобождения и безопасный снимок бюджета сохраняются в `llm_gateway_reserve_releases` и `llm_gateway_budget_observations`. Агрегат шлюза не называется квитанцией конкретного запроса.

Для автономного продолжения включён дополнительный операторский флаг `LLM_GATEWAY_RELEASE_FAILED_TEACHERS=1`: та же передача разрешается для неудачных timeout/network-запросов текущего дня только с purpose `relevance_teacher_v2` и моделями из разрешённого каталога. Вызовы анализа сайта и активные запросы не затрагиваются. Пара с неизвестным исходом остаётся без метки и не повторяется; следующая отдельная очередь может продолжить работу под проверяемым потолком шлюза. Без этого явного флага автоматического выбора таких резервов нет.

Общий учёт расходов консервативно повышается до большего из текущего значения и расхода шлюза; он никогда не уменьшается. Новые запросы по-прежнему резервируют свободную часть общего лимита и рассчитываются по квитанциям. Если тарифы неизвестны, окончательное ограничение неизвестной стоимости обеспечивает дневной потолок самого шлюза; проверка снимка не является тарифом или точной стоимостью следующего ответа. Новый timeout останавливает свою партию без автоматического повтора.

Текущее выполнение: `alphich-label-case-v9-scheduled.service` (название историческое) и следующая за ним `alphich-label-case-v9-full100.service`. Вторая партия содержит 100 пар, 55 сюжетов, 48 компаний, поровну CEO/CFO. Сервис ждёт окончания первой партии, а не времени суток. На каждой модели действуют межвызовные пределы стоимости и времени; общий дневной предел остаётся 500 ₽.

После большой партии `alphich-label-case-v9-finish.service` возобновляет оставшиеся пары ЦБ под прежними ID. Затем `alphich-label-case-v9-continuation.service` выполняет очереди 100 + 100 + 92 новых пар по дополнительным публикациям. Все очереди проверены обеими моделями на VPS без платных вызовов. Неудачный read-only GET бюджета можно повторить до трёх раз, а незапущенную пару — возобновить с тем же ID; платные POST не повторяются. После отдельных остановок новые непересекающиеся очереди могут продолжить работу. Финальный сбор берёт только валидные, полностью учтённые ответы и согласие двух моделей, сохраняя отклонения отдельно. Обучение общего silver остаётся shadow-only.

Проверка текущих счётчиков без модели и ключей:

```bash
python3 /home/alphich/alphich-auto-labeling-20261024-v6/data/private/hackathon-audit-2026-10-24/labeling-status-v9.py
```

Финальный отчёт и общий silver будут указаны в `data/private/relevance-ml/continuation-run-v9/result.json`; пока сервис работает, проверяются checkpoint-файлы и `progress.json`. Число ответов моделей, уникальных новостей и согласованных пар — разные счётчики.

Фактическое выполнение и состояние расходов: [automatic-labeling-results-2026-10-24.md](automatic-labeling-results-2026-10-24.md).
