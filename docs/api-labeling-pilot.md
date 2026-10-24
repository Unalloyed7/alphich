# API-разметка после аудита, 24 октября 2026

Рабочий путь: **разрешённая GLM через API → предварительные метки → человеческая проверка → CPU-классификатор → теневые предсказания**. Обычные completion-запросы не изменяют веса GLM/Qwen. Дообучение генеративной Qwen и её GPU-среда для этого пути не нужны. CPU-модель учится вероятности класса 4–5; окончательный бизнес-балл 0–5 и доступ к генерации инсайта остаются у действующего GLM-конвейера после обязательной проверки E5.

## Получить реализацию перед запуском команд

Новые команды находятся в ветке **`fix/semantic-gate-api-labeling`**. На `main` до объединения соответствующего PR их нет. Архив `api-labeling-pilot-v1.tar.gz` содержит данные и инструкцию, а исходники нужно получить из этой ветки. Первый переход из существующего чистого checkout:

```bash
git fetch origin
git switch --track origin/fix/semantic-gate-api-labeling
git rev-parse HEAD
export PYTHONPATH=backend:backend/src:knowledge/src
python -m alphich_kb.relevance_ml judge --help
python -m alphich_kb.relevance_ml train --help
python scripts/prepare_api_labeling.py --help
```

В `judge --help` должны быть `--pilot-plan`, `--offset`, `--dry-run`, `--max-seconds`, `--max-cost-rub`, а в `train --help` — `--validation-input`, `--test-input`, `--pilot-plan`. Для запуска понадобятся зависимости backend/knowledge. Фактический SHA сверяется с PR; до бесплатного preflight платные команды не запускаются.

Готовый архив распаковывается **в `data/private/relevance-ml`**, чтобы получился путь `data/private/relevance-ml/api-pilot-v1/manifest.json`. В этом случае повторная подготовка из CSV в разделе 1 не нужна: начинайте с раздела 2. Приватное окружение backend настраивается отдельно от архива; ключи не копируются в Git.

## Что учтено из аудита

| Замечание | Реализация / граница |
| --- | --- |
| Старые 29 787 пар плохо представляют крупные компании | Новый выбор профилей по source-bound выручке строго выше 15 млрд RUB за 2025; заглушки и неясные единицы исключаются. Это критерий выборки, а не доказанный банковский сегмент. Охват «юрлицо или группа» сохраняется неизвестным. |
| Один CFO и давние статьи | У пилота обе роли, явные `as_of` и окно публикаций; по умолчанию 45 дней. Возраст статьи не превращается в бизнес-балл. |
| Нужны неизменяемые входы и независимая проверка | Train/validation/test фиксируются **до разметки**, с SHA-256 файлов и входов. Все пары одного сюжета и роли находятся в одном разделе. |
| API-учителю нужен аудит | Тот же `Provider`, общая PostgreSQL-БД расходов, квитанции, токены, стоимость принятой метки, задержки, ошибки, возобновляемые ID. |
| Ошибки API не должны становиться отрицательными метками | Ошибка схемы/цитат, 503 и неизвестный исход остаются без метки. Неизвестное списание останавливает запуск до сверки; автоматического нового вызова нет. |
| Учитель не является gold | Для независимых validation/test запрещены подсказки модели, нужны два согласованных человека/арбитраж. `train --pilot-plan` проверяет исходные разделы; человеческие файлы не переразбиваются заново. |
| Нужны метрики ранжирования | Precision@5, graded nDCG@5, recall класса 4–5, пропуски пятёрок, отчёт по компании и роли. Общий рейтинг разных компаний явно обозначен диагностическим. |

Аудит описывал более ранний SHA. Модуль `relevance_ml` уже находится в main, а итоговый runtime-балл выставляет GLM. Возврат к `grade_event` из запасного плана аудита не применяется. Документ с предложениями не меняет ранее установленный пользователем общий лимит **500 ₽/сутки**. Его рекомендация «300 ₽ на разметку» не создаёт отдельную защищённую квоту.

## 1. Подготовка без API

Из корня репозитория, с зависимостями backend и knowledge:

```bash
export PYTHONPATH=backend:backend/src:knowledge/src

python scripts/prepare_api_labeling.py \
  --job-dir data/private/relevance-ml/corpus-10k-2025 \
  --companies-csv /путь/к/исходному/df_Uid_data_hack.csv \
  --as-of 2026-10-24T02:19:09+03:00 \
  --output-dir data/private/relevance-ml/api-pilot-v1 \
  --limit 600 --max-companies 24 --max-age-days 45
```

Нужен исходный CSV организаторов; `data/synthetic` содержит маленький искусственный пример для разработки. Вывод содержит агрегаты отбора, версии исходников, `train.jsonl`, `validation.jsonl`, `test.jsonl`, `manifest.json`. Каталог не перезаписывается. `--required-company UID` включает нужный исходный профиль только при соблюдении критерия; несуществующий/неподходящий UID останавливает подготовку.

Фактически подготовленный локальный пилот `data/private/relevance-ml/api-pilot-v1`: **24 профиля, 600 пар, 360/120/120**. Это неразмеченная стратифицированная выборка. Положительные классы пока не подтверждены. Компания может встречаться в разных разделах на разных сюжетах: проверка переноса на новые компании потребует отдельного company holdout. Пересказы с изменённым текстом требуют проверки `cluster_id` человеком.

## 2. Preflight первых 10 запросов

```bash
python -m alphich_kb.relevance_ml judge \
  --input data/private/relevance-ml/api-pilot-v1/train.jsonl \
  --pilot-plan data/private/relevance-ml/api-pilot-v1/manifest.json \
  --output data/private/relevance-ml/api-pilot-v1/preflight-10 \
  --batch-id api-pilot-v1-10 --offset 0 --limit 10 \
  --env-file .env --dry-run
```

`--dry-run` не открывает БД и не делает HTTP-запросов. С каталогом моделей из `.env` проверяется фактический размер всего API-конверта, включая промпт и JSON-схему. Без модели выводит ошибку, без её каталожных настроек явно сообщает, что ограничение размера не проверено. Превышение контекста не означает оценку 0: исправьте состав партии до платного запуска.

## 3. Ограниченный оплачиваемый пилот

Команда выполняется в backend-окружении с **существующей общей PostgreSQL-БД команды** и настроенным разрешённым API. Наличие общей БД нужно для единого учёта расходов сайта и разметки; вычисления учителя выполняет API. Секреты берутся из окружения, не из аргументов команды или Git.

```bash
python -m alphich_kb.relevance_ml judge \
  --input data/private/relevance-ml/api-pilot-v1/train.jsonl \
  --pilot-plan data/private/relevance-ml/api-pilot-v1/manifest.json \
  --output data/private/relevance-ml/api-teacher-pilot-10 \
  --batch-id api-pilot-v1-10 --offset 0 --limit 10 \
  --max-seconds 1200 --live --shared-ledger
```

Перед массовым продолжением посмотрите `audit/summary-*.json`: число принятых меток, все оплаченные неудачи, известная стоимость, незакрытые резервы, p50/p95, пропуски usage, коэффициент квитанции. Цена принятой метки включает неудачные оплаченные попытки. При неизвестном начислении она не рассчитывается как достоверная цена.

Один запуск ограничен 100 парами. Следующая партия использует **другой `--batch-id`, другой `--output` и следующий `--offset`**, например offset 10/limit 50. Повтор той же партии сохраняет ID, offset, limit и каталог: завершённые метки не оплачиваются заново. `--max-cost-rub N` — остановка **между** вызовами по уже известной стоимости партии. Следующая квитанция может превысить N; это не отдельный атомарный подлимит. Общий дневной лимит 500 ₽ контролирует Provider; расходы и резервы других процессов сохраняются.

```bash
python -m alphich_kb.relevance_ml export-labels \
  --batch-dir data/private/relevance-ml/api-teacher-pilot-10 \
  --output data/private/relevance-ml/api-teacher-labels-10.jsonl
```

## 4. Label Studio и человеческие оценки

Используется существующий импортёр. Его историческое имя `relevance-qwen` и дополнительные SFT-файлы сохранены для совместимости; CPU-обучению нужен **`reviewed-pairs.jsonl`**, а не SFT-completions.

Train можно показывать с предложениями API:

```bash
python -m alphich_kb.relevance_ml.label_studio export \
  --input data/private/relevance-ml/api-teacher-labels-10.jsonl \
  --output-dir data/private/relevance-ml/ls-api-train-10 \
  --split train --limit 10 --with-predictions
```

Validation/test экспортируются непосредственно из замороженных файлов, без `--with-predictions`:

```bash
python -m alphich_kb.relevance_ml.label_studio export \
  --input data/private/relevance-ml/api-pilot-v1/validation.jsonl \
  --output-dir data/private/relevance-ml/ls-api-validation \
  --split validation --limit 120

python -m alphich_kb.relevance_ml.label_studio export \
  --input data/private/relevance-ml/api-pilot-v1/test.jsonl \
  --output-dir data/private/relevance-ml/ls-api-test \
  --split test --limit 120
```

В Label Studio импортируйте `tasks.json` и `label-config.xml`; двум людям нужны независимые задания без взаимных ответов и подсказок учителя. Сверяются балл, причинная связь, точные цитаты, факты компании, масштаб, география и временная применимость. Несогласованные ответы идут на арбитраж; импортёр не создаёт из конфликта готовую метку.

После двух независимых экспортов:

```bash
python -m alphich_kb.relevance_ml.label_studio import \
  --bundle-dir data/private/relevance-ml/ls-api-validation \
  --annotations /путь/reviewer1-validation.json /путь/reviewer2-validation.json \
  --output-dir data/private/relevance-ml/human-api-validation
```

Повторите для test. Идентификаторы разметчиков должны быть различимы. Одно подтверждение и assisted review полезны для обучения, но не становятся независимым gold. Отложенный test не открывается для настройки промпта/порогов.

## 5. CPU-обучение на готовой разметке

После достаточного количества проверенных положительных и отрицательных примеров:

```bash
python -m alphich_kb.relevance_ml train \
  --input data/private/relevance-ml/api-teacher-labels-10.jsonl \
  --allow-teacher \
  --validation-input data/private/relevance-ml/human-api-validation/reviewed-pairs.jsonl \
  --test-input data/private/relevance-ml/human-api-test/reviewed-pairs.jsonl \
  --pilot-plan data/private/relevance-ml/api-pilot-v1/manifest.json \
  --output-dir data/private/relevance-ml/model-api-v1
```

Десять меток — проверка процесса, а не обещанный достаточный обучающий набор. Подставьте собранный train, не добавляя в него validation/test. Обучение отклоняет неизвестные метки, единственный класс и несогласованную человеческую проверку. Исходный E5 закреплён; обучаются коэффициенты классификатора и scaler на train. Если часть пилота ещё не размечена, отчёт указывает покрытие каждого раздела.

Порог 0,5 — технический исходный параметр. `--min-recall` выбирает порог только на независимой validation. Итоговый test используется после фиксации решений, не для дальнейшей настройки. Сохраняются число положительных, интервальная оговорка recall, ошибки на grade 5 и происхождение меток. Результат пилота не измеряет качество на всех компаниях и не доказывает, что вероятность откалибрована.

Новый классификатор остаётся в `shadow_only`: он не удаляет новости и не разрешает карточки. Его боевое подключение и любые заявления об экономии требуют независимой оценки. Публикация полнотекстового корпуса, исходных CSV или ключей в Git не нужна.
