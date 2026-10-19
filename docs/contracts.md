# Контракты данных Alphich

Контракты фиксируются до параллельной разработки интерфейса, API и агента. Ниже целевой формат; серверные схемы и OpenAPI появятся при реализации.

## Основные сущности

| Сущность | Основные поля |
| --- | --- |
| company | id, source_row, profile_version, group_id, okved_code/name, region, revenue_raw, revenue_unit, products, countries, quality_flags |
| company_feature | id, company_id, name, value, unit, method, source_refs, coverage, version |
| payment_edge | payer_id, beneficiary_id, amount_raw, count, source_rows, is_self, ambiguity_flags |
| person_company_link | person_id, company_id, source_row; тип связи unknown |
| source_document | id, kind, title, url, published_at, publication_precision, fetched_at, content_hash, version, extraction_status |
| document_chunk | id, document_id, parent_id, page/исходные spans, section, text, text_version, topics, forecast_horizon, ocr_checked |
| bank_claim | id, document_chunk_id, kind, attributed_to, metric, geography, scenario, value/unit, observed_period, forecast_horizon, position_role, revision_of, verification_status |
| event | id, version, title, published_at, event_at, topics, entities, fact_ids, source_ids, completeness |
| event_fact | id, event_id, event_version, text, source_chunk_ids, source_available_at, extraction_status, verification_status |
| search_facet | id, company_id, profile_version, role, text, feature_ids, template_version |
| embedding_space | id, model_id/revision, tokenizer/config, dimension, prefix_mode, normalization, preprocessing/chunker_version |
| text_embedding | owner_kind/id/version, text_hash, embedding_space_id, vector, created_at |
| retrieval_run | id, corpus_kind/version, query/facets, filters, embedding_space_id, ranking_version, candidates/ranks, exclusions, selected_evidence_ids |
| insight | id, company_id, role, event_id, as_of, status, claims, risks, opportunities, next_steps, missing_data, run_id, cache_key |
| answer | id, insight_id, question, claims, evidence_ids, missing_data, status, run_id, created_at |
| agent_run / agent_step | id, state, action, tool_arguments, evidence_ids, validation_results, timestamps, model, prompt_version, usage |
| job / llm_usage / feedback | Состояние задачи; резерв и расход бюджета; оценка пользователя или эксперта |

Ссылки на сырой набор и оригиналы файлов хранятся на сервере. Публичный интерфейс получает только разрешенные фрагменты и безопасные URL. Неизвестные единицы и даты передаются как null, а не выводятся из порядка величины числа.

## Карточка события

```json
{
  "id": "insight_example",
  "company_id": "company_example",
  "role": "cfo",
  "event_id": "event_example",
  "as_of": "2026-10-20T00:00:00+03:00",
  "analysis_mode": "current",
  "status": "ready_with_conditions",
  "headline": "Что означает событие для компании",
  "relevance_reasons": [
    {"feature_id": "feature_credit", "reason": "Есть кредитный продукт"}
  ],
  "claims": [
    {
      "id": "claim_1",
      "kind": "company_fact",
      "text": "В профиле указан кредитный продукт",
      "evidence_ids": ["feature_credit"],
      "conditions": []
    },
    {
      "id": "claim_2",
      "kind": "model_inference",
      "text": "При плавающей ставке стоимость обслуживания долга может оставаться высокой",
      "evidence_ids": ["feature_credit", "event_fact_example"],
      "conditions": ["Неизвестны тип ставки и условия договора"]
    }
  ],
  "bank_view": {
    "status": "not_found",
    "claim_ids": [],
    "reason": "Подходящая датированная оценка не найдена"
  },
  "risks": [],
  "opportunities": [],
  "next_steps": [
    {
      "text": "Уточнить тип ставки и дату ее пересмотра",
      "basis_claim_ids": ["claim_1", "claim_2"],
      "conditions": ["Условия договора отсутствуют в профиле"]
    }
  ],
  "missing_data": ["Условия кредитного договора"],
  "verification": {"status": "passed_with_conditions", "issues": []},
  "run_id": "run_example"
}
```

Это пример структуры, а не готовая карточка реального события. `bank_view.status` принимает `found`, `historical_only`, `conflicting`, `not_found`. Если позиция банка найдена, соответствующие `claims` имеют `kind=bank_view`, ссылку на фрагмент с датой и страницей.

`found` означает подходящую оценку в загруженном корпусе, а не подтверждение текущей позиции всего банка. У `bank_claim` значение `position_role=current_in_document` относится к дате документа; `previous_view` и `conditional` не подменяют его. Дата публикации и горизонт сравниваются отдельно.

Все типы тезисов: `news_fact`, `company_fact`, `bank_view`, `model_inference`. В массив evidence входят ID, выданные сервером, а не придуманные моделью ссылки. Проценты концентрации ссылаются на рассчитанный `company_feature`, где сохранен знаменатель и правило исключения self-payments.

`risks` и `opportunities` содержат ID тезисов из `claims`; отдельного неподтвержденного текста в них нет. Каждый `next_step` имеет `text`, `basis_claim_ids` и `conditions`. Проверка проходит по всей карточке, включая заголовок, причины релевантности и действия; тезисы не могут обходить ее через свободные поля.

## Общий формат доказательства

`GET /evidence/{id}` возвращает объект с общими полями `id`, `kind`, `source_version`, `company_id` при применимости, `text`, `source_refs`, `published_at`, `quality_flags`. Дополнительные поля зависят от `kind`:

| kind | Обязательные дополнительные поля |
| --- | --- |
| company_feature | feature_name, value, unit, method, coverage, profile_version; dataset и исходные строки в source_refs |
| event_fact | event_id, event_version, source_chunk_ids; подтверждающий текст публикации доступен через source_refs |
| document_chunk | document_id, page или source_span, url, content_hash, extraction_status; для PDF страница обязательна |

Каждый evidence ID неизменяем и закреплен за конкретной версией данных. Обновление источника создает новый ID; прежняя карточка сохраняет прежние доказательства и дату. ID в JSON примерах условные; production формат не должен ссылаться на изменяемое значение «последняя версия». Наследование доступа проверяется и для всех вложенных source_refs.

Embedding — средство поиска, а не тип доказательства. Смысловое сходство не становится news_fact/company_fact. Для банковского утверждения доказательство ссылается на точный document_chunk/span. Parent expansion добавляет контекст; если используется подтверждение вне исходного child, сервер выдает отдельный evidence ID на нужный span. Ссылки на соседние абзацы не подменяются ссылкой на поисковое совпадение.

## Результат отбора и поиска

Каждый элемент ленты содержит `event_id/version`, `priority`, `relevance_tier`, `reason_codes`, `matched_feature_ids`, `matched_fact_ids`, `semantic_facet_ids`, `coverage`, `limitations`, `retrieval_run_id`. Уровни `direct`, `context`, `hypothesis` описывают основание релевантности; они не являются вероятностями ущерба. Если известен только смысловой матч, не создаем отсутствующий fact ID.

В retrieval run сохраняются lexical/dense/fusion ранги кандидатов, примененные фильтры доступа и времени, ID расширенного parent, выбранные/отклоненные bank claims и причины исключения. Два контура имеют `corpus_kind=news|bank`, а заданный сервером контекст доступа не заменяется аргументами модели.

Запрос возвращает пригодные evidence IDs, статус полноты и режим `hybrid|lexical_fallback`. Режим входит в ключ кэша. Версии embeddings не заменяют версии evidence; изменение кодировщика требует повторного поиска, изменение OCR также требует новых ID текстовых доказательств.

Дата анализа и доступ зафиксированы в задаче сервером и применяются ко всем инструментам, расширениям и обходам revision. Историческая сводка события использует только допустимые на тот момент публикации и факты; `source_available_at` относится к подтверждающей версии источника, не к дате описанного события или времени извлечения LLM.

## Основной API

Все маршруты ниже имеют префикс `/api/v1`. Сервер проверяет разрешенные company ID независимо от параметров клиента.

| Метод и путь | Назначение |
| --- | --- |
| POST /demo-session | Создать явно обозначенную демосессию из разрешенного набора |
| GET /companies | Компании, доступные пользователю, и краткие профили |
| GET /companies/{id}/profile | Профиль, версия, расчетные признаки и ограничения данных |
| GET /companies/{id}/feed?role=…&as_of=… | Отранжированные события, причины релевантности и статусы анализа |
| POST /companies/{id}/insights | Поставить анализ event_id, роли и даты в очередь; вернуть 202 и job_id |
| GET /jobs/{id} | Статус, этап и безопасное описание ошибки |
| GET /insights/{id} | Готовая карточка, источники, дата и статус проверки |
| POST /insights/{id}/questions | Уточнение в том же контексте; 202 и job_id |
| GET /answers/{id} | Ответ на вопрос с claims, источниками, статусом и тем же контекстом insight |
| GET /evidence/{id} | Подтверждающий фрагмент с источником и страницей |
| GET /runs/{id} | Список действий и результатов инструментов для объяснения анализа |
| POST /publications | Новая публикация: текст и метаданные; URL необязателен; 202 и job_id |
| GET /events/{id} | Созданное событие, источники и статус извлечения |
| POST /feedback | Оценка релевантности или сообщение об ошибке |
| GET /health | Готовность приложения и обязательных зависимостей |

Для постановки задачи клиент передает `Idempotency-Key`. Доступ к job, run и evidence наследуется от исходной карточки/компании. Запрос вопроса получает тот же профиль и `as_of`; он не расширяет доступ к чужим данным.

Общий job response: `id`, `status`, `stage`, `result_type`, `result_id`, `error`. До завершения `result_id=null`. При успехе тип `insight`, `answer` или `event` определяет маршрут получения результата. Если публикация объединена с существующим событием, возвращается его ID и новая версия. Ответ диалога использует те же типы claims и обязательную проверку, что карточка.

## Команды агента

```json
{
  "action": "search_bank_materials",
  "arguments": {
    "query": "прогноз ключевой ставки на конец 2026 года",
    "topics": ["rates"],
    "as_of": "2026-10-20",
    "limit": 5
  }
}
```

Runtime добавляет контекст доступа сам. Схема запрещает неизвестные действия, лишние поля и превышение лимитов. Инструмент возвращает `evidence_id`, title, дату, страницу, текст, ограничения извлечения и признак применимости по времени. Команда `finish` передает черновик карточки, который проходит обязательную серверную проверку.

## Статусы и ошибки

Job: `queued → running → succeeded | failed | budget_exceeded`. Insight: `ready`, `ready_with_conditions`, `insufficient_evidence`. Пустая лента, отсутствие банковского материала, неполный текст новости и недоступность модели имеют разные пользовательские состояния.

При превышении лимита не запускаем скрытые повторные запросы. При недоступном полном тексте сохраняем статус `excerpt_only`. При ошибке OCR фрагмент не считается проверенным доказательством для точного числа.
