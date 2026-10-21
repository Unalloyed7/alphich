# Cloud.ru: публичная конфигурация Alphich для команды

Это документация **к репозиторию**, а не гарантия текущего состояния VPS. Конфигурация не содержит ключей, паролей и частных данных. GitHub-репозиторий публичный: не добавляйте в него содержимое `.env`, VPN-подписку, SSH-ключи, дампы БД, индексы или внутренние данные компании.

## Приоритет настроек при автоматическом деплое

По порядку применяются:

1. `compose.yaml` — базовые шесть сервисов и подстановка приватного `.env`.
2. `/home/alphich/.config/alphich/compose.vpn.yml` — закрытый Docker override для Xray.
3. `compose.cloudru-ai.yaml` — публичный override моделей, их лимитов и способа учёта стоимости.

**Последний файл переопределяет совпадающие переменные среды.** Например, запись Qwen `output_limit=4096` только в `.env` **не действует**, когда активный Compose использует публичный `compose.cloudru-ai.yaml` со значением 2500. Поэтому публичный каталог синхронизируется в GitHub. Пароли и `LLM_API_KEY` остаются исключительно в `.env` на сервере.

## Модели, бюджет и автоматизация

| Параметр | В публичном профиле / проверенное значение | Комментарий |
| --- | --- | --- |
| `LLM_BASE_URL` | `https://changellenge.up.railway.app/v1` | Публичный адрес организаторов; не токен |
| `LLM_EXTRACTION_MODEL` | `z-ai/glm-5.3-flash` | Выбрана для извлечения в текущем GitHub-профиле после ошибки HTTP 503 у Qwen |
| `LLM_ANALYSIS_MODEL` | `z-ai/glm-5.3-flash` | Выбрана для анализа/проверки |
| Каталог Qwen | `qwen/qwen3.6-35b-a3b`, `input_limit=32768`, `output_limit=4096` | **Разрешена в каталоге, но сейчас не выбрана для извлечения** |
| Каталог GLM | `z-ai/glm-5.3-flash`, `input_limit=32768`, `output_limit=2500` | Активная модель |
| Формат ответа | `response_mode=json_schema`, `reasoning_enabled=false` | Обе модели; сервер отдельно проверяет JSON |
| `LLM_BUDGET_MODE` | `gateway_receipt` | Учёт по квитанции шлюза |
| `LLM_RECEIPT_RUB_PER_UNIT` | `1` | **Допущение для этого конкретного шлюза**, не официальный валютный курс |
| `AI_AUTOMATION_ENABLED` | `0` | Массовые платные AI-задания отключены |
| `LLM_RECONCILE_ZERO_HTTP_RECEIPTS` | `1` | Только подтверждённые нулевые квитанции; без автоматического повтора неизвестных исходов |

Проверенное на стенде значение `LLM_DAILY_LIMIT_RUB=10` хранится в серверном `.env` и намеренно **не фиксируется публичным override**. Лимит работает в рамках допущения о единицах квитанции; он не гарантирует максимальный счёт провайдера. Общекомандный бюджет организаторов 500 единиц — отдельная величина. Не увеличивайте лимит и не включайте автоматическую генерацию без согласования.

Стандартные параметры из `.env.example` (если оператор их не переопределил): `LLM_TIMEOUT_SECONDS=45`, `LLM_MAX_CALLS=8`, `LLM_MAX_TOOLS=8`, `AI_AUTOMATION_ROLES=cfo`, `AI_AUTOMATION_MAX_EXTRACTIONS=20`, `AI_AUTOMATION_MAX_CARDS=10`, `AI_AUTOMATION_SCAN_LIMIT=1000`, `NEWS_FEED_MAX_EVENTS=2000`.

## Сервисы, сеть, данные

- Compose: `postgres`, `api`, `worker`, `analysis-worker`, `frontend`, `rss-collector` (последний с `--profile news`). PostgreSQL сохраняется в Docker volume `pgdata`.
- Xray запущен отдельно, в сети Docker `alphich-ai-proxy`. В приватном override доступ через `HTTPS_PROXY=http://alphich-xray:10809` есть у `api` и `analysis-worker`. Системная маршрутизация Cloud.ru не меняется, наружу не публикуется порт прокси. **Параметры VLESS/REALITY и URL подписки не коммитятся**.
- HTTPS и Basic Auth сайта обслуживает Caddy. Его пароль не является частью Git.
- Локальный E5 ONNX INT8 и индекс RAG, PDF/CSV, исходные документы, данные компаний, секреты и PostgreSQL backups хранятся вне репозитория. Не запускайте `docker compose down -v`.
- RSS может обновляться независимо от платной генерации; карточки запускаются вручную, пока `AI_AUTOMATION_ENABLED=0`.

## Как обновляется сайт

Коллега делает ветку → Pull Request → успешный CI → merge в `main` → успешный push CI → автоматический `Deploy Cloud.ru (manual or after CI)`. Workflow проверяет SHA, делает приватный `pg_dump`, обновляет код и пересобирает сервисы с VPN-override. **Для изменений кода SSH коллеге не нужен**. Переменные `.env`, файлы VPN, доступы, миграции схемы и восстановление требуют отдельного администрирования. Автоматического отката БД нет. Защитите `main` обязательными PR и CI.

## Проверка реально применённых значений на Cloud.ru (бесплатная)

На VPS из `~/alphich`:

```bash
sudo docker compose \
  -f compose.yaml \
  -f ~/.config/alphich/compose.vpn.yml \
  -f compose.cloudru-ai.yaml \
  --profile news config --quiet

sudo docker compose \
  -f compose.yaml \
  -f ~/.config/alphich/compose.vpn.yml \
  -f compose.cloudru-ai.yaml \
  exec -T analysis-worker python - <<'PYTHON_CHECK'
from app.intelligence.settings import Settings
from app.intelligence.automation import AutomationConfig
s = Settings.from_env()
print('EXTRACTION_MODEL:', s.extraction_model)
print('ANALYSIS_MODEL:', s.analysis_model)
print('QWEN_OUTPUT_LIMIT:', s.models['qwen/qwen3.6-35b-a3b'].output_limit)
print('GLM_OUTPUT_LIMIT:', s.models['z-ai/glm-5.3-flash'].output_limit)
print('BUDGET_MODE:', s.budget_mode)
print('DAILY_LIMIT_RUB:', s.daily_limit_micros / 1_000_000)
print('AI_AUTOMATION:', AutomationConfig.from_env().enabled)
PYTHON_CHECK
```

Эти команды не печатают секреты и не выполняют платных генераций. **Не публикуйте полный вывод `docker compose config`**: в нём могут оказаться секреты. Если Qwen понадобится для извлечения снова, подготовьте отдельный PR с проверкой надёжности HTTP 503, завершения ответов и расходов: повышение его лимита в каталоге само по себе не переключает модель.

См. [LLM и бюджет](llm.md), [деплой](cloudru-deploy.md), [резервные копии](backup-and-deployment.md).
