# Alphich backend — company foundation + news ingestion

This is a **working backend foundation**, not the entire news-analysis product.
Current endpoints implement database health, bounded demo sessions, accessible company list,
profile/evidence-ready feature extraction, and real queued news ingestion with a separate worker.
The separate [LLM runtime](../docs/llm.md) implements extraction, scoped RAG, an agent,
verified insights, questions and budget accounting. Organizer gateway access is verified and a real
personalized card with bank evidence has been produced. Automated tests use scripted responses;
explicit live evaluation uses the real gateway. RSS collection remains a separate task.
See [NEWS_INGESTION.md](NEWS_INGESTION.md) for the raw ingestion routes and protocol.

## Run with Docker Compose

From the repository root:

```sh
mkdir -p data/raw
# Download df_Uid_data_hack.csv from the organizers' folder into data/raw/.
# Never commit organizers' raw CSV or their LLM API keys to the public repo.
docker compose up -d --build
docker compose run --rm api python -m app.importer /data/raw/df_Uid_data_hack.csv
curl http://127.0.0.1:8000/api/v1/health
```

`DEMO_COMPANY_IDS` defines the **server** allowlist (default `U_9250,U_99802,U_44792`).
These are references to the organizers' anonymized IDs, not actual organizations.
`POSTGRES_PASSWORD` should be set outside local development. The API port binds only to localhost.

To test without the organizers' CSV, first run `DEMO_COMPANY_IDS=DEMO_A,DEMO_B docker compose up -d --build`, then:

```sh
docker compose run --rm -v ./backend/fixtures:/fixtures:ro api python -m app.importer /fixtures/companies.csv
```

## Demo request

```sh
curl -X POST http://127.0.0.1:8000/api/v1/demo-session \
  -H 'Content-Type: application/json' \
  -d '{"company_id":"U_9250","role":"cfo"}'
# Save the returned token in a shell variable, e.g. TOKEN=<token>
curl -H "X-Demo-Session: $TOKEN" http://127.0.0.1:8000/api/v1/companies
curl -H "X-Demo-Session: $TOKEN" http://127.0.0.1:8000/api/v1/companies/U_9250/profile
```

`/docs` shows OpenAPI; `/api/v1/health` reports DB connectivity. Sessions last two hours.
Switch companies by starting a NEW allowed session. No arbitrary company reads are permitted.
This is a **demo gate**, not production authentication; do not expose it publicly with
real/sensitive profiles. The CSV contains anonymized data. Raw rows never appear in the API.

## Data-quality semantics

- IDs and OKVED strings stay strings; missing input remains `null`.
- Boolean `1.0`/`0.0` is parsed; ambiguous values retain a quality flag.
- The importer preserves each source row in `company_raw_rows` and quarantines the complete
  conflicting-ID group (`company_quarantine`); identical repeated profiles preserve row provenance.
- Import is all-or-nothing. A second import requires `--replace`, invalidating old demo sessions.
- `revenue_raw` has `revenue_unit: null`; no assumed ruble units or snapshot date.
- FTS flags/country conflicts are reported; a country alone does not reveal trade direction.
- Each feature evidence ID is deterministically bound to the immutable profile version and source rows.
- Do **not** add payment-graph signals before excluding self-payments and documenting coverage.

## Test locally

```sh
cd backend
python -m pip install -r requirements.txt
pytest -q
```

For local SQLite development only, use `DATABASE_URL=sqlite:///./alphich.db`.
The Compose deployment uses PostgreSQL. P0 uses SQLAlchemy `create_all`; use Alembic
migrations before shipping a schema with ongoing data evolution. Full transitive lock
and automated CI are outstanding deliverables.

## News ingestion and worker

The Compose stack now runs a separate `worker` service. It polls persistent `jobs`,
creates a raw `event` from each submitted publication, and uses bounded retries.

Follow **[NEWS_INGESTION.md](NEWS_INGESTION.md)** for complete curl examples and
security limitations. `POST /publications` returns **202**; poll `GET /jobs/{id}`
and then fetch `GET /events/{id}`. `GET /sources/{id}` returns the original pasted
text to the same company. Raw events have `facts=[]` until an explicit LLM analysis task completes extraction.

## LLM анализ

В `app/intelligence/` добавлены отдельный worker, API анализа и вопросов, проверки и общий бюджет. Настройки шлюза, список маршрутов, ограничения и проверка запуска — в [docs/llm.md](../docs/llm.md). Генерация требует реальных model ID, ключа и настроенного учёта бюджета: подтверждённых тарифов либо квитанций шлюза с явно заданной единицей стоимости. API загрузки публикаций сохраняет прежний контракт.
