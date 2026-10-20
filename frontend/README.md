# Alphich frontend (P0)

React + TypeScript + Vite UI. The UI performs real requests to the backend (no demo/mock event generator):

- `POST /api/v1/demo-session` — server-validated company/role selection;
- `GET /api/v1/companies`, `/companies/{id}/profile` — company data;
- `POST /api/v1/publications` with a unique `Idempotency-Key` — new publication;
- `GET /api/v1/jobs/{id}` — polls worker status;
- `GET /api/v1/companies/{id}/events` — stored events for the authorized company;
- `GET /api/v1/events/{id}`, `/sources/{id}` — event details and raw source.

**This is not an AI news interpretation interface yet.** This UI displays ingestion events and original sources. The server also has separate insight, question, evidence and run endpoints described in [docs/llm.md](../docs/llm.md); the UI still needs to connect those endpoints. Real model credentials are not available yet. No fake AI claims are displayed.

## Quick start via Docker (recommended)

From repository root, the easiest route is **one command**:

```sh
./scripts/start-demo.sh
```

The script starts containers and imports sample profiles only if the DB is empty; it refuses to overwrite another dataset. Alternatively, run the steps manually:

```sh
DEMO_COMPANY_IDS=DEMO_A,DEMO_B docker compose up -d --build
# Import two synthetic profiles. Run once per fresh DB, use --replace to reset the company dataset if needed.
docker compose run --rm -v ./backend/fixtures:/fixtures:ro api python -m app.importer /fixtures/companies.csv
```

Open **http://127.0.0.1:5173**, enter `DEMO_A` and select CFO or CEO. Submit a publication, watch status, inspect the event/source. Check `docker compose logs -f worker` if jobs stay queued. `http://127.0.0.1:8000/docs` opens FastAPI OpenAPI.

If the database already contains synthetic company profiles, there is no need to reimport them. The SQL-based sample profiles are for demos only.

To use organizers' (anonymized) company data instead:

```sh
# Download df_Uid_data_hack.csv into data/raw. Never commit this file.
docker compose down
DEMO_COMPANY_IDS=U_9250,U_99802,U_44792 docker compose up -d --build
# This replaces synthetic company profiles, and expires existing demo sessions.
docker compose run --rm api python -m app.importer /data/raw/df_Uid_data_hack.csv --replace
```

## Local development with hot reload

Run backend via Compose in one terminal:

```sh
DEMO_COMPANY_IDS=DEMO_A,DEMO_B docker compose up -d postgres api worker
```

Then, in another terminal:

```sh
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173. Vite proxies `/api` to the API at `127.0.0.1:8000`, so no CORS or frontend API keys are needed. Docker uses an Nginx reverse proxy with the same paths. `npm run build` validates TypeScript and builds the optimized frontend.

Direct dependencies are pinned. A complete transitive `package-lock.json` should be generated with `npm install --package-lock-only` and committed as soon as npm registry access is available (this development environment cannot resolve the registry). Update Dockerfile to `npm ci` once lockfile is present.

## Safety and data boundaries

- The browser keeps the two-hour **demo** session token in `sessionStorage`, not `localStorage`. Do **not** use this authentication scheme for real bank customers.
- Demo company access is enforced server-side. The UI does not bypass it.
- The event list shows only the selected company; job IDs can be fetched only by their issuing demo session.
- User-provided news text is displayed as React text (not HTML). URLs are hyperlinks only; no server-side fetch or verification.
- This is a prototype for a local machine. Do not expose Docker's demo API to the Internet with sensitive datasets.

## Operational UI (no AI)

The dashboard now requests recent ingestion jobs using `GET /api/v1/jobs?limit=20`,
scoped to the *exact demo session*. A page reload in the same session restores
jobs from PostgreSQL. A new login creates a new session and cannot view the
previous session's jobs; that company's events remain accessible.

The Events tab retrieves 20 records at a time using the bounded `offset`
parameter of `GET /api/v1/companies/{id}/events`. Its search box searches only
currently loaded titles, not the full database. Use **Показать ещё события** to
load older pages. See `docs/qa-news-ui.md` for the local manual test checklist.
