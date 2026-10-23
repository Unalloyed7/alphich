# Alphich frontend

React + TypeScript + Vite UI. The UI performs real requests to the backend (no demo/mock event generator):

- `POST /api/v1/demo-session` — server-validated company/role selection;
- `GET /api/v1/companies`, `/companies/{id}/profile` — company data;
- `POST /api/v1/publications` with a unique `Idempotency-Key` — new publication;
- `GET /api/v1/jobs/{id}` — polls worker status;
- `GET /api/v1/companies/{id}/events` — stored events for the authorized company;
- `GET /api/v1/events/{id}`, `/sources/{id}` — event details and raw source.

The Analytics tab and event panel connect the feed, insight generation, questions, evidence and audit endpoints described in [docs/llm.md](../docs/llm.md). Generation starts only after an explicit user action. Real organizer gateway access is verified; credentials stay on the server. Full container/browser verification remains to be run using [the integration checklist](../docs/integration-smoke.md).

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
npm ci
npm run dev
```

Open http://127.0.0.1:5173. Vite proxies `/api` to the API at `127.0.0.1:8000`, so no CORS or frontend API keys are needed. Docker uses an Nginx reverse proxy with the same paths. `npm run build` validates TypeScript and builds the optimized frontend.

Runtime and development dependencies are pinned in `package.json` and `package-lock.json`.
Docker uses `npm ci` and includes `public/` (the application favicon).

## Redesigned briefing UI

The overview starts with a dark briefing summary and three interactive widgets: importance distribution, available bank views and data-check actions. Widgets and cards share one read-only feed state. Missing data is shown separately from zero values. Publication processing
is available under a disclosure. The full feed supports importance/bank-position filters,
search and sorting. Event analysis has separate summary, evidence and question views; original
news sources remain accessible. The layout supports desktop and phone navigation.

Detailed repository review, design rationale and integration findings: [frontend-redesign.md](../docs/frontend-redesign.md).

The decision workspace adds bookmarked/unread filters, keyboard quick search (Ctrl/Cmd+K or `/`),
event deep links and browser history, a breakdown of claim origins, dated bank citations,
readable audit steps and explicit server-saved feedback. Search covers loaded data only.
Bookmarks and reading state stay in this browser tab, scoped by company and role; feedback
belongs to the current demo session and saved insight. See [criteria mapping and limits](../docs/frontend-decision-workspace.md).

### User and company profile

«Моя компания» opens a visible profile of the active role and company, with role priorities,
industry, registration, raw revenue and product/trade flags. Missing data is explicit;
source versions and detailed flags are available in disclosures. Role changes use the
existing server-validated session flow. Failed profile loads have a retry action.
CEO and CFO remain the two supported analysis perspectives; the audience rationale and
expansion plan are in [user-perspectives.md](../docs/user-perspectives.md).

### Browser verification

Start Vite in one terminal:

```sh
npm ci
npm run dev -- --port 5174
```

Run in another terminal (from `frontend`):

```sh
npx playwright install chromium
npm run test:ui
```

Alternatively use an installed Chrome:

```sh
PLAYWRIGHT_CHANNEL=chrome npm run test:ui
```

`UI_BASE_URL` overrides `http://127.0.0.1:5174`. Set `UI_SCREENSHOT_DIR` to a local directory
if screenshots are needed. Browser tests use synthetic intercepted API responses, never the
live analysis gateway or paid model requests. The 14 scenarios cover slow-feed request deduplication and timeout backoff, dashboard values and navigation, missing/empty/ungraded data, filters/search, bookmarks, deep links, browser history, explicit feedback, evidence, raw sources,
question submission, publication/generation/discovery actions, company switching, responsive
layouts at 320/390/768/1024/1440 px, focus trapping and automated WCAG checks.

A running backend must match the checked-out repository. Older containers can lack
`/discovery`, feedback routes and relevance fields. Rebuild services together when preparing the integrated demo;
the UI displays unavailable states without inventing missing grades or results.

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
