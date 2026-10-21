# Alphich: RSS integration and public-demo acceptance

## Goal and cost boundary

RSS updates are read-only from the browser and do **not** call Qwen/GLM. The existing RSS system jobs are not owned by a browser session; therefore the Events list polls its company endpoint every 15 seconds while the page is visible, and the ranked Feed polls every 20 seconds. No frontend timer posts to `/insights` or `/questions`. A new company/role resets event lists, cards, jobs and source views before showing the new session.

## Local rollout without resetting the database

```sh
git fetch origin
git switch main
git pull --ff-only
git switch -c feat/rss-frontend-docker-acceptance
git apply --check ~/Downloads/alphich_rss_integration.patch
git apply ~/Downloads/alphich_rss_integration.patch
node --experimental-transform-types --test scripts/tests/test_recent_events.mjs # or use Node 22 Docker image
docker compose up -d --build
python3 scripts/verify-stack.py --import-fixtures
```

Offline six-container integration (runs a test-only collector overlay; **do not use on the public deployment**):

```sh
python3 scripts/verify-rss-stack.py
```

The smoke config (`scripts/rss-smoke.compose.yml`) temporarily replaces the RSS collector's live polling command with an idle command. A separate command in that service injects deterministic feed and article HTML into the **actual PostgreSQL / ingestion worker**. It checks `RSS → source + job → event → API`, replays a 304 after restarting the collector, and checks there is still one job/source/event. It does not fetch external articles, call an LLM, or use a demo-session token to create jobs. It inserts one synthetic article in `DEMO_A`. Once complete the test service is stopped, avoiding an accidentally suspended collector in production.

To start the **real** RSS collector after the acceptance test:

```sh
docker compose --profile news up -d --force-recreate rss-collector
docker compose --profile news exec -T rss-collector python -m app.rss --status
```

Use `RSS_COMPANY_IDS=DEMO_A,DEMO_B` (or another allowlisted, imported subset). Check `docker compose logs --tail=100 rss-collector`. Note that the first live run depends on RSS hosts and their article extraction policies.

## Manual browser acceptance

1. DEMO_A/CFO → Events, keep the page open. When RSS collector/worker stores a new article it should appear within approximately 15 seconds, without a full page reload. The ranked Analytics feed refreshes independently within approximately 20 seconds, but only includes items selected as relevant.
2. Open RSS event and primary source. Confirm title and full extracted text. RSS data provenance should be distinguished from manually pasted text; individual links can change or fail.
3. Click "Сменить", sign into DEMO_B/CEO, confirm no DEMO_A event/cards are visible and profile is correct. Repeat DEMO_A/CEO and DEMO_B/CFO. Validate isolation through `python3 scripts/verify-personas.py`.
4. On a test database that **already has a saved insight** for the scoped role, run `python3 scripts/verify-personas.py --require-insight`. It must fail rather than silently skip if there is no saved insight. Verify saved evidence, journal and answer in the browser. Do not generate new paid content without explicit budget approval.
5. Restart the collector (`docker compose --profile news restart rss-collector`) and refresh the source state using `python -m app.rss --status`. The previously imported article must not multiply.

## Deployment preparation (no VM purchased yet)

- Prefer Ubuntu 24.04 `x86_64` VPS with ≥4 vCPU, ≥8 GB RAM, ≥60 GB SSD. Check actual ONNX E5 RAM and Linux compatibility using `docker compose exec analysis-worker ...` before demonstrating.
- Keep the existing DB on a persistent Docker volume. **Never run `docker compose down -v`** on a populated environment. Back up PostgreSQL before upgrades using `pg_dump`, and retain an offline verified copy.
- Copy ignored `data/models/e5-base` and `data/processed/bank-index-v1` to the host privately; check exact version/hash. Never commit them or `.env`/API keys to GitHub.
- Configure `DEMO_COMPANY_IDS`, `RSS_COMPANY_IDS`, a unique `POSTGRES_PASSWORD`, and a deliberately low `LLM_DAILY_LIMIT_RUB`. Disable paid actions until their shared budget is approved.
- Place an HTTPS reverse proxy (e.g. Caddy) on host port 443 with authentication covering the **entire** website including `/api/*` to prevent public users from consuming LLM budget. Bind frontend/API to loopback only; do not expose PostgreSQL (5432). Restrict SSH with firewall and keys.
- On first start, ensure DEMO_A and DEMO_B are imported without replacing an existing company dataset. Verify app health, free stack smoke and four personas. Existing Mac insight IDs do not transfer with source code; use a deliberate database migration/restore or an explicitly approved new demo analysis.
- Verify from an external network; then test restart, disk space, memory pressure, backups and recovery. Plan removal/stop and cost alerts after 14 days.
