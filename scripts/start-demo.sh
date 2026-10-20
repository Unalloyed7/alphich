#!/usr/bin/env bash
# Start the local UI + API + worker + PostgreSQL with the bundled synthetic profiles.
set -euo pipefail
cd "$(dirname "$0")/.."
export DEMO_COMPANY_IDS="${DEMO_COMPANY_IDS:-DEMO_A,DEMO_B}"

docker compose up -d --build
if docker compose exec -T api python -c '
from app.core import Company, make_session_factory
from sqlalchemy import select
_, factory = make_session_factory()
with factory() as db:
    profiles = db.scalars(select(Company.id).limit(1)).all()
    demos = all(db.get(Company, company) is not None for company in ("DEMO_A", "DEMO_B"))
if demos: raise SystemExit(0)
if profiles: raise SystemExit(2)
raise SystemExit(1)
'; then
    echo "DEMO_A and DEMO_B are already available."
else
    rc=$?
    if [ "$rc" -eq 1 ]; then
        docker compose run --rm -v ./backend/fixtures:/fixtures:ro api python -m app.importer /fixtures/companies.csv
    else
        echo >&2 "Existing database contains another dataset. Refusing to overwrite it."
        echo >&2 "See frontend/README.md to switch datasets deliberately."
        exit "$rc"
    fi
fi

echo "Open http://127.0.0.1:5173 and choose DEMO_A / CFO"
