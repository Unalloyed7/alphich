#!/usr/bin/env bash
# Consistent PostgreSQL backup from running Alphich Compose; no LLM/API calls.
set -euo pipefail
umask 077

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
BACKUP_DIR="${ALPHICH_BACKUP_DIR:-$ROOT/backups}"

if [ "$#" -ne 0 ]; then
  echo >&2 "Usage: bash scripts/backup-postgres.sh"
  exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
  echo >&2 "Docker is required. Start Docker Desktop and the Alphich stack first."
  exit 1
fi

mkdir -p "$BACKUP_DIR"
chmod 700 "$BACKUP_DIR"
BACKUP_DIR="$(cd "$BACKUP_DIR" && pwd -P)"
temporary="$(mktemp "$BACKUP_DIR/.postgres-XXXXXXXX.dump")"
cleanup() { rm -f "$temporary"; }
trap cleanup EXIT

cd "$ROOT"
echo "Creating consistent pg_dump snapshot from Compose service postgres..."
# Credentials are supplied by the existing container environment, not printed.
# pg_dump uses a transactionally consistent snapshot and does not stop writers.
if ! docker compose exec -T postgres sh -eu -c \
  'exec pg_dump --format=custom --compress=6 --no-owner --no-privileges -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
  > "$temporary"; then
  echo >&2 "Backup failed. No incomplete backup is retained."
  exit 1
fi
if [ ! -s "$temporary" ]; then
  echo >&2 "Backup is empty; aborting."
  exit 1
fi

# Check archive TOC without touching the production database.
docker run --rm --network none --read-only \
  --mount "type=bind,src=$BACKUP_DIR,dst=/archives,readonly" \
  --entrypoint pg_restore postgres:16.8 \
  --list "/archives/$(basename "$temporary")" >/dev/null

filename="alphich-postgres-$(date -u '+%Y%m%dT%H%M%SZ')-${RANDOM}${RANDOM}.dump"
destination="$BACKUP_DIR/$filename"
mv "$temporary" "$destination"
chmod 600 "$destination"
if command -v shasum >/dev/null 2>&1; then
  digest="$(shasum -a 256 "$destination" | awk '{print $1}')"
elif command -v sha256sum >/dev/null 2>&1; then
  digest="$(sha256sum "$destination" | awk '{print $1}')"
else
  echo >&2 "Neither shasum nor sha256sum is installed; cannot verify backup checksum."
  rm -f "$destination"
  exit 1
fi
printf '%s  %s\n' "$digest" "$filename" > "$destination.sha256"
chmod 600 "$destination.sha256"
echo "BACKUP_OK: $destination"
echo "SHA256: $digest"
echo "Next: bash scripts/verify-postgres-backup.sh '$destination' --require-insight"
echo "Keep .dump and .sha256 private; do not commit or upload them."
