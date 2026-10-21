#!/usr/bin/env bash
# Test-restore an Alphich pg_dump in an isolated ephemeral PostgreSQL 16 container.
# NEVER accesses/replaces the live database or its Docker Compose volume.
set -euo pipefail
umask 077

usage() {
  echo >&2 "Usage: bash scripts/verify-postgres-backup.sh PATH_TO_DUMP [--require-insight]"
  exit 2
}
[ "$#" -ge 1 ] && [ "$#" -le 2 ] || usage
ARCHIVE="$1"
REQUIRE_INSIGHT=0
if [ "$#" -eq 2 ]; then
  [ "$2" = "--require-insight" ] || usage
  REQUIRE_INSIGHT=1
fi
[ -f "$ARCHIVE" ] && [ -s "$ARCHIVE" ] || { echo >&2 "Backup missing or empty: $ARCHIVE"; exit 1; }
backup_dir="$(cd "$(dirname "$ARCHIVE")" && pwd -P)"
backup_name="$(basename "$ARCHIVE")"
archive="$backup_dir/$backup_name"
checksum="$archive.sha256"
if [ ! -f "$checksum" ]; then
  echo >&2 "Missing checksum sidecar: $checksum"
  exit 1
fi
expected="$(awk 'NR == 1 {print $1}' "$checksum")"
if command -v shasum >/dev/null 2>&1; then
  actual="$(shasum -a 256 "$archive" | awk '{print $1}')"
elif command -v sha256sum >/dev/null 2>&1; then
  actual="$(sha256sum "$archive" | awk '{print $1}')"
else
  echo >&2 "No SHA-256 tool found."
  exit 1
fi
if [ -z "$expected" ] || [ "$actual" != "$expected" ]; then
  echo >&2 "SHA256 mismatch; refusing to restore archive."
  exit 1
fi
echo 'PASS backup SHA-256'
command -v docker >/dev/null 2>&1 || { echo >&2 'Docker is required'; exit 1; }

container="alphich-backup-check-$$-${RANDOM}"
started=0
cleanup() {
  if [ "$started" -eq 1 ]; then
    docker rm -f "$container" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo 'Starting isolated temporary PostgreSQL 16 (no network, no published ports)...'
docker run -d --rm --network none \
  --name "$container" \
  -e POSTGRES_PASSWORD=local_restore_verification_only \
  -e POSTGRES_DB=alphich_restore_check \
  postgres:16.8 >/dev/null
started=1
ready=0
for _ in $(seq 1 45); do
  if docker exec "$container" pg_isready -U postgres -d alphich_restore_check >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
[ "$ready" -eq 1 ] || { echo >&2 'Temporary PostgreSQL did not start'; exit 1; }

docker cp "$archive" "$container:/tmp/alphich-backup.dump"
docker exec "$container" pg_restore \
  --exit-on-error --no-owner --no-privileges \
  -U postgres -d alphich_restore_check /tmp/alphich-backup.dump

# Required schema. These counts refer exclusively to the isolated test database.
counts="$(docker exec "$container" psql -XAt -v ON_ERROR_STOP=1 \
  -U postgres -d alphich_restore_check -c \
  "SELECT (SELECT count(*) FROM companies)::text || ',' ||
          (SELECT count(*) FROM events)::text || ',' ||
          (SELECT count(*) FROM source_documents)::text || ',' ||
          (SELECT count(*) FROM insights)::text || ',' ||
          (SELECT count(*) FROM insight_answers)::text;")"
IFS=',' read -r companies events sources insights answers <<< "$counts"
for count in "$companies" "$events" "$sources" "$insights" "$answers"; do
  [[ "$count" =~ ^[0-9]+$ ]] || { echo >&2 "Unexpected database counts: $counts"; exit 1; }
done
if [ "$companies" -eq 0 ] || [ "$events" -eq 0 ] || [ "$sources" -eq 0 ]; then
  echo >&2 "Incomplete demo data after restore: $counts"
  exit 1
fi
if [ "$REQUIRE_INSIGHT" -eq 1 ] && [ "$insights" -eq 0 ]; then
  echo >&2 'No saved insights restored (--require-insight)'
  exit 1
fi
echo "PASS restored rows: companies=$companies events=$events sources=$sources insights=$insights answers=$answers"
echo 'RESTORE_CHECK_OK: live database was not modified'
