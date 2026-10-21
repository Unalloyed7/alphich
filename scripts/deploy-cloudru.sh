#!/usr/bin/env bash
# Manual, CI-gated deployment for the existing Cloud.ru Alphich VPS.
# Run as user alphich, with a full 40-character commit SHA from GitHub Actions.
set -Eeuo pipefail
umask 077

TARGET_SHA="${1:-}"
ROOT="/home/alphich/alphich"
OVERLAY="/home/alphich/.config/alphich/compose.vpn.yml"
LOCK="/home/alphich/.config/alphich/deploy.lock"

if [[ ! "$TARGET_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo >&2 'Expected a full 40-character hexadecimal commit SHA.'
  exit 2
fi
if [[ "$(id -un)" != 'alphich' ]]; then
  echo >&2 'Deploy must run as user alphich, not root.'
  exit 2
fi
if [[ ! -f "$OVERLAY" || ! -s "$ROOT/.env" ]]; then
  echo >&2 'Missing private VPN Compose override or .env; refusing deploy.'
  exit 2
fi
if ! sudo -n true; then
  echo >&2 'Passwordless sudo for the Docker command is required.'
  exit 2
fi
cd "$ROOT"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo >&2 'Another deployment is in progress.'
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo >&2 'Git worktree has modifications; refusing to overwrite.'
  exit 1
fi

git fetch --no-tags origin main
REMOTE_MAIN="$(git rev-parse refs/remotes/origin/main)"
if [[ "$REMOTE_MAIN" != "$TARGET_SHA" ]]; then
  echo >&2 'Selected CI-tested commit is not current origin/main; refusing deploy.'
  exit 1
fi
CURRENT="$(git rev-parse HEAD)"
if [[ "$CURRENT" == "$TARGET_SHA" ]]; then
  echo "REDEPLOY_SAME_COMMIT: $TARGET_SHA"
fi
if ! git merge-base --is-ancestor "$CURRENT" "$TARGET_SHA"; then
  echo >&2 'Non-fast-forward update detected; refusing deploy.'
  exit 1
fi

# Snapshot the live DB before changing the working tree or containers.
mkdir -p backups
chmod 700 backups
TMP_BACKUP="$(mktemp "$ROOT/backups/.predeploy-XXXXXXXX.dump")"
trap 'rm -f "$TMP_BACKUP"' EXIT
sudo -n docker compose -f compose.yaml -f "$OVERLAY" \
  exec -T postgres sh -eu -c \
  'exec pg_dump --format=custom --compress=6 --no-owner --no-privileges -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
  > "$TMP_BACKUP"
if [[ ! -s "$TMP_BACKUP" ]]; then
  echo >&2 'Database backup is empty; refusing deploy.'
  exit 1
fi
sudo -n docker run --rm --network none --read-only \
  --mount "type=bind,src=$ROOT/backups,dst=/archives,readonly" \
  --entrypoint pg_restore postgres:16.8 \
  --list "/archives/$(basename "$TMP_BACKUP")" >/dev/null
BACKUP="$ROOT/backups/alphich-predeploy-$(date -u +%Y%m%dT%H%M%SZ)-${TARGET_SHA:0:12}.dump"
mv -- "$TMP_BACKUP" "$BACKUP"
chmod 600 "$BACKUP"
(cd "$ROOT/backups" && sha256sum "$(basename "$BACKUP")" > "$(basename "$BACKUP").sha256")
chmod 600 "$BACKUP.sha256"
echo "BACKUP_OK: $BACKUP"

# Do not run destructive migrations or delete the PostgreSQL volume.
git merge --ff-only "$TARGET_SHA"
sudo -n docker compose -f compose.yaml -f "$OVERLAY" --profile news config --quiet
sudo -n docker compose -f compose.yaml -f "$OVERLAY" --profile news \
  up -d --build --wait --wait-timeout 180
curl --fail --silent --show-error --retry 12 --retry-delay 3 \
  --retry-connrefused --max-time 5 http://127.0.0.1:8000/api/v1/health >/dev/null
sudo -n docker compose -f compose.yaml -f "$OVERLAY" --profile news ps
echo "DEPLOY_OK: $(git rev-parse HEAD)"
