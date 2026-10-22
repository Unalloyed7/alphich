#!/usr/bin/env python3
"""Versioned RSS corpus -> independent Quality Lab workflow.

Uses only stdlib, never calls LLM. `prepare` works on an already transferred
Cloud.ru snapshot. `sync` reads public rss_articles through SSH (read-only)
then runs the identical validation and preparation. Explicit human approval is
required before the SQLite benchmark can be modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_DB = ROOT / 'data/private/quality-review.sqlite3'
DEFAULT_RUNS = ROOT / 'data/private/quality-pipeline'
FIXTURES = ROOT / 'knowledge/fixtures'
MAX_ARTICLES = 10000
MAX_TEXT_LENGTH = 100_000
MAX_SNAPSHOT_BYTES = 250 * 1024 * 1024
SOURCES = ('v2', 'v3')
# Public RSS source domains. New feeds require an explicit review of this allowlist.
PUBLIC_HOSTS = frozenset({
    'www.cbr.ru','www.interfax.ru','stroygaz.ru','www.cnews.ru',
    'zoom.cnews.ru','www.agroinvestor.ru','www.elec.ru','mintrans.gov.ru',
    'www.mintrans.gov.ru','www.trendforce.com','www.nalog.gov.ru',
})


def digest(raw):
    if isinstance(raw, str):
        raw = raw.encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8')


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def private_file(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as target:
        target.write(data)
    path.chmod(0o600)


def exact_public_url(url):
    if not isinstance(url, str) or len(url) > 2048:
        return None
    try:
        p = urlsplit(url)
        if (p.scheme != 'https' or p.hostname not in PUBLIC_HOSTS or p.username or p.password
                or p.port not in (None, 443) or p.fragment):
            return None
        # Preserve query (CBR's ?file=... is the article identity).
        return url
    except ValueError:
        return None


def article_date_valid(value):
    if value is None:
        return True
    if not isinstance(value, str) or not 8 <= len(value) <= 40:
        return False
    try:
        datetime.fromisoformat(value.replace('Z', '+00:00'))
        return True
    except ValueError:
        return False


def fixtures_map(folder):
    events, hashes = {}, {}
    for dataset in SOURCES:
        p = folder / f'real_news_selection_labels.{dataset}.json'
        raw = p.read_bytes()
        value = json.loads(raw)
        hashes[dataset] = digest(raw)
        for s in value['sources']:
            key = (dataset, s['event_id'])
            if key in events or not exact_public_url(s.get('url')):
                raise ValueError('Duplicate event ID or disallowed URL in fixture')
            events[key] = {'title': s.get('title') or s.get('short_title') or s['event_id'], 'url': s['url']}
    return events, hashes


def load_snapshot(path):
    if not path.is_file() or path.stat().st_size > MAX_SNAPSHOT_BYTES:
        raise ValueError('Snapshot missing or too large')
    raw = path.read_bytes()
    # Both prior manual export JSON and one-JSON-object-per-line RSS stream.
    if path.suffix == '.jsonl':
        articles = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
        claimed = None
    else:
        val = json.loads(raw)
        if not isinstance(val, dict) or not isinstance(val.get('articles'), list):
            raise ValueError('Expected {articles:[...], matches?:[...]} export')
        articles = val['articles']
        claimed = val.get('matches')
    if len(articles) > MAX_ARTICLES:
        raise ValueError('Snapshot exceeds maximum article count')
    ids = set()
    for article in articles:
        if not isinstance(article, dict):
            raise ValueError('Invalid article in RSS snapshot')
        ident = article.get('id')
        if not isinstance(ident, str) or not re.fullmatch(r'[0-9a-f]{64}', ident) or ident in ids:
            raise ValueError('Duplicate/invalid RSS article ID')
        ids.add(ident)
        if not exact_public_url(article.get('url')) or not exact_public_url(article.get('final_url')):
            raise ValueError('Disallowed RSS URL')
        body = article.get('body')
        if not isinstance(body, str) or not 80 <= len(body) <= MAX_TEXT_LENGTH:
            raise ValueError('Missing, too short or oversized article text')
        if not article_date_valid(article.get('published_at')):
            raise ValueError('Invalid article published_at')
    if claimed is not None:
        if not isinstance(claimed, list):
            raise ValueError('Invalid matches array')
        for entry in claimed:
            if not isinstance(entry, list) or len(entry) != 3 or entry[0] not in SOURCES:
                raise ValueError('Invalid previous match entry')
            if entry[2] not in ids:
                raise ValueError('Match references missing article')
    return articles, digest(raw)


def database_status(db_path, fixture_hashes):
    if not db_path.is_file():
        raise ValueError('Quality Lab DB not found: ' + str(db_path))
    with sqlite3.connect(db_path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        meta = dict(db.execute('SELECT key,value FROM metadata'))
        for version, sha in fixture_hashes.items():
            if meta.get(version + '_sha256') != sha:
                raise ValueError('Quality Lab fixture version differs: ' + version)
        data = {}
        for row in db.execute('SELECT dataset,event_id,url,article_text,published_at FROM pairs'):
            key = (row['dataset'], row['event_id'])
            data.setdefault(key, []).append(dict(row))
        review_count = db.execute('SELECT COUNT(*) FROM reviews').fetchone()[0]
        arbitration_count = db.execute('SELECT COUNT(*) FROM adjudications').fetchone()[0]
    return data, bool(review_count or arbitration_count)


def choose_article(articles, target_url):
    possible = [a for a in articles if target_url in (a['url'], a['final_url'])]
    # Do not silently choose one of multiple versions of a URL.
    if len(possible) > 1:
        return None, 'ambiguous'
    return (possible[0], 'matched') if possible else (None, 'missing')


def prepare(*, snapshot, db_path, fixtures, output_dir):
    if output_dir.exists():
        raise ValueError('Run directory already exists: ' + str(output_dir))
    articles, snapshot_sha = load_snapshot(snapshot)
    events, fixture_hashes = fixtures_map(fixtures)
    current, locked = database_status(db_path, fixture_hashes)
    records, candidates = [], {}
    counts = {'new': 0, 'already_present': 0, 'conflict': 0,
              'missing': 0, 'ambiguous': 0, 'not_in_db': 0}
    for (dataset, event_id), source in sorted(events.items()):
        original = source['url']
        candidate, state = choose_article(articles, original)
        current_rows = current.get((dataset, event_id), [])
        if not current_rows:
            state = 'not_in_db'
        elif any(r['url'] != original for r in current_rows):
            raise ValueError('Fixture/DB source URL mismatch: ' + event_id)
        elif candidate is not None:
            old = {r['article_text'] for r in current_rows if r['article_text'] is not None}
            if old:
                state = 'already_present' if old == {candidate['body']} else 'conflict'
            else:
                state = 'new'
        counts[state] += 1
        row = {'dataset': dataset, 'event_id': event_id, 'title': source['title'],
               'url': original, 'status': state, 'rss_id': candidate['id'] if candidate else None,
               'date': candidate.get('published_at') if candidate else None,
               'body_chars': len(candidate['body']) if candidate else None,
               'body_sha256': digest(candidate['body']) if candidate else None,
               'company_pairs': len(current_rows)}
        records.append(row)
        if state == 'new':
            if event_id in candidates and candidates[event_id]['text'] != candidate['body']:
                raise ValueError('Event ID collision across datasets: ' + event_id)
            candidates[event_id] = {'text': candidate['body'], 'published_at': candidate['published_at']}
    # Do not lose the rest of the RSS feed: retain it as a separate, *unlabelled*
    # source catalogue for a future larger review dataset (not this frozen DB).
    matched_ids = {r['rss_id'] for r in records if r['rss_id'] is not None}
    unmatched = [a for a in articles if a['id'] not in matched_ids]
    # Create only after validation is complete. State is immutable and private.
    output_dir.mkdir(parents=True, exist_ok=False)
    output_dir.chmod(0o700)
    manifest = {'schema': 'alphich-quality-corpus-v1', 'prepared_at': now(),
                'source_file_sha256': snapshot_sha, 'source_file': snapshot.name,
                'fixture_sha256': fixture_hashes, 'db_had_human_reviews': locked,
                'counts': counts, 'records': records,
                'candidate_articles_sha256': digest(json_bytes(candidates))}
    manifest['unmatched_rss_articles'] = len(unmatched)
    unmatched_bytes = b''.join((json.dumps(a, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8') for a in unmatched)
    manifest['unmatched_rss_sha256'] = digest(unmatched_bytes)
    private_file(output_dir / 'manifest.json', json_bytes(manifest))
    private_file(output_dir / 'candidate_articles.json', json_bytes(candidates))
    private_file(output_dir / 'unmatched_rss_articles.jsonl', unmatched_bytes)
    e = html.escape
    blocks = ['<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Alphich RSS verification</title>',
              '<style>body{font:15px/1.5 system-ui;max-width:960px;margin:28px auto}article{border:1px solid #ccc;padding:18px;margin:18px 0;border-radius:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere}.warn{color:#9a4b00}</style></head><body>',
              '<h1>Сверка RSS-источников · только для оператора</h1>',
              '<p class="warn">Тексты не подтверждены человеком. Проверяйте оригинал и дату перед approve.</p>']
    for r in records:
        if r['status'] != 'new':
            continue
        art = candidates[r['event_id']]
        blocks += [f'<article><h2>{e(r["title"])}</h2>',
                   f'<p>event_id: <code>{e(r["event_id"])}</code> · {e(str(r["date"] or "Дата неизвестна"))}</p>',
                   f'<p><a href="{e(r["url"],quote=True)}" rel="noopener noreferrer" target="_blank">Открыть оригинал</a></p>',
                   '<pre>' + e(art['text']) + '</pre></article>']
    blocks.append('</body></html>')
    private_file(output_dir / 'preview.html', '\n'.join(blocks).encode('utf-8'))
    return {'run_dir': str(output_dir), 'counts': counts, 'db_locked': locked,
            'unmatched_rss_articles': len(unmatched),
            'new_pair_texts': sum(r['company_pairs'] for r in records if r['status'] == 'new')}


def approve(*, run_dir, db_path, fixtures, ids):
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Approve only distinct explicitly human-verified event IDs')
    m = json.loads((run_dir / 'manifest.json').read_text('utf-8'))
    raw = (run_dir / 'candidate_articles.json').read_bytes()
    if digest(raw) != m['candidate_articles_sha256']:
        raise ValueError('Candidate file changed after prepare')
    candidates = json.loads(raw)
    if any(ident not in candidates for ident in ids):
        raise ValueError('Unmatched or non-new event ID; inspect preview first')
    _, hashes = fixtures_map(fixtures)
    if hashes != m['fixture_sha256']:
        raise ValueError('Fixture changed after prepare')
    current, locked = database_status(db_path, hashes)
    if locked:
        raise ValueError('Quality Lab already has human reviews; benchmark inputs are frozen')
    records = {r['event_id']:r for r in m['records'] if r['status'] == 'new'}
    for ident in ids:
        if digest(candidates[ident]['text']) != records[ident]['body_sha256']:
            raise ValueError('Article contents changed after prepare')
        if not article_date_valid(candidates[ident]['published_at']):
            raise ValueError('Invalid publication date')
        for (dataset, event_id), rows in current.items():
            if event_id == ident and any(row['article_text'] is not None for row in rows):
                raise ValueError('Existing text must never be replaced: ' + ident)
    import quality_review as qr
    # Begin one atomic transaction, rechecking under SQLite write lock.
    with qr.managed(db_path) as db:
        db.execute('BEGIN IMMEDIATE')
        if (db.execute('SELECT COUNT(*) FROM reviews').fetchone()[0] or
                db.execute('SELECT COUNT(*) FROM adjudications').fetchone()[0]):
            raise ValueError('Human review started during import')
        pair_count = 0
        for ident in ids:
            r = records[ident]
            rows = db.execute('SELECT id,url,article_text FROM pairs WHERE event_id=?', (ident,)).fetchall()
            if not rows or any(x['url'] != r['url'] or x['article_text'] is not None for x in rows):
                raise ValueError('Source changed or already present; import aborted: ' + ident)
            val = candidates[ident]
            db.execute('UPDATE pairs SET article_text=?, published_at=COALESCE(?,published_at) WHERE event_id=? AND article_text IS NULL',
                       (val['text'], val['published_at'], ident))
            pair_count += len(rows)
        audit_key = 'quality_pipeline:' + digest(str(run_dir.resolve()) + ':' + ','.join(sorted(ids)))[:40]
        if db.execute('SELECT 1 FROM metadata WHERE key=?', (audit_key,)).fetchone():
            raise ValueError('Duplicate import')
        db.execute('INSERT INTO metadata(key,value) VALUES (?,?)',
                   (audit_key, json.dumps({'approved_ids': sorted(ids), 'manifest_sha256': digest(json_bytes(m)),
                                           'imported_at': now(), 'pairs': pair_count})))
    return {'approved_articles': len(ids), 'updated_pairs': pair_count, 'db': str(db_path)}


# A stable, server-side read-only query over the public RSS article table.
# PostgreSQL json_build_object safely JSON-escapes newlines and text quotes.
_REMOTE_QUERY = """SELECT json_build_object('id',id,'url',url,'final_url',final_url,
  'title',title,'body',body,'published_at',published_at,
  'fetched_at',fetched_at,'content_hash',content_hash)::text
FROM rss_articles
WHERE length(body) BETWEEN 80 AND 100000
  AND split_part(url,'/',3) <> 'rss-smoke.alphich-example.org'
ORDER BY id LIMIT 10000;"""


def sync(*, ssh_host, ssh_user, ssh_key, output_dir, db_path, fixtures):
    if not re.fullmatch(r'[a-zA-Z0-9_.-]+', ssh_user) or not re.fullmatch(r'[a-zA-Z0-9.-]+', ssh_host):
        raise ValueError('Invalid SSH endpoint')
    if output_dir.exists():
        raise ValueError('Run already exists')
    # No remote usernames, passwords, company profiles or auth secrets in SQL.
    remote = ("cd ~/alphich && sudo -n docker compose -f compose.yaml "
              "-f ~/.config/alphich/compose.vpn.yml exec -T postgres "
              "psql -U alphich -d alphich -XAt -v ON_ERROR_STOP=1 -c " +
              "'" + _REMOTE_QUERY.replace("'", "'\\''") + "'")
    cmd = ['ssh', '-i', str(ssh_key), '-o', 'IdentitiesOnly=yes',
           '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
           f'{ssh_user}@{ssh_host}', remote]
    # No interactive password prompts; if sudo needs password, fail cleanly.
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
                          check=False)
    if proc.returncode != 0:
        raise ValueError('Remote read failed (check SSH key, sudo -n and compose); nothing imported')
    if len(proc.stdout) > MAX_SNAPSHOT_BYTES:
        raise ValueError('Remote RSS response exceeds size cap')
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    parent.chmod(0o700)
    # Snapshot belongs in the private run folder; do not expose remote stderr.
    output_dir.mkdir(mode=0o700, exist_ok=False)
    snapshot = output_dir / 'rss_snapshot.jsonl'
    try:
        private_file(snapshot, proc.stdout)
        # prepare() owns its own immutable directory, sibling of snapshot.
        result = prepare(snapshot=snapshot, db_path=db_path, fixtures=fixtures,
                         output_dir=output_dir / 'review')
        result['snapshot'] = str(snapshot)
        return result
    except BaseException:
        # Keep raw private snapshot for operator forensics, no deletion of evidence.
        raise


def main():
    ap = argparse.ArgumentParser(description='Local verified RSS->Quality Lab pipeline (no LLM)')
    sub = ap.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='Stage a transferred read-only RSS snapshot')
    prep.add_argument('--rss-export', type=Path, required=True)
    prep.add_argument('--run', required=True)
    sync_p = sub.add_parser('sync', help='Read public RSS corpus from Cloud.ru over SSH and stage')
    sync_p.add_argument('--run', required=True)
    sync_p.add_argument('--ssh-host', default='89.232.176.172')
    sync_p.add_argument('--ssh-user', default='alphich')
    sync_p.add_argument('--ssh-key', type=Path, default=Path.home()/'.ssh/alphich_cloudru')
    app = sub.add_parser('approve', help='Import only manually verified new articles into Quality Lab')
    app.add_argument('--run', required=True)
    app.add_argument('--ids', nargs='+', required=True)
    for p in (prep, sync_p, app):
        p.add_argument('--runs-dir', type=Path, default=DEFAULT_RUNS)
        p.add_argument('--db', type=Path, default=DEFAULT_DB)
        p.add_argument('--fixtures-dir', type=Path, default=FIXTURES)
    args = ap.parse_args()
    try:
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,49}', args.run):
            raise ValueError('Run ID must be alphanumeric, dash or underscore')
        path = args.runs_dir / args.run
        if args.command == 'prepare':
            outcome = prepare(snapshot=args.rss_export, db_path=args.db,
                              fixtures=args.fixtures_dir, output_dir=path)
        elif args.command == 'sync':
            outcome = sync(ssh_host=args.ssh_host, ssh_user=args.ssh_user, ssh_key=args.ssh_key,
                           output_dir=path, db_path=args.db, fixtures=args.fixtures_dir)
        else:
            review = path / 'review' if (path / 'review').is_dir() else path
            outcome = approve(run_dir=review, db_path=args.db, fixtures=args.fixtures_dir, ids=args.ids)
        print(json.dumps(outcome, ensure_ascii=False))
    except (ValueError, OSError, sqlite3.Error, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        ap.exit(2, 'ERROR: ' + str(exc) + '\n')


if __name__ == '__main__':
    main()
