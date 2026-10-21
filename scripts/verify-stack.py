#!/usr/bin/env python3
"""Non-destructive Alphich Docker/API integration smoke test.

By default: no LLM calls, no budget consumption, no database reset.
--live-analysis explicitly permits paid requests via the existing analysis-worker.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SERVICES = {'postgres', 'api', 'worker', 'analysis-worker', 'frontend'}


class CheckFailed(RuntimeError):
    pass


def compose(*args: str, capture: bool = False) -> str:
    command = ['docker', 'compose', *args]
    try:
        result = subprocess.run(command, cwd=ROOT, check=True, text=True,
                                stdout=subprocess.PIPE if capture else None)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise CheckFailed(f"Docker Compose command failed: {' '.join(command)}") from exc
    return result.stdout.strip() if capture else ''


def request(base: str, method: str, path: str, *, token: str | None = None,
            payload: dict | None = None, key: str | None = None, timeout: float = 15) -> tuple[int, dict]:
    headers = {'Accept': 'application/json'}
    if token:
        headers['X-Demo-Session'] = token
    if key:
        headers['Idempotency-Key'] = key
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
    if body is not None:
        headers['Content-Type'] = 'application/json'
    req = Request(base + path, data=body, headers=headers, method=method)
    try:
        with urlopen(req, timeout=timeout) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        try:
            data = json.load(error)
        except (ValueError, OSError):
            data = {}
        return error.code, data
    except (URLError, TimeoutError) as exc:
        raise CheckFailed(f'API unavailable: {path} ({type(exc).__name__})') from exc


def expect(status: int, expected: int, label: str, data: dict | None = None) -> None:
    if status != expected:
        detail = data.get('detail') if data else None
        raise CheckFailed(f'{label}: expected HTTP {expected}, got {status}, detail={detail}')
    print(f'PASS {label} (HTTP {status})', flush=True)


def wait_job(base: str, token: str, job_id: str, timeout_seconds: int) -> dict:
    deadline = time.monotonic() + timeout_seconds
    seen = set()
    while time.monotonic() < deadline:
        status, result = request(base, 'GET', f'/jobs/{job_id}', token=token)
        expect(status, 200, 'Job accessible') if status != 200 else None
        state = result.get('status')
        if state not in seen:
            print(f'  Job {job_id[:18]} -> {state} / {result.get("stage")}', flush=True)
            seen.add(state)
        if state == 'succeeded':
            if not result.get('result_id'):
                raise CheckFailed('Succeeded job has no result_id')
            return result
        if state == 'failed':
            raise CheckFailed(f'Job failed with safe error code: {result.get("error")}')
        time.sleep(2)
    raise CheckFailed(f'Job did not complete within {timeout_seconds}s')


def wait_api(base: str, timeout_seconds: int = 90) -> None:
    end = time.monotonic() + timeout_seconds
    while time.monotonic() < end:
        try:
            code, data = request(base, 'GET', '/health', timeout=3)
            if code == 200 and data.get('database') == 'ready':
                print('PASS API/PostgreSQL health', flush=True)
                return
        except CheckFailed:
            pass
        time.sleep(2)
    raise CheckFailed('Health did not become ready')


def session(base: str, company: str, role: str = 'cfo') -> tuple[int, dict]:
    return request(base, 'POST', '/demo-session', payload={'company_id': company, 'role': role})


def run(args: argparse.Namespace) -> None:
    if args.start:
        print('Starting all 5 services. Existing PostgreSQL volume is preserved.', flush=True)
        compose('up', '-d', '--build')
    running = set(compose('ps', '--services', '--status', 'running', capture=True).splitlines())
    if missing := SERVICES - running:
        raise CheckFailed(f'Missing running containers: {", ".join(sorted(missing))}')
    print('PASS 5 Docker services running: ' + ', '.join(sorted(SERVICES)), flush=True)
    base = args.base.rstrip('/') + '/api/v1'
    wait_api(base)
    # Also check the frontend's nginx entrypoint, without evaluating JavaScript.
    try:
        with urlopen(args.frontend, timeout=8) as response:
            if response.status != 200:
                raise CheckFailed(f'Frontend returned HTTP {response.status}')
    except (URLError, TimeoutError) as exc:
        raise CheckFailed('Frontend is unavailable') from exc
    print('PASS Frontend serves HTML', flush=True)

    code, auth = session(base, args.company)
    if code == 404 and args.import_fixtures and args.company in {'DEMO_A', 'DEMO_B'}:
        print('Demo profiles missing; importing fixtures without --replace.', flush=True)
        compose('run', '--rm', '-v', './backend/fixtures:/fixtures:ro', 'api',
                'python', '-m', 'app.importer', '/fixtures/companies.csv')
        code, auth = session(base, args.company)
    expect(code, 201, f'Demo session for {args.company}', auth)
    token = auth['token']  # Never log or persist session tokens.
    status, profile = request(base, 'GET', f'/companies/{args.company}/profile', token=token)
    expect(status, 200, 'Company profile', profile)
    if profile.get('id') != args.company:
        raise CheckFailed('Company profile mismatch')

    other_token = None
    other_company = 'DEMO_B' if args.company == 'DEMO_A' else 'DEMO_A'
    c, other = session(base, other_company)
    if c == 201:
        other_token = other['token']
    else:
        print('SKIP cross-company ACL: second demo company unavailable', flush=True)

    unique = uuid.uuid4().hex
    title = 'Тест конвейера публикаций — ' + unique[:12]
    text = ('Синтетическая публикация для автоматической интеграционной проверки. '
            'Здесь проверяется сохранение исходного текста и работа worker; '
            'новость не содержит подтверждённых фактов о внешнем мире. '
            f'Уникальный идентификатор: {unique}.')
    code, created = request(base, 'POST', '/publications', token=token,
                            key='smoke-publication-' + unique,
                            payload={'title': title, 'text': text})
    expect(code, 202, 'Publication queued', created)
    ingest = wait_job(base, token, created['id'], args.timeout)
    code, event = request(base, 'GET', '/events/' + ingest['result_id'], token=token)
    expect(code, 200, 'Event returned from PostgreSQL', event)
    if event.get('title') != title or event.get('company_id') != args.company:
        raise CheckFailed('Unexpected event content/owner')
    if not event.get('source_ids'):
        raise CheckFailed('Event has no source reference')
    source_id = event['source_ids'][0]
    code, source = request(base, 'GET', '/sources/' + source_id, token=token)
    expect(code, 200, 'Primary source returned', source)
    if source.get('text') != text or source.get('url_verified') is not False:
        raise CheckFailed('Source text/provenance mismatch')
    code, events = request(base, 'GET', f'/companies/{args.company}/events?limit=20&offset=0', token=token)
    expect(code, 200, 'Paginated event list', events)
    if not any(x.get('id') == event['id'] for x in events.get('items', [])):
        raise CheckFailed('New event absent from page one')
    code, jobs = request(base, 'GET', '/jobs?limit=20', token=token)
    expect(code, 200, 'Session job history', jobs)
    if not any(x.get('id') == created['id'] for x in jobs.get('items', [])):
        raise CheckFailed('New ingestion job absent from history')
    if other_token:
        cross, _ = request(base, 'GET', '/events/' + event['id'], token=other_token)
        expect(cross, 404, 'Cross-company event denied')
        cross, _ = request(base, 'GET', '/sources/' + source_id, token=other_token)
        expect(cross, 404, 'Cross-company source denied')
        cross, _ = request(base, 'GET', '/jobs/' + created['id'], token=other_token)
        expect(cross, 404, 'Cross-session job denied')

    code, configured = request(base, 'GET', '/llm/status', token=token)
    expect(code, 200, 'Shared LLM budget / readiness API', configured)
    print(f'  Configured={configured.get("configured")}, budget_mode={configured.get("budget_mode")}, '
          f'prices_confirmed={configured.get("prices_confirmed")}, '
          f'bank_index_present={configured.get("bank_index_present")}, spent_RUB={configured.get("spent_rub")}', flush=True)
    code, feed = request(base, 'GET', f'/companies/{args.company}/feed', token=token, timeout=60)
    expect(code, 200, 'Personalized feed API (read-only)', feed)
    if not isinstance(feed.get('items'), list):
        raise CheckFailed('Feed items must be a list')
    print(f'PASS feed returned {len(feed["items"])} candidates', flush=True)

    if not args.live_analysis:
        print('SKIP paid analysis & questions. Use --live-analysis to opt in.', flush=True)
        return
    # `configured` is authoritative: token_prices requires confirmed prices,
    # while gateway_receipt uses validated receipts and does NOT require
    # prices_confirmed=true. The API already validates its chosen budget mode.
    if configured.get('configured') is not True:
        raise CheckFailed('Live analysis requested but authorized model gateway/budget is not configured')
    if not configured.get('bank_index_present'):
        raise CheckFailed('Bank index is missing; build/attach it before paid analysis')

    code, analysis = request(base, 'POST', f'/companies/{args.company}/insights', token=token,
                             key='smoke-insight-' + unique, payload={'event_id': event['id']})
    expect(code, 202, 'Analysis job enqueued', analysis)
    analyzed = wait_job(base, token, analysis['id'], args.analysis_timeout)
    code, card = request(base, 'GET', '/insights/' + analyzed['result_id'], token=token)
    expect(code, 200, 'Insight returned from analysis-worker', card)
    if card.get('company_id') != args.company or card.get('event_id') != event['id']:
        raise CheckFailed('Insight scope mismatch')
    print('PASS insight status: ' + str(card.get('status')), flush=True)
    for claim in card.get('claims', [])[:3]:
        for evidence_id in claim.get('evidence_ids', [])[:3]:
            code, _ = request(base, 'GET', f'/evidence/{evidence_id}?insight_id={card["id"]}', token=token)
            expect(code, 200, 'Claim evidence accessible')
    if card.get('run_id'):
        code, _ = request(base, 'GET', '/runs/' + card['run_id'], token=token)
        expect(code, 200, 'Agent audit run accessible')
    if other_token:
        cross, _ = request(base, 'GET', '/insights/' + card['id'], token=other_token)
        expect(cross, 404, 'Cross-company insight denied')
    if args.ask_question:
        code, q = request(base, 'POST', f'/insights/{card["id"]}/questions', token=token,
                          key='smoke-question-' + unique,
                          payload={'question': 'Какие исходные данные нужно уточнить для более точной оценки?'})
        expect(code, 202, 'Question enqueued', q)
        answered = wait_job(base, token, q['id'], args.analysis_timeout)
        code, result = request(base, 'GET', '/answers/' + answered['result_id'], token=token)
        expect(code, 200, 'Follow-up answer retrieved', result)
        if result.get('insight_id') != card['id']:
            raise CheckFailed('Question not scoped to parent insight')
    code, final_budget = request(base, 'GET', '/llm/status', token=token)
    expect(code, 200, 'Final shared budget inspection', final_budget)
    print('  Shared budget spent_RUB:', final_budget.get('spent_rub'), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', action='store_true', help='Build and start all 5 Compose services')
    parser.add_argument('--import-fixtures', action='store_true', help='Import DEMO_A/B only when not yet in DB')
    parser.add_argument('--company', default='DEMO_A')
    parser.add_argument('--base', default='http://127.0.0.1:8000')
    parser.add_argument('--frontend', default='http://127.0.0.1:5173')
    parser.add_argument('--timeout', type=int, default=90, help='Ingestion job timeout')
    parser.add_argument('--analysis-timeout', type=int, default=600)
    parser.add_argument('--live-analysis', action='store_true', help='Explicit permission for budgeted model requests')
    parser.add_argument('--ask-question', action='store_true', help='One additional paid analysis question')
    args = parser.parse_args()
    if args.ask_question and not args.live_analysis:
        parser.error('--ask-question requires --live-analysis')
    try:
        run(args)
    except CheckFailed as error:
        print('FAIL ' + str(error), file=sys.stderr)
        return 1
    print('DONE: requested checks passed', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
