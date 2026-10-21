#!/usr/bin/env python3
"""Offline-safe 2x2 persona / ACL acceptance test against a running Alphich API.

No Qwen/GLM calls. Defaults to read-only requests; --seed-missing writes only
synthetic source texts if a company has no events. Never prints session tokens.
"""
from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
import sys
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class CheckFailed(RuntimeError):
    pass


class NoRedirect(HTTPRedirectHandler):
    """Never forward site credentials or demo tokens to a redirect target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def site_auth(base: str) -> dict[str, str]:
    """Optional reverse-proxy login, separate from the application demo token."""
    username = os.getenv('ALPHICH_SMOKE_USER', '')
    password = os.getenv('ALPHICH_SMOKE_PASSWORD', '')
    parsed = urlsplit(base)
    if parsed.username is not None or parsed.password is not None:
        raise CheckFailed('Do not embed credentials in the base URL')
    if not username and not password:
        return {}
    if not username or not password or ':' in username:
        raise CheckFailed('Set both ALPHICH_SMOKE_USER and ALPHICH_SMOKE_PASSWORD')
    if parsed.scheme != 'https' or not parsed.hostname:
        raise CheckFailed('Site Basic Auth requires an HTTPS base URL')
    encoded = base64.b64encode(f'{username}:{password}'.encode('utf-8')).decode('ascii')
    return {'Authorization': 'Basic ' + encoded}


def request(base: str, method: str, path: str, *, token: str | None = None,
            payload: dict | None = None, key: str | None = None) -> tuple[int, dict]:
    # Explicit allowlist: this script must never issue a billable LLM request.
    if method == 'POST' and path not in {'/demo-session', '/publications'}:
        raise CheckFailed(f'Forbidden POST for unpaid acceptance test: {path}')
    if method not in {'GET', 'POST'}:
        raise CheckFailed(f'Forbidden method: {method}')
    headers = {'Accept': 'application/json', **site_auth(base)}
    if token:
        headers['X-Demo-Session'] = token
    if key:
        headers['Idempotency-Key'] = key
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
    if body is not None:
        headers['Content-Type'] = 'application/json'
    try:
        with build_opener(NoRedirect()).open(
            Request(base + path, data=body, headers=headers, method=method), timeout=30
        ) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        try:
            data = json.load(error)
        except (ValueError, OSError):
            data = {}
        return error.code, data
    except (URLError, TimeoutError) as exc:
        raise CheckFailed(f'API unavailable at {path}: {type(exc).__name__}') from exc


def expect(code: int, wanted: int, label: str) -> None:
    if code != wanted:
        raise CheckFailed(f'{label}: expected HTTP {wanted}, got {code}')
    print(f'PASS {label} (HTTP {code})', flush=True)


def seeded_event(base: str, token: str, company: str, timeout: int = 90) -> tuple[str, str]:
    unique = uuid.uuid4().hex
    code, job = request(base, 'POST', '/publications', token=token,
                        key='persona-seed-' + unique,
                        payload={'title': f'Синтетическая проверка прав доступа {company} {unique[:8]}',
                                 'text': ('Синтетический текст для теста изоляции двух компаний. '
                                          'Он не является сообщением о реальной экономике. '
                                          f'Уникальный идентификатор {unique}.')})
    expect(code, 202, f'{company}: synthetic publication queued')
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code, state = request(base, 'GET', '/jobs/' + job['id'], token=token)
        expect(code, 200, f'{company}: seeded job scoped to session') if code != 200 else None
        if state.get('status') == 'succeeded':
            code, event = request(base, 'GET', '/events/' + state['result_id'], token=token)
            expect(code, 200, f'{company}: seeded event available')
            if not event.get('source_ids'):
                raise CheckFailed(f'{company}: seeded event has no source')
            return event['id'], event['source_ids'][0]
        if state.get('status') == 'failed':
            raise CheckFailed(f'{company}: ingestion failed: {state.get("error")}')
        time.sleep(2)
    raise CheckFailed(f'{company}: ingestion timeout')


def choose_saved_insight(feed: dict) -> str | None:
    for item in feed['items']:
        found = item.get('insight')
        if isinstance(found, dict) and isinstance(found.get('id'), str):
            return found['id']
    return None


def run(args: argparse.Namespace) -> None:
    if args.company_a == args.company_b:
        raise CheckFailed('Use two distinct companies')
    base = args.base.rstrip('/') + '/api/v1'
    tokens: dict[tuple[str, str], str] = {}
    feeds: dict[tuple[str, str], dict] = {}
    events: dict[str, tuple[str, str]] = {}
    companies = (args.company_a, args.company_b)

    for company in companies:
        for role in ('cfo', 'ceo'):
            code, result = request(base, 'POST', '/demo-session',
                                   payload={'company_id': company, 'role': role})
            expect(code, 201, f'{company}/{role}: session')
            if result.get('company_id') != company or result.get('role') != role or not result.get('token'):
                raise CheckFailed(f'{company}/{role}: invalid session scope')
            token = result['token']
            tokens[company, role] = token
            code, profile = request(base, 'GET', f'/companies/{company}/profile', token=token)
            expect(code, 200, f'{company}/{role}: profile')
            if profile.get('id') != company or profile.get('role') != role:
                raise CheckFailed(f'{company}/{role}: profile role/company mismatch')
            code, feed = request(base, 'GET', f'/companies/{company}/feed', token=token)
            expect(code, 200, f'{company}/{role}: feed')
            if not isinstance(feed.get('items'), list):
                raise CheckFailed(f'{company}/{role}: feed items not a list')
            feeds[company, role] = feed
            other = args.company_b if company == args.company_a else args.company_a
            for resource in ('profile', 'feed'):
                path = f'/companies/{other}/' + resource
                code, _ = request(base, 'GET', path, token=token)
                expect(code, 403, f'{company}/{role}: other-company {resource} denied')

    for company in companies:
        token = tokens[company, 'cfo']
        code, page = request(base, 'GET', f'/companies/{company}/events?limit=1&offset=0', token=token)
        expect(code, 200, f'{company}: event listing')
        items = page.get('items')
        if not isinstance(items, list):
            raise CheckFailed(f'{company}: event items not a list')
        if items:
            event_id = items[0]['id']
            code, detail = request(base, 'GET', '/events/' + event_id, token=token)
            expect(code, 200, f'{company}: own event')
            if not detail.get('source_ids'):
                raise CheckFailed(f'{company}: event has no source')
            events[company] = event_id, detail['source_ids'][0]
        elif args.seed_missing:
            events[company] = seeded_event(base, token, company)
        else:
            print(f'SKIP {company}: no existing event; use --seed-missing for full ACL coverage', flush=True)

    for company, (event_id, source_id) in events.items():
        other = args.company_b if company == args.company_a else args.company_a
        for role in ('cfo', 'ceo'):
            own_token = tokens[company, role]
            foreign_token = tokens[other, role]
            for resource, ident in (('events', event_id), ('sources', source_id)):
                path = f'/{resource}/{quote(ident, safe="")}'
                code, _ = request(base, 'GET', path, token=own_token)
                expect(code, 200, f'{company}/{role}: own {resource} readable')
                code, _ = request(base, 'GET', path, token=foreign_token)
                expect(code, 404, f'{other}/{role}: foreign {resource} denied')

    insight_id = args.insight_id or choose_saved_insight(feeds[args.company_a, 'cfo'])
    if not insight_id:
        if args.require_insight or args.answer_id:
            raise CheckFailed('No saved CFO insight available; provide --insight-id')
        print('SKIP saved insight/answer ACL: no CFO insight found (zero LLM calls)', flush=True)
    else:
        owner = tokens[args.company_a, 'cfo']
        code, insight = request(base, 'GET', f'/insights/{quote(insight_id, safe="")}', token=owner)
        expect(code, 200, 'Saved CFO insight readable')
        if insight.get('company_id') != args.company_a or insight.get('role') != 'cfo':
            raise CheckFailed('Saved insight is not owned by first company CFO')
        for company, role in ((args.company_a, 'ceo'), (args.company_b, 'cfo'), (args.company_b, 'ceo')):
            code, _ = request(base, 'GET', f'/insights/{quote(insight_id, safe="")}', token=tokens[company, role])
            expect(code, 404, f'{company}/{role}: CFO insight denied')
        evidence_id = next((e for claim in insight.get('claims', [])
                            for e in claim.get('evidence_ids', [])), None)
        if evidence_id:
            path = f'/evidence/{quote(evidence_id, safe="")}?insight_id={quote(insight_id, safe="")}'
            code, _ = request(base, 'GET', path, token=owner)
            expect(code, 200, 'CFO claim evidence readable')
            for company, role in ((args.company_a, 'ceo'), (args.company_b, 'cfo')):
                code, _ = request(base, 'GET', path, token=tokens[company, role])
                expect(code, 404, f'{company}/{role}: CFO evidence denied')
        if insight.get('run_id'):
            path = '/runs/' + quote(insight['run_id'], safe='')
            code, _ = request(base, 'GET', path, token=owner)
            expect(code, 200, 'CFO agent run readable')
            code, _ = request(base, 'GET', path, token=tokens[args.company_a, 'ceo'])
            expect(code, 404, 'Other role agent run denied')
        if args.answer_id:
            path = '/answers/' + quote(args.answer_id, safe='')
            code, answer = request(base, 'GET', path, token=owner)
            expect(code, 200, 'Saved CFO follow-up answer readable')
            if answer.get('insight_id') != insight_id:
                raise CheckFailed('Saved answer does not belong to selected insight')
            for company, role in ((args.company_a, 'ceo'), (args.company_b, 'cfo')):
                code, _ = request(base, 'GET', path, token=tokens[company, role])
                expect(code, 404, f'{company}/{role}: saved answer denied')

    print('DONE: four personas verified; no billable model requests', flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', default='http://127.0.0.1:8000')
    p.add_argument('--basic-user', help='HTTPS site login; password is prompted or read from ALPHICH_SMOKE_PASSWORD')
    p.add_argument('--company-a', default='DEMO_A')
    p.add_argument('--company-b', default='DEMO_B')
    p.add_argument('--seed-missing', action='store_true', help='Write ONLY synthetic publications where no events exist')
    p.add_argument('--insight-id', help='Existing CFO card ID, strictly read-only')
    p.add_argument('--answer-id', help='Existing CFO answer ID, strictly read-only')
    p.add_argument('--require-insight', action='store_true', help='Fail if no existing CFO card is available')
    args = p.parse_args()
    try:
        if args.basic_user:
            parsed = urlsplit(args.base)
            if parsed.scheme != 'https' or not parsed.hostname or parsed.username is not None:
                raise CheckFailed('Site Basic Auth requires an HTTPS base URL without credentials')
            os.environ['ALPHICH_SMOKE_USER'] = args.basic_user
            if not os.getenv('ALPHICH_SMOKE_PASSWORD'):
                os.environ['ALPHICH_SMOKE_PASSWORD'] = getpass.getpass('Site password: ')
        run(args)
    except (CheckFailed, KeyError, ValueError, TypeError) as exc:
        print(f'FAIL {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
