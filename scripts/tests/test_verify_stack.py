"""Safe offline contract tests for the integration smoke runner."""
from __future__ import annotations

import importlib.util
import io
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / 'verify-stack.py'
spec = importlib.util.spec_from_file_location('alphich_verify_stack', MODULE)
smoke = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(smoke)


class Response:
    status = 200
    def __enter__(self): return self
    def __exit__(self, *args): return None


class SmokeTests(unittest.TestCase):
    def args(self, **extra):
        arguments = dict(start=False, import_fixtures=False, company='DEMO_A',
                         base='http://localhost:8000', frontend='http://localhost:5173',
                         timeout=2, analysis_timeout=2, live_analysis=False, ask_question=False)
        arguments.update(extra)
        return types.SimpleNamespace(**arguments)

    def fake_request(self, base, method, path, *, token=None, payload=None, key=None, timeout=15):
        self.calls.append((method, path))
        if path == '/demo-session':
            return (201, {'token': 'secret_token_a'}) if payload['company_id'] == 'DEMO_A' else (403, {})
        if path == '/companies/DEMO_A/profile': return 200, {'id': 'DEMO_A'}
        if path == '/publications': return 202, {'id': 'job_test', 'status': 'queued'}
        if path == '/jobs/job_test': return 200, {'id': 'job_test', 'status': 'succeeded', 'result_id': 'event_test'}
        if path == '/events/event_test': return 200, {'id': 'event_test', 'company_id': 'DEMO_A',
                'title': payload['title'] if payload else self.last_title, 'source_ids': ['src_test']}
        if path == '/sources/src_test':
            return 200, {'text': self.last_text, 'url_verified': False}
        if path == '/companies/DEMO_A/events?limit=20&offset=0':
            return 200, {'items': [{'id': 'event_test'}]}
        if path == '/jobs?limit=20': return 200, {'items': [{'id': 'job_test'}]}
        if path == '/llm/status': return 200, self.llm_status
        if path == '/companies/DEMO_A/feed': return 200, {'items': []}
        # Scripted responses exercise the live-mode control flow WITHOUT
        # contacting a model or charging the shared team budget.
        if path == '/companies/DEMO_A/insights' and method == 'POST':
            return 202, {'id': 'job_insight', 'status': 'queued'}
        if path == '/jobs/job_insight':
            return 200, {'id': 'job_insight', 'status': 'succeeded', 'result_id': 'insight_test'}
        if path == '/insights/insight_test':
            return 200, {'id': 'insight_test', 'company_id': 'DEMO_A',
                         'event_id': 'event_test', 'status': 'ready_with_conditions',
                         'run_id': 'run_test',
                         'claims': [{'evidence_ids': ['ev_test']}]}
        if path == '/evidence/ev_test?insight_id=insight_test':
            return 200, {'id': 'ev_test', 'text': 'Synthetic evidence'}
        if path == '/runs/run_test': return 200, {'status': 'succeeded', 'steps': []}
        if path == '/insights/insight_test/questions' and method == 'POST':
            return 202, {'id': 'job_question', 'status': 'queued'}
        if path == '/jobs/job_question':
            return 200, {'id': 'job_question', 'status': 'succeeded', 'result_id': 'answer_test'}
        if path == '/answers/answer_test':
            return 200, {'id': 'answer_test', 'insight_id': 'insight_test', 'claims': []}
        raise AssertionError(f'Unexpected API call: {method} {path}')

    def setUp(self):
        self.calls = []
        self.llm_status = {'configured': False, 'prices_confirmed': False,
                           'budget_mode': 'token_prices', 'bank_index_present': False,
                           'spent_rub': '0'}
        self.last_title = 'recorded in create request'
        self.last_text = 'recorded in create request'
        original = self.fake_request
        def record(*args, **kwargs):
            if args[2] == '/publications':
                self.last_title = kwargs['payload']['title']
                self.last_text = kwargs['payload']['text']
            return original(*args, **kwargs)
        self.record = record

    def test_default_never_sends_paid_requests(self):
        with (patch.object(smoke, 'compose', return_value='postgres\napi\nworker\nanalysis-worker\nfrontend'),
              patch.object(smoke, 'wait_api'),
              patch.object(smoke, 'urlopen', return_value=Response()),
              patch.object(smoke, 'request', side_effect=self.record),
              redirect_stdout(io.StringIO()) as output):
            smoke.run(self.args())
        self.assertNotIn('/insights', str(self.calls))
        self.assertTrue(any(path == '/llm/status' for _, path in self.calls))
        self.assertIn('SKIP paid analysis', output.getvalue())

    def test_explicit_live_requires_configured_gateway(self):
        with (patch.object(smoke, 'compose', return_value='postgres\napi\nworker\nanalysis-worker\nfrontend'),
              patch.object(smoke, 'wait_api'),
              patch.object(smoke, 'urlopen', return_value=Response()),
              patch.object(smoke, 'request', side_effect=self.record),
              redirect_stdout(io.StringIO())):
            with self.assertRaisesRegex(smoke.CheckFailed, 'not configured'):
                smoke.run(self.args(live_analysis=True))
        self.assertFalse(any('/insights' in path for _, path in self.calls))

    def test_gateway_receipt_ready_without_confirmed_token_prices(self):
        """A receipt-configured server may correctly report prices_confirmed=false."""
        self.llm_status.update(configured=True, prices_confirmed=False,
                               budget_mode='gateway_receipt', bank_index_present=True)
        with (patch.object(smoke, 'compose', return_value='postgres\napi\nworker\nanalysis-worker\nfrontend'),
              patch.object(smoke, 'wait_api'),
              patch.object(smoke, 'urlopen', return_value=Response()),
              patch.object(smoke, 'request', side_effect=self.record),
              redirect_stdout(io.StringIO())):
            smoke.run(self.args(live_analysis=True, ask_question=True))
        self.assertIn(('POST', '/companies/DEMO_A/insights'), self.calls)
        self.assertIn(('GET', '/evidence/ev_test?insight_id=insight_test'), self.calls)
        self.assertIn(('GET', '/runs/run_test'), self.calls)
        self.assertIn(('POST', '/insights/insight_test/questions'), self.calls)
        self.assertIn(('GET', '/answers/answer_test'), self.calls)

    def test_ready_requires_server_configured_not_legacy_price_flag(self):
        self.llm_status.update(configured=False, prices_confirmed=True,
                               bank_index_present=True)
        with (patch.object(smoke, 'compose', return_value='postgres\napi\nworker\nanalysis-worker\nfrontend'),
              patch.object(smoke, 'wait_api'),
              patch.object(smoke, 'urlopen', return_value=Response()),
              patch.object(smoke, 'request', side_effect=self.record),
              redirect_stdout(io.StringIO())):
            with self.assertRaisesRegex(smoke.CheckFailed, 'not configured'):
                smoke.run(self.args(live_analysis=True))
        self.assertNotIn(('POST', '/companies/DEMO_A/insights'), self.calls)

    def test_live_requires_bank_index_even_if_server_ready(self):
        self.llm_status.update(configured=True, prices_confirmed=False,
                               budget_mode='gateway_receipt', bank_index_present=False)
        with (patch.object(smoke, 'compose', return_value='postgres\napi\nworker\nanalysis-worker\nfrontend'),
              patch.object(smoke, 'wait_api'),
              patch.object(smoke, 'urlopen', return_value=Response()),
              patch.object(smoke, 'request', side_effect=self.record),
              redirect_stdout(io.StringIO())):
            with self.assertRaisesRegex(smoke.CheckFailed, 'Bank index is missing'):
                smoke.run(self.args(live_analysis=True))
        self.assertNotIn(('POST', '/companies/DEMO_A/insights'), self.calls)

    def test_missing_containers_are_fatal(self):
        with patch.object(smoke, 'compose', return_value='api\npostgres'):
            with self.assertRaisesRegex(smoke.CheckFailed, 'Missing running containers'):
                smoke.run(self.args())


if __name__ == '__main__':
    unittest.main()
