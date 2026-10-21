"""Mocked, unpaid company/role and saved-insight ACL smoke checks."""
import importlib.util
import io
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / 'verify-personas.py'
spec = importlib.util.spec_from_file_location('alphich_verify_personas', MODULE)
personas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(personas)


class PersonaTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.alter_role_acl = False
        self.has_saved_card = True
        self.a_has_event = True
        self.b_has_event = True

    def args(self, **changes):
        data = dict(base='http://local:8000', company_a='DEMO_A', company_b='DEMO_B',
                    insight_id=None, answer_id=None, require_insight=False, seed_missing=False)
        data.update(changes)
        return types.SimpleNamespace(**data)

    def response(self, base, method, path, *, token=None, payload=None, key=None):
        self.calls.append((method, path))
        if path == '/demo-session':
            company, role = payload['company_id'], payload['role']
            return 201, {'company_id': company, 'role': role, 'token': f'{company}:{role}'}
        company, role = token.split(':')
        if path.startswith('/companies/') and path.endswith('/profile'):
            wanted = path.split('/')[2]
            if wanted != company:
                return 403, {}
            return 200, {'id': company, 'role': role}
        if path.startswith('/companies/') and path.endswith('/feed'):
            wanted = path.split('/')[2]
            if wanted != company:
                return 403, {}
            result = []
            if self.has_saved_card and company == 'DEMO_A' and role == 'cfo':
                result = [{'event_id': 'event_A', 'insight': {'id': 'insight_A'}}]
            return 200, {'items': result}
        if path.startswith('/companies/') and '/events?' in path:
            wanted = path.split('/')[2]
            if wanted != company:
                return 403, {}
            found = self.a_has_event if company == 'DEMO_A' else self.b_has_event
            return 200, {'items': [{'id': 'event_A' if company == 'DEMO_A' else 'event_B'}] if found else []}
        if path in ('/events/event_A', '/events/event_B'):
            owned = 'DEMO_A' if path == '/events/event_A' else 'DEMO_B'
            if company != owned:
                return 404, {}
            return 200, {'id': path.rsplit('/', 1)[-1], 'source_ids': ['source_A' if owned == 'DEMO_A' else 'source_B']}
        if path in ('/sources/source_A', '/sources/source_B'):
            owned = 'DEMO_A' if path == '/sources/source_A' else 'DEMO_B'
            return (200, {}) if owned == company else (404, {})
        if path == '/insights/insight_A':
            if (company, role) == ('DEMO_A', 'cfo') or (self.alter_role_acl and company == 'DEMO_A'):
                return 200, {'id': 'insight_A', 'company_id': 'DEMO_A', 'role': 'cfo',
                             'claims': [{'evidence_ids': ['claim_A']}], 'run_id': 'run_A'}
            return 404, {}
        if path == '/evidence/claim_A?insight_id=insight_A':
            return (200, {}) if (company, role) == ('DEMO_A', 'cfo') else (404, {})
        if path == '/runs/run_A':
            return (200, {}) if (company, role) == ('DEMO_A', 'cfo') else (404, {})
        if path == '/answers/answer_A':
            return (200, {'insight_id': 'insight_A'}) if (company, role) == ('DEMO_A', 'cfo') else (404, {})
        if path == '/publications':
            return 202, {'id': 'job_seed'}
        if path == '/jobs/job_seed':
            return 200, {'status': 'succeeded', 'result_id': 'event_B'}
        raise AssertionError(f'Unexpected {method} {path}')

    def test_matrix_includes_four_roles_and_acl_with_no_paid_posts(self):
        with patch.object(personas, 'request', side_effect=self.response), redirect_stdout(io.StringIO()) as output:
            personas.run(self.args(answer_id='answer_A', insight_id='insight_A'))
        self.assertEqual(4, sum(1 for method, path in self.calls if path == '/demo-session'))
        self.assertNotIn('/companies/DEMO_A/insights', str(self.calls))
        self.assertFalse(any(method == 'POST' and path not in ('/demo-session', '/publications') for method, path in self.calls))
        self.assertIn('saved answer denied', output.getvalue())
        self.assertIn('four personas verified', output.getvalue())

    def test_wrong_role_access_fails(self):
        self.alter_role_acl = True
        with patch.object(personas, 'request', side_effect=self.response), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(personas.CheckFailed, 'expected HTTP 404, got 200'):
                personas.run(self.args(require_insight=True))

    def test_absent_card_is_skipped_unless_required(self):
        self.has_saved_card = False
        with patch.object(personas, 'request', side_effect=self.response), redirect_stdout(io.StringIO()) as output:
            personas.run(self.args())
        self.assertIn('SKIP saved insight/answer ACL', output.getvalue())
        with patch.object(personas, 'request', side_effect=self.response), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(personas.CheckFailed, 'No saved CFO insight'):
                personas.run(self.args(require_insight=True))

    def test_can_seed_missing_company_with_synthetic_publication(self):
        self.b_has_event = False
        with patch.object(personas, 'request', side_effect=self.response), redirect_stdout(io.StringIO()):
            personas.run(self.args(seed_missing=True))
        self.assertIn(('POST', '/publications'), self.calls)
        self.assertEqual(sum(1 for method, path in self.calls if path == '/publications'), 1)

    def test_paid_or_mutating_endpoints_are_blocked_by_request_allowlist(self):
        for url in ('/companies/DEMO_A/insights', '/insights/insight_A/questions'):
            with self.assertRaisesRegex(personas.CheckFailed, 'Forbidden POST'):
                personas.request('http://local/api/v1', 'POST', url, payload={'event_id': 'x'})

    def test_same_company_ids_are_invalid(self):
        with self.assertRaisesRegex(personas.CheckFailed, 'distinct'):
            personas.run(self.args(company_b='DEMO_A'))


if __name__ == '__main__':
    unittest.main()
