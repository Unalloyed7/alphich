"""The production overlay must preserve secrets, proxy routing and the shared cap."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.intelligence.settings import Settings

ROOT = Path(__file__).resolve().parents[2]


class CloudruAIConfigTests(unittest.TestCase):
    def test_demo_overlay_repairs_stale_prices_without_changing_budget_or_proxy(self):
        if not shutil.which('docker'):
            self.skipTest('Docker Compose CLI is not installed')
        version = subprocess.run(['docker', 'compose', 'version'], capture_output=True)
        if version.returncode:
            self.skipTest('Docker Compose plugin is not installed')
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / 'fixture.env'
            env_file.write_text('LLM_API_KEY=test-only-key\nLLM_DAILY_LIMIT_RUB=10\n'
                                'LLM_BUDGET_MODE=token_prices\nLLM_PRICES_CONFIRMED=0\n'
                                'LLM_MODEL_CATALOG_JSON=[]\n')
            vpn = Path(directory) / 'vpn.yaml'
            vpn.write_text('services:\n  api:\n    environment:\n'
                           '      HTTPS_PROXY: http://test-proxy:1080\n'
                           '      LLM_BUDGET_MODE: token_prices\n'
                           '  analysis-worker:\n    environment:\n'
                           '      HTTPS_PROXY: http://test-proxy:1080\n')
            # Do not inherit any operator secrets or LLM configuration into the fixture.
            clean_env = {k: v for k, v in os.environ.items() if not k.startswith(('LLM_', 'AI_AUTOMATION_'))}
            result = subprocess.run(
                ['docker', 'compose', '--env-file', str(env_file), '-f', str(ROOT / 'compose.yaml'),
                 '-f', str(vpn), '-f', str(ROOT / 'compose.cloudru-ai.yaml'), 'config', '--format', 'json'],
                capture_output=True, text=True, env=clean_env, check=True)
            config = json.loads(result.stdout)
        for service in ('api', 'analysis-worker'):
            with self.subTest(service=service):
                env = config['services'][service]['environment']
                self.assertEqual(env['LLM_API_KEY'], 'test-only-key')
                self.assertEqual(env['HTTPS_PROXY'], 'http://test-proxy:1080')
                self.assertEqual(env['LLM_DAILY_LIMIT_RUB'], '10')
                self.assertEqual(env['AI_AUTOMATION_ENABLED'], '0')
                self.assertEqual(env['LLM_PRICES_CONFIRMED'], '0')
                with patch.dict(os.environ, env, clear=True):
                    settings = Settings.from_env()
                settings.require_ready()
                self.assertEqual(settings.budget_mode, 'gateway_receipt')
                self.assertEqual(settings.daily_limit_micros, 10_000_000)


if __name__ == '__main__':
    unittest.main()
