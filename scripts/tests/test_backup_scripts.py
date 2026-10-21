"""Safe static and negative tests for deployment backup entry points (no Docker needed)."""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKUP = ROOT / 'scripts' / 'backup-postgres.sh'
VERIFY = ROOT / 'scripts' / 'verify-postgres-backup.sh'


class BackupScriptsTest(unittest.TestCase):
    def test_bash_syntax(self):
        for script in (BACKUP, VERIFY):
            with self.subTest(script=script):
                subprocess.run(['bash', '-n', str(script)], check=True)

    def test_restore_rejects_missing_file_without_docker(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(['bash', str(VERIFY), str(Path(directory) / 'missing.dump')],
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('missing or empty', result.stderr)

    def test_restore_rejects_mismatched_checksum_before_docker(self):
        with tempfile.TemporaryDirectory() as directory:
            dump = Path(directory) / 'test.dump'
            dump.write_bytes(b'not a real database')
            Path(str(dump) + '.sha256').write_text('0' * 64 + '  test.dump\n')
            result = subprocess.run(['bash', str(VERIFY), str(dump)], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('SHA256 mismatch', result.stderr)

    def test_backup_requires_docker_without_creating_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, ALPHICH_BACKUP_DIR=str(Path(directory) / 'out'), PATH=directory)
            # bash executable is an absolute path: PATH can be isolated from Docker.
            result = subprocess.run(['/bin/bash', str(BACKUP)], env=env, text=True,
                                    capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Docker is required', result.stderr)
            self.assertFalse((Path(directory) / 'out').exists())


    def test_backup_creates_private_archive_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_docker = Path(directory) / 'docker'
            fake_docker.write_text('''#!/bin/sh
if [ "$1" = "compose" ]; then printf 'FAKE_POSTGRES_DUMP\\n'; exit 0; fi
if [ "$1" = "run" ]; then exit 0; fi
exit 1
''')
            fake_docker.chmod(0o755)
            backup_dir = Path(directory) / 'backups'
            env = dict(os.environ, ALPHICH_BACKUP_DIR=str(backup_dir),
                       PATH=directory + os.pathsep + os.environ['PATH'])
            result = subprocess.run(['bash', str(BACKUP)], env=env, text=True,
                                    capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            files = list(backup_dir.glob('*.dump'))
            self.assertEqual(len(files), 1)
            self.assertEqual(len(list(backup_dir.glob('*.sha256'))), 1)
            self.assertEqual(files[0].stat().st_mode & 0o777, 0o600)
            self.assertEqual(backup_dir.stat().st_mode & 0o777, 0o700)
            self.assertIn('BACKUP_OK:', result.stdout)

    def test_test_restore_reports_saved_insight_on_fake_docker(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            fake_docker = Path(directory) / 'docker'
            fake_docker.write_text('''#!/bin/sh
case "$1" in
  run) printf 'fake-container\\n'; exit 0 ;;
  exec)
    case "$3" in
      psql) printf '2,5,5,1,1\\n' ;;
    esac
    exit 0 ;;
  cp|rm) exit 0 ;;
esac
exit 1
''')
            fake_docker.chmod(0o755)
            dump = Path(directory) / 'snapshot.dump'
            dump.write_bytes(b'a valid archive is assumed in the mocked docker test')
            digest = hashlib.sha256(dump.read_bytes()).hexdigest()
            Path(str(dump) + '.sha256').write_text(digest + '  snapshot.dump\n')
            env = dict(os.environ, PATH=directory + os.pathsep + os.environ['PATH'])
            result = subprocess.run(['bash', str(VERIFY), str(dump), '--require-insight'],
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('insights=1', result.stdout)
            self.assertIn('RESTORE_CHECK_OK', result.stdout)

if __name__ == '__main__':
    unittest.main()
