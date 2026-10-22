"""Offline tests for the RSS->Quality Lab staging/import workflow."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quality_corpus_pipeline as p
import quality_review as qr


class CorpusPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fixtures = self.root / 'fixtures'
        self.fixtures.mkdir()
        self.ids = [f'event_{i}' for i in range(3)]
        self.urls = [f'https://www.cbr.ru/press/pr/?file={i}.htm' for i in range(3)]
        # Same structure as real v2/v3, no backend required.
        for ds in ('v2', 'v3'):
            sources = [{'event_id': eid, 'title': f'Title for {eid}', 'url': url,
                        'grades': {'COMP_A': 5}, 'rationale': {'COMP_A': 'LEAKED_AI_TEACHER'}}
                       for eid, url in zip(self.ids, self.urls)]
            labels = [{'event_id': eid, 'company_id': company, 'grade': (5 if company == 'COMP_A' else 0),
                       'rationale': 'LEAKED_AI_TEACHER', 'assessment': {'answer': 'SECRET'}}
                      for eid in self.ids for company in ('COMP_A', 'COMP_B')]
            (self.fixtures/f'real_news_selection_labels.{ds}.json').write_text(
                json.dumps({'version': ds, 'sources': sources, 'labels': labels}))
        self.db = self.root / 'reviews.sqlite3'
        self.tokens = qr.initialize(self.db, self.fixtures, ['expertA','expertB'], 'arbiter')['users']
        self.text = 'News about key rate and verified bank press release. ' * 3
        self.rss = self.root / 'rss-quality-matches.json'
        self.articles = [self.make_article(i) for i in range(3)]
        self.save()

    def make_article(self, i):
        return {'id': f'{i+1:064x}', 'url': self.urls[i], 'final_url': self.urls[i],
                'title': f'Title for {self.ids[i]}', 'body': self.text + str(i),
                'published_at': '2026-10-08', 'fetched_at': '2026-10-22T01:00:00Z',
                'content_hash': 'any-rss-hash'}

    def save(self, articles=None):
        self.rss.write_text(json.dumps({'matches': [['v3', e, a['id']] for e, a in zip(self.ids, self.articles)],
                                        'articles': articles if articles is not None else self.articles}))

    def stage(self, run='first'):
        folder = self.root / run
        result = p.prepare(snapshot=self.rss, db_path=self.db, fixtures=self.fixtures, output_dir=folder)
        return folder, result

    def test_prepare_reads_prior_export_and_never_modifies_database(self):
        before = self.db.read_bytes()
        folder, result = self.stage()
        self.assertEqual(result['counts']['new'], 6)  # 3 events x 2 fixture versions
        self.assertEqual(result['new_pair_texts'], 12)  # x 2 company pairs
        self.assertEqual(before, self.db.read_bytes())
        self.assertFalse((folder/'approved_articles.json').exists())
        self.assertTrue((folder/'unmatched_rss_articles.jsonl').exists())
        draft = (folder/'candidate_articles.json').read_text()
        self.assertNotIn('LEAKED_AI_TEACHER', draft)
        self.assertNotIn('LEAKED_AI_TEACHER', (folder/'manifest.json').read_text())
        self.assertIn(self.ids[0], draft)
        self.assertEqual(len(json.loads(draft)), 3)

    def test_approve_explicit_one_and_never_overwrite(self):
        folder, _ = self.stage()
        ans = p.approve(run_dir=folder, db_path=self.db, fixtures=self.fixtures, ids=[self.ids[0]])
        self.assertEqual(ans['updated_pairs'], 4)  # v2/v3 x 2 companies
        with qr.managed(self.db) as db:
            values = db.execute('SELECT DISTINCT article_text FROM pairs WHERE event_id=?',
                                (self.ids[0],)).fetchall()
            self.assertEqual([v[0] for v in values], [self.articles[0]['body']])
            self.assertEqual(db.execute('SELECT COUNT(*) FROM pairs WHERE article_text IS NOT NULL').fetchone()[0], 4)
        with self.assertRaisesRegex(ValueError, 'Existing text'):
            p.approve(run_dir=folder, db_path=self.db, fixtures=self.fixtures, ids=[self.ids[0]])

    def test_previous_manual_text_preserved_and_conflicts_reported(self):
        with qr.managed(self.db) as db:
            db.execute('UPDATE pairs SET article_text=? WHERE event_id=?', ('Previously verified text', self.ids[0]))
            db.execute('UPDATE pairs SET article_text=? WHERE event_id=?', (self.articles[1]['body'], self.ids[1]))
        folder, result = self.stage()
        self.assertEqual(result['counts']['conflict'], 2)
        self.assertEqual(result['counts']['already_present'], 2)
        self.assertEqual(result['counts']['new'], 2)
        draft = json.loads((folder/'candidate_articles.json').read_text())
        self.assertEqual(list(draft), [self.ids[2]])

    def test_frozen_after_first_human_review(self):
        folder, _ = self.stage()
        with qr.managed(self.db) as db:
            pid = db.execute('SELECT id FROM pairs LIMIT 1').fetchone()[0]
            db.execute('INSERT INTO reviews(pair_id,reviewer,revision,grade,justification,geography,scale,applicability,geographic_limit,scale_limit,applicability_limit,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                       (pid,'expertA',1,3,'Reason long enough','unknown','unknown','unknown','Unknown','Unknown','Unknown',qr.now()))
        with self.assertRaisesRegex(ValueError, 'human reviews'):
            p.approve(run_dir=folder, db_path=self.db, fixtures=self.fixtures, ids=[self.ids[0]])

    def test_detects_tampering_of_prepared_text(self):
        folder, _ = self.stage()
        path = folder/'candidate_articles.json'
        val = json.loads(path.read_text())
        val[self.ids[0]]['text'] += ' tampered'
        path.write_text(json.dumps(val))
        with self.assertRaisesRegex(ValueError, 'changed after prepare'):
            p.approve(run_dir=folder, db_path=self.db, fixtures=self.fixtures, ids=[self.ids[0]])

    def test_fixture_change_blocks_import(self):
        folder, _ = self.stage()
        path = self.fixtures/'real_news_selection_labels.v3.json'
        data = json.loads(path.read_text());data['new'] = 'changed'
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'Fixture changed'):
            p.approve(run_dir=folder, db_path=self.db, fixtures=self.fixtures, ids=[self.ids[0]])

    def test_invalid_urls_or_fake_ip_not_imported(self):
        self.articles[0]['url'] = 'http://localhost/private'
        self.save()
        with self.assertRaisesRegex(ValueError, 'Disallowed RSS URL'):
            self.stage()

    def test_ambiguous_versions_are_not_silently_selected(self):
        self.articles.append({**self.make_article(0), 'id': f'{9:064x}', 'body': 'Another version of article ' * 4})
        self.save()
        folder, result = self.stage()
        self.assertEqual(result['counts']['ambiguous'], 2)
        self.assertEqual(result['counts']['new'], 4)
        self.assertNotIn(self.ids[0], json.loads((folder/'candidate_articles.json').read_text()))

    def test_keeps_unmatched_articles_separate_for_future_corps(self):
        external = {'id': f'{10:064x}', 'url':'https://www.interfax.ru/business/1000000',
                    'final_url':'https://www.interfax.ru/business/1000000',
                    'title':'Real extra article', 'body':'Additional business news ' * 6,
                    'published_at':None}
        self.articles.append(external)
        self.save()
        folder, result = self.stage()
        self.assertEqual(result['unmatched_rss_articles'], 1)
        rows = [json.loads(line) for line in (folder/'unmatched_rss_articles.jsonl').read_text().splitlines()]
        self.assertEqual(rows, [external])
        self.assertNotIn(external['id'], (folder/'candidate_articles.json').read_text())

    def test_html_preview_escapes_raw_articles(self):
        self.articles[0]['body'] = '<script>alert(1)</script>' + ' news content' * 20
        self.save()
        folder, _ = self.stage()
        contents = (folder/'preview.html').read_text()
        self.assertNotIn('<script>alert(1)</script>', contents)
        self.assertIn('&lt;script&gt;', contents)

    def test_jsonl_remote_corpus_supported(self):
        path = self.root/'rss.jsonl'
        path.write_text('\n'.join(json.dumps(item) for item in self.articles))
        folder = self.root/'jsonl_run'
        result = p.prepare(snapshot=path,db_path=self.db,fixtures=self.fixtures,output_dir=folder)
        self.assertEqual(result['counts']['new'], 6)

    def test_manifest_marks_existing_review_lock(self):
        with qr.managed(self.db) as db:
            pid = db.execute('SELECT id FROM pairs LIMIT 1').fetchone()[0]
            db.execute('INSERT INTO reviews(pair_id,reviewer,revision,grade,justification,geography,scale,applicability,geographic_limit,scale_limit,applicability_limit,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                       (pid,'expertA',1,3,'Reason long enough','unknown','unknown','unknown','Unknown','Unknown','Unknown',qr.now()))
        folder, result = self.stage()
        self.assertTrue(result['db_locked'])
        self.assertTrue(json.loads((folder/'manifest.json').read_text())['db_had_human_reviews'])

    def test_private_files_and_no_reuse_of_run_id(self):
        folder, _ = self.stage()
        for f in ('candidate_articles.json', 'preview.html', 'manifest.json'):
            self.assertEqual((folder/f).stat().st_mode & 0o777, 0o600)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.stage()

    def test_reject_duplicate_manual_ids(self):
        folder, _ = self.stage()
        with self.assertRaisesRegex(ValueError, 'distinct'):
            p.approve(run_dir=folder, db_path=self.db, fixtures=self.fixtures,
                      ids=[self.ids[0], self.ids[0]])

    def test_sync_remote_one_command_stages_readonly_data(self):
        raw = ("\n".join(json.dumps(a) for a in self.articles) + "\n").encode('utf-8')
        result = type('Result', (), {'returncode':0, 'stdout':raw, 'stderr':b''})()
        with patch.object(p.subprocess, 'run', return_value=result) as mocked:
            ans = p.sync(ssh_host='89.232.176.172',ssh_user='alphich',
                         ssh_key=self.root/'sshkey',output_dir=self.root/'sync-ok',
                         db_path=self.db,fixtures=self.fixtures)
        self.assertEqual(ans['counts']['new'], 6)
        self.assertTrue((self.root/'sync-ok'/'rss_snapshot.jsonl').exists())
        self.assertTrue((self.root/'sync-ok'/'review'/'preview.html').exists())
        cmd = mocked.call_args[0][0]
        self.assertEqual(cmd[0], 'ssh')
        self.assertIn('BatchMode=yes', cmd)
        self.assertIn('SELECT json_build_object', cmd[-1])
        self.assertIn('sudo -n', cmd[-1])
        self.assertNotIn('POSTGRES_PASSWORD', cmd[-1])

    def test_readonly_remote_query_and_fail_closed_remote(self):
        self.assertIn('FROM rss_articles', p._REMOTE_QUERY)
        self.assertNotIn('DELETE ', p._REMOTE_QUERY.upper())
        with patch.object(p.subprocess, 'run', return_value=type('Result',(),{'returncode':1, 'stdout':b'', 'stderr':b'secret'})()):
            with self.assertRaisesRegex(ValueError, 'Remote read failed') as cm:
                p.sync(ssh_host='89.232.176.172', ssh_user='alphich', ssh_key=self.root/'sshkey',
                       output_dir=self.root/'sync_run', db_path=self.db, fixtures=self.fixtures)
            self.assertNotIn('secret', str(cm.exception))


if __name__ == '__main__':
    unittest.main()
