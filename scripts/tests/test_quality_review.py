"""Run: python3 -m unittest discover -s scripts/tests -p test_quality_review.py -v"""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quality_review as qr


def sample_files(directory):
    for version, n in (("v2",2),("v3",1)):
        sources=[]; labels=[]
        for i in range(n):
            event=f"event_{version}_{i}"
            sources.append({"event_id":event,"title":f"News {i} version {version}","url":f"https://example.org/{event}",
                "grades":{"COMP_A":5,"COMP_B":0},"rationale":{"COMP_A":"AI secret conclusion"}})
            for company in ("COMP_A","COMP_B"):
                labels.append({"event_id":event,"company_id":company,"grade":5 if company=="COMP_A" else 0,
                    "rationale":"DO_NOT_LEAK_AI_RATIONALE", "assessment":{"business_channel":"DO_NOT_LEAK"}})
        (directory / f"real_news_selection_labels.{version}.json").write_text(json.dumps({"version":version,"sources":sources,"labels":labels}),encoding="utf-8")


class LocalReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.folder=Path(self.tmp.name)
        self.fixtures=self.folder/"fixtures";self.fixtures.mkdir()
        sample_files(self.fixtures)
        self.db=self.folder/"private.sqlite3"
        self.created=qr.initialize(self.db,self.fixtures,["analystA","analystB"],"arbitrator")
        self.tokens=self.created["users"]
        class Handler(qr.ReviewHandler):pass
        Handler.db_path=self.db
        self.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.root="http://127.0.0.1:"+str(self.server.server_address[1])
        self.review={"id":1,"grade":4,"justification":"Существенное событие в отрасли при условиях",
          "geography":"national","scale":"material","applicability":"conditional",
          "geographic_limit":"Неизвестно место операций","scale_limit":"Масштаб для компании неизвестен",
          "applicability_limit":"Нужно подтвердить договоры"}

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=4)
        self.tmp.cleanup()

    def call(self,path,token=None,body=None):
        headers={}
        if token:headers['X-Review-Token']=token
        if body is not None:headers['Content-Type']='application/json'
        request=Request(self.root+path,data=json.dumps(body).encode() if body is not None else None,headers=headers,
                        method='POST' if body is not None else 'GET')
        try:
            with urlopen(request,timeout=4) as r:return r.status,r.read().decode()
        except HTTPError as e:return e.code,e.read().decode()

    def test_counts_blind_and_unauthorized(self):
        self.assertEqual(self.created['pairs'],6)
        self.assertEqual(self.db.stat().st_mode&0o777,0o600)
        self.assertEqual(self.call('/api/cases')[0],401)
        status,body=self.call('/api/cases',self.tokens['analystA'])
        self.assertEqual(status,200)
        self.assertEqual(json.loads(body)['total'],6)
        self.assertNotIn('DO_NOT_LEAK',body)
        self.assertNotIn('grades',body)
        status,body=self.call('/api/case?id=1',self.tokens['analystA'])
        self.assertEqual(status,200)
        self.assertNotIn('DO_NOT_LEAK',body)
        self.assertNotIn('prediction',body)
        self.assertIsNone(json.loads(body)['article_text'])
        self.assertIsNone(json.loads(body)['company_context'])

    def test_review_revisions_second_reader_and_arbiter(self):
        a,b,j=self.tokens['analystA'],self.tokens['analystB'],self.tokens['arbitrator']
        self.assertEqual(self.call('/api/adjudicate',a,self.review)[0],403)
        self.assertEqual(self.call('/api/adjudicate',j,self.review)[0],400)
        self.assertEqual(self.call('/api/review',a,self.review)[0],200)
        self.assertEqual(self.call('/api/review',a,{**self.review,"grade":3})[0],200)
        view=json.loads(self.call('/api/case?id=1',b)[1]);self.assertNotIn('reviews',view)
        self.assertIsNone(view.get('own'))
        self.assertEqual(self.call('/api/review',b,{**self.review,"grade":2})[0],200)
        view=json.loads(self.call('/api/case?id=1',j)[1]);self.assertEqual(view['state'],'disagreement')
        self.assertEqual([r['grade'] for r in view['reviews']],[3,2])
        self.assertEqual(self.call('/api/adjudicate',j,{**self.review,"grade":2})[0],200)
        self.assertEqual(self.call('/api/review',b,self.review)[0],400)
        output=self.folder/'snapshot.json'
        stats=qr.export_reviews(self.db,output,True)
        self.assertEqual(stats['final'],1)
        self.assertEqual(stats['awaiting_reviews'],5)
        parsed=json.loads(output.read_text())
        record=next(r for r in parsed['items'] if r['event_id']=='event_v2_0' and r['company_id']=='COMP_A')
        self.assertEqual(record['gold']['grade'],2)
        self.assertEqual(record['original_model_annotation_not_gold']['grade'],5)
        self.assertEqual(len(parsed['history']['reviews']),3)
        self.assertEqual(len(parsed['history']['adjudications']),1)
        self.assertEqual(output.stat().st_mode&0o777,0o600)
        with self.assertRaises(ValueError):qr.export_reviews(self.db,output)

    def test_agreed_requires_matching_dimensions(self):
        a,b,j=self.tokens['analystA'],self.tokens['analystB'],self.tokens['arbitrator']
        self.call('/api/review',a,self.review)
        self.call('/api/review',b,self.review)
        arb=json.loads(self.call('/api/case?id=1',j)[1]);self.assertEqual(arb['state'],'agreed')
        exported=self.folder/'snapshot-agreed.json'
        stats=qr.export_reviews(self.db,exported)
        self.assertEqual(stats['final'],1)
        self.assertNotIn('original_model_annotation_not_gold',exported.read_text())

    def test_wrong_input_and_context_guard(self):
        a=self.tokens['analystA'];r={**self.review,'grade':6}
        self.assertEqual(self.call('/api/review',a,r)[0],400)
        self.assertEqual(self.call('/api/review',a,{**self.review,'justification': 'x'})[0],400)
        self.assertEqual(self.call('/api/review',a,self.review)[0],200)
        context=self.folder/'context.json'
        context.write_text(json.dumps({'COMP_A':{'verified_at':'2026-10-21', 'features':[{'name':'okved','value':'42.99','source_ref':'registry'}]}}))
        with qr.managed(self.db) as db:
            with self.assertRaises(ValueError):qr.enrich_data(db,None,context)

    def test_enrich_requires_evidence_and_blindness(self):
        source=self.folder/'articles.json';source.write_text(json.dumps({'event_v2_0':{'text':'Real article text','published_at':'2026-10-21'}}))
        context=self.folder/'context.json';context.write_text(json.dumps({'COMP_A':{'verified_at':'2026-10-21','features':[{'name':'okved','value':'42.99','source_ref':'registry'}]}}))
        with qr.managed(self.db) as db:
            updated=qr.enrich_data(db,source,context)
        self.assertEqual(updated,(2,3))
        view=json.loads(self.call('/api/case?id=1',self.tokens['analystA'])[1])
        # first pair ordering may belong to COMP_A or COMP_B; article for v2_0 always filled
        self.assertFalse(view['article_text_missing'])
        self.assertEqual(view['published_at'],'2026-10-21')
        self.assertNotIn('DO_NOT_LEAK_AI_RATIONALE',json.dumps(view))


if __name__=='__main__':unittest.main()
