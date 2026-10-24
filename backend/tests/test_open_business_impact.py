"""Nonfinancial events reach a grounded card; all gateways here are scripted."""
import json

import httpx
import pytest

from app.intelligence.evidence import verify_draft
from app.intelligence.researcher import eligible_claims
from app.intelligence.schemas import CardDraft
from app.intelligence.worker import process_next
from test_intelligence import ScriptedGateway, queued_insight, runtime, store, stored_event


class OpenImpactGateway(ScriptedGateway):
    def __init__(self, topic):
        super().__init__()
        self.topic = topic

    def __call__(self, request):
        response = super().__call__(request)
        title, data = self.seen[-1]
        payload = response.json()
        result = json.loads(payload['choices'][0]['message']['content'])
        if title == 'ExtractionDraft':
            result['facts'][0]['attributes'].update(topics=[self.topic], kind='report',
                business_effect=self.topic, event_status='reported')
        elif title == 'RelevanceJudgment':
            result['reason'] = 'Событие может затронуть безопасность или доверие к строительной организации; конкретная применимость требует проверки.'
        elif title in {'Decision', 'FinalDecision'} and result['decision']['action'] == 'finish':
            draft = result['decision']['draft']
            draft['inferences'][0].update(
                text='Если работы компании затрагивает описанный инцидент, возможна утрата доверия заказчиков и нарушение непрерывности работы.',
                conditions=['Участие компании и затронутые процессы необходимо подтвердить'])
            draft['next_steps'][0].update(
                text='Проверить связь площадок и процессов компании с инцидентом; при подтверждении назначить ответственную функцию и подготовить меры защиты и коммуникацию.',
                conditions=['Инцидент затрагивает процессы компании'])
            draft['missing_data'] = ['Связь компании с описанными площадками и процессами']
        payload['choices'][0]['message']['content'] = json.dumps(result, ensure_ascii=False)
        return httpx.Response(response.status_code, json=payload)


@pytest.mark.parametrize('topic', ['reputation', 'cybersecurity', 'safety', 'accessibility'])
def test_nonfinancial_subject_reaches_the_existing_e5_grade_and_verified_card_pipeline(store, topic):
    client, factory, headers, settings = store
    text = ('В строительной отрасли сообщено об инциденте безопасности и доверия заказчиков. '
            'Обстоятельства расследуются; участие конкретных организаций не установлено.')
    event = stored_event(client, factory, headers['DEMO_A'], text=text,
                         title='Проверка безопасности и доверия в строительной отрасли')
    job = queued_insight(client, headers['DEMO_A'], event).json()
    worker, gateway = runtime(factory, settings, OpenImpactGateway(topic))
    assert process_next(factory, worker)
    result = client.get('/api/v1/jobs/' + job['id'], headers=headers['DEMO_A']).json()
    assert result['status'] == 'succeeded', result
    assert result['result_type'] == 'insight', {'job': result, 'stages': [
        (title, data.get('validation_issues'), data.get('candidates'), len(data.get('news_facts', [])))
        for title, data in gateway.seen]}
    card = client.get('/api/v1/insights/' + result['result_id'], headers=headers['DEMO_A']).json()
    assert card['relevance_grade'] == 4
    assert any('утрата доверия' in claim['text'] for claim in card['claims'] if claim['kind'] == 'model_inference')
    event_data = client.get('/api/v1/events/' + event, headers=headers['DEMO_A']).json()
    assert topic in event_data['facts'][0]['attributes']['topics']
    author = next(data for title, data in gateway.seen if title in {'Decision', 'FinalDecision'})
    assert author['business_focus']['driver'] == 'open'
    assert author['inference_basis']['company_ids'] == author['business_focus']['company_ids']


def bank_claim(cid, topic, day, **extras):
    return {'id': cid, 'kind': 'bank_claim', 'claim_kind': 'interpretation', 'topic': topic,
            'metric': None, 'forecast_horizon': None, 'geography': 'GLOBAL', 'scenario': 'base',
            'attributed_to': 'alfa_bank', 'position_role': 'current_in_document',
            'verification_status': 'visually_verified', 'published_at': day,
            'text': 'Проверенный тезис по теме ' + topic, **extras}


def test_unrelated_qualitative_bank_views_do_not_suppress_each_other_in_handoff_or_final_review():
    older = bank_claim('reputation', 'reputation', '2026-09-01')
    newer = bank_claim('cybersecurity', 'cybersecurity', '2026-10-01')
    registry = {row['id']: row for row in [older, newer]}
    assert eligible_claims(registry, '2026-10-24') == set(registry)
    draft = CardDraft.model_validate({'claims': [
        {'id': 'c1', 'kind': 'bank_view', 'text': older['text'], 'evidence_ids': [older['id']]},
        {'id': 'c2', 'kind': 'bank_view', 'text': newer['text'], 'evidence_ids': [newer['id']]}]})
    assert verify_draft(draft, registry) == []
    revision = bank_claim('reputation-revised', 'reputation', '2026-10-20', revision_of='reputation')
    registry[revision['id']] = revision
    assert eligible_claims(registry, '2026-10-24') == {'cybersecurity', 'reputation-revised'}
    assert 'c1:bank_claim_superseded_in_context' in verify_draft(draft, registry)
