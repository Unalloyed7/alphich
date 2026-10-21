import { test } from 'node:test';
import assert from 'node:assert/strict';
import { relevanceDescription, claimReferenceTexts } from '../../frontend/src/insightPresentation.ts';

test('server relevance explanation takes precedence over fallback', () => {
  assert.equal(relevanceDescription({ code: 'credit_rate', explanation: 'Есть кредитный продукт', feature_ids: ['abc'] }), 'Есть кредитный продукт');
  assert.equal(relevanceDescription({ reason: 'Открытый текст' }), 'Открытый текст');
  assert.equal(relevanceDescription({ code: 'credit_rate' }), 'Основание отбора: credit_rate');
  assert.equal(relevanceDescription({}), null);
});

test('risk/opportunity refs resolve to exact server-owned claims', () => {
  const claims = [
    { id: 'c1', text: 'Факт новости' },
    { id: 'c4', text: 'Если договор плавающий, расходы зависят от ставки' },
  ];
  assert.deepEqual(claimReferenceTexts(['c4'], claims), ['Если договор плавающий, расходы зависят от ставки']);
  assert.deepEqual(claimReferenceTexts(['absent'], claims), ['Тезис absent не найден в карточке']);
  assert.deepEqual(claimReferenceTexts([], claims), []);
});

test('evidence uses parent insight scope; answer evidence includes answer_id', async () => {
  const { analysisApi } = await import('../../frontend/src/api.ts');
  const calls = [];
  const previous = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return { ok: true, json: async () => ({ text: 'Сохранённый фрагмент' }) };
  };
  try {
    await analysisApi.evidence('bank-clm/1', 'insight-1', 'demo-session');
    await analysisApi.evidence('bank-clm/2', 'insight-1', 'demo-session', 'answer-2');
    assert.equal(calls.length, 2);
    const first = new URL(calls[0].url, 'http://localhost');
    const second = new URL(calls[1].url, 'http://localhost');
    assert.equal(first.pathname, '/api/v1/evidence/bank-clm%2F1');
    assert.equal(first.searchParams.get('insight_id'), 'insight-1');
    assert.equal(first.searchParams.has('answer_id'), false);
    assert.equal(second.searchParams.get('insight_id'), 'insight-1');
    assert.equal(second.searchParams.get('answer_id'), 'answer-2');
    assert.equal(calls[1].options.headers['X-Demo-Session'], 'demo-session');
  } finally {
    globalThis.fetch = previous;
  }
});

test('question is a separate, explicit idempotent POST', async () => {
  const { analysisApi } = await import('../../frontend/src/api.ts');
  const calls = [];
  const previous = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return { ok: true, json: async () => ({ id: 'job1', status: 'queued' }) };
  };
  try {
    await analysisApi.ask('insight-1', 'Какие условия кредитного договора известны?', 'session-token');
    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, '/api/v1/insights/insight-1/questions');
    assert.equal(calls[0].options.method, 'POST');
    assert.equal(calls[0].options.headers['X-Demo-Session'], 'session-token');
    assert.match(calls[0].options.headers['Idempotency-Key'], /^question-/);
    assert.deepEqual(JSON.parse(calls[0].options.body), { question: 'Какие условия кредитного договора известны?' });
  } finally {
    globalThis.fetch = previous;
  }
});
