import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react';
import { analysisApi, api } from './api';
import type { Job, DemoSession } from './types';
import type { Answer, AuditRun, Claim, Evidence, FeedItem, FeedResult, Insight, LlmStatus } from './analysis';
import { claimReferenceTexts, relevanceDescription } from './insightPresentation';
import './insight.css';

function message(error: unknown): string {
  return error instanceof Error ? error.message : 'Не удалось выполнить запрос';
}

function dateLabel(value?: string | null): string {
  if (!value) return 'Не указана';
  const time = new Date(value);
  return Number.isNaN(time.getTime()) ? value : time.toLocaleString('ru-RU');
}

const claimKinds: Record<Claim['kind'], string> = {
  news_fact: 'Факт новости', company_fact: 'Признак компании', bank_view: 'Позиция Альфа-Банка',
  model_inference: 'Интерпретация AI',
};

function BudgetStatus({ value }: { value: LlmStatus | null }) {
  if (!value) return <p className="insight-muted">Проверяем настройки аналитического шлюза…</p>;
  return <div className="insight-budget" role="status">
    <span>{value.configured ? 'Модели настроены' : 'Модели не настроены — генерация отключена'}</span>
    <span>Бюджет (UTC {value.budget_day_utc}): {value.spent_rub} ₽ потрачено · {value.reserved_rub} ₽ зарезервировано · лимит {value.daily_limit_rub} ₽</span>
    {!value.bank_index_present && <span>Банковский индекс не найден: подтверждённая позиция банка может быть недоступна.</span>}
  </div>;
}

function Claims({ value, onEvidence }: { value: Insight | Answer; onEvidence: (id: string, answerId?: string) => void }) {
  return <div className="insight-claims">
    {value.claims.length === 0 && <p className="insight-muted">Подтверждённых тезисов пока нет.</p>}
    {value.claims.map(claim => <article className="insight-claim" key={claim.id}>
      <span className={`insight-claim-kind insight-claim-${claim.kind}`}>{claimKinds[claim.kind] ?? claim.kind}</span>
      <p>{claim.text}</p>
      {claim.conditions?.length > 0 && <p className="insight-muted">Условия: {claim.conditions.join(' · ')}</p>}
      <div className="insight-evidence-links">{claim.evidence_ids.map(id => <button type="button" key={id} onClick={() => onEvidence(id, 'insight_id' in value ? value.id : undefined)}>Источник · {id.slice(0, 18)}</button>)}</div>
    </article>)}
  </div>;
}

/** Event-scoped analysis panel. Uses ONLY pre-existing server analysis endpoints. */
export function EventInsight({ session, eventId, onInsightReady }: { session: DemoSession; eventId: string; onInsightReady: (eventId: string, status: string) => void }) {
  const [status, setStatus] = useState<LlmStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [insight, setInsight] = useState<Insight | null>(null);
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [question, setQuestion] = useState('');
  const [questionJob, setQuestionJob] = useState<Job | null>(null);
  const [evidence, setEvidence] = useState<Evidence | null>(null);
  const [evidenceError, setEvidenceError] = useState<string | null>(null);
  const [evidenceLoading, setEvidenceLoading] = useState(false);
  const evidenceRef = useRef<HTMLDivElement>(null);
  const evidenceRequest = useRef(0);
  const [run, setRun] = useState<AuditRun | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    void analysisApi.status(session.token).then(x => { if (active) setStatus(x); })
      .catch(e => { if (active) setStatusError(message(e)); });
    // A previously generated result is read from the server, never fabricated locally.
    void analysisApi.feed(session.company_id, session.token).then(async feed => {
      const id = feed.items.find(item => item.event_id === eventId)?.insight?.id;
      if (!id) return;
      const item = await analysisApi.insight(id, session.token);
      if (active) { setInsight(item); onInsightReady(eventId, item.status); }
    }).catch(() => { /* Feed is optional if ranking isn't available. */ });
    return () => { active = false; };
  }, [session.company_id, session.token, eventId, onInsightReady]);

  const poll = useCallback((current: Job, setter: (j: Job) => void, load: (id: string) => Promise<void>) => {
    let canceled = false;
    const next = async () => {
      try {
        const latest = await api.job(current.id, session.token);
        if (canceled) return;
        setter(latest);
        if (latest.status === 'succeeded' && latest.result_id) await load(latest.result_id);
      } catch (e) { if (!canceled) setError(message(e)); }
    };
    void next();
    const timer = window.setInterval(() => { if (!canceled) void next(); }, 2500);
    return () => { canceled = true; window.clearInterval(timer); };
  }, [session.token]);

  useEffect(() => {
    if (!job || (job.status !== 'queued' && job.status !== 'running')) return;
    return poll(job, setJob, async id => { const card = await analysisApi.insight(id, session.token); setInsight(card); onInsightReady(eventId, card.status); void analysisApi.status(session.token).then(setStatus).catch(() => { }); });
  }, [job?.id, job?.status, poll, session.token, onInsightReady]);

  useEffect(() => {
    if (!questionJob || (questionJob.status !== 'queued' && questionJob.status !== 'running')) return;
    return poll(questionJob, setQuestionJob, async id => { setAnswer(await analysisApi.answer(id, session.token)); void analysisApi.status(session.token).then(setStatus).catch(() => { }); });
  }, [questionJob?.id, questionJob?.status, poll, session.token]);

  const start = async () => {
    setBusy(true); setError(null); setAnswer(null); setEvidence(null); setRun(null);
    try {
      const next = await analysisApi.createInsight(session.company_id, eventId, session.token);
      setJob(next);
      if (next.status === 'succeeded' && next.result_id) { const card = await analysisApi.insight(next.result_id, session.token); setInsight(card); onInsightReady(eventId, card.status); }
    } catch (e) { setError(message(e)); } finally { setBusy(false); }
  };

  const ask = async (e: FormEvent) => {
    e.preventDefault();
    if (!insight || question.trim().length < 6) return;
    setBusy(true); setError(null); setAnswer(null);
    try {
      const next = await analysisApi.ask(insight.id, question.trim(), session.token);
      setQuestionJob(next);
      if (next.status === 'succeeded' && next.result_id) setAnswer(await analysisApi.answer(next.result_id, session.token));
    } catch (err) { setError(message(err)); } finally { setBusy(false); }
  };

  const showEvidence = async (id: string, answerId?: string) => {
    if (!insight) return;
    const requestId = ++evidenceRequest.current;
    setEvidence(null); setEvidenceError(null); setEvidenceLoading(true);
    try {
      const result = await analysisApi.evidence(id, insight.id, session.token, answerId);
      if (requestId === evidenceRequest.current) setEvidence(result);
    } catch (e) {
      if (requestId === evidenceRequest.current) setEvidenceError(message(e));
    } finally {
      if (requestId === evidenceRequest.current) setEvidenceLoading(false);
    }
  };

  useEffect(() => {
    if (evidenceLoading || evidence || evidenceError) {
      evidenceRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [evidenceLoading, evidence, evidenceError]);

  const showRun = async (id: string) => {
    setRun(null); setError(null);
    try { setRun(await analysisApi.run(id, session.token)); } catch (e) { setError(message(e)); }
  };

  return <section className="insight-panel" aria-label="Персонализированная аналитика">
    <header className="insight-heading"><div><small>ПЕРСОНАЛИЗИРОВАННАЯ АНАЛИТИКА</small><h3>Влияние на компанию</h3></div></header>
    <BudgetStatus value={status} />
    {statusError && <p className="insight-error">Статус шлюза: {statusError}</p>}
    {!insight && <p className="insight-muted">Проверяемая аналитическая карточка пока не сформирована. Никаких прогнозов из исходной новости автоматически не приписываем банку.</p>}
    <button type="button" className="insight-primary" disabled={busy || !status?.configured || job?.status === 'queued' || job?.status === 'running'} onClick={() => void start()}>
      {busy ? 'Подождите…' : insight ? 'Сформировать заново' : 'Сформировать инсайт'}
    </button>
    {job && <p className="insight-job">Задача {job.id.slice(0, 16)}: {job.status} · {job.stage}{job.error ? ` · ${job.error}` : ''}</p>}
    {error && <p className="insight-error" role="alert">{error}</p>}
    {insight && <div className="insight-result">
      <h4>{insight.headline}</h4><p className="insight-muted">Статус: {insight.status}. Анализ на {dateLabel(insight.as_of)}. Проверка: {insight.verification?.status}.</p>
      {typeof insight.relevance_grade === 'number' && <p><strong>Значимость: {insight.relevance_grade}/5.</strong> {insight.relevance_selected ? 'Проходит отбор в ленту.' : 'Ниже порога отбора в ленту.'}</p>}
      <p><strong>Позиция банка:</strong> {insight.bank_view.status === 'found' ? 'Найдена в источниках' : insight.bank_view.status === 'historical_only' ? 'Только исторические материалы' : insight.bank_view.status === 'conflicting' ? 'Обнаружен конфликт источников' : 'Подтверждённая позиция отсутствует'}</p>
      {insight.relevance_reasons?.length > 0 && <div><h5>Почему актуально</h5>{insight.relevance_reasons.map((reason, index) => <p key={index}>{relevanceDescription(reason) ?? 'Описание основания не передано сервером'}</p>)}</div>}
      <h5>Выводы и доказательства</h5><Claims value={insight} onEvidence={(id) => void showEvidence(id)} />
      {insight.risks.length > 0 && <div><h5>Риски</h5>{claimReferenceTexts(insight.risks, insight.claims).map((text, index) => <p key={index}>{text}</p>)}</div>}
      {insight.opportunities.length > 0 && <div><h5>Возможности</h5>{claimReferenceTexts(insight.opportunities, insight.claims).map((text, index) => <p key={index}>{text}</p>)}</div>}
      {insight.next_steps.length > 0 && <div><h5>Следующие шаги</h5>{insight.next_steps.map((step, idx) => <p key={idx}>{step.text}{step.conditions?.length ? ` · При условиях: ${step.conditions.join('; ')}` : ''}</p>)}</div>}
      {insight.missing_data?.length > 0 && <p className="insight-muted">Не хватает данных: {insight.missing_data.join('; ')}</p>}
      {(insight.verification?.issues ?? []).length > 0 && <p className="insight-muted">Проверка: {(insight.verification?.issues ?? []).join('; ')}</p>}
      <button type="button" className="insight-link" onClick={() => void showRun(insight.run_id)}>Журнал действий агента</button>
      <form className="insight-question" onSubmit={e => { void ask(e); }}><label htmlFor={`question-${eventId}`}>Уточняющий вопрос</label>
        <textarea id={`question-${eventId}`} rows={3} minLength={6} maxLength={1000} value={question} onChange={e => setQuestion(e.target.value)} placeholder="Какие данные нужно уточнить, чтобы оценить влияние?" required />
        <button type="submit" disabled={busy || !status?.configured || question.trim().length < 6 || questionJob?.status === 'queued' || questionJob?.status === 'running'}>Задать вопрос</button>
      </form>
      {questionJob && <p className="insight-job">Вопрос: {questionJob.status} · {questionJob.stage} {questionJob.error ?? ''}</p>}
      {answer && <div className="insight-answer"><h5>Ответ: {answer.question}</h5><Claims value={answer} onEvidence={(id, answerId) => void showEvidence(id, answerId)} /></div>}
    </div>}
    {(evidenceLoading || evidenceError || evidence) && <div ref={evidenceRef} className="insight-evidence-region">
    {evidenceLoading && <p className="insight-muted" role="status">Загружаем подтверждающий фрагмент…</p>}
    {evidenceError && <p className="insight-error" role="alert">Доказательство: {evidenceError}</p>}
    {evidence && <div className="insight-evidence" aria-live="polite"><h5>Подтверждающий фрагмент</h5><p>{evidence.text ?? 'Текст недоступен в этом типе доказательства.'}</p>
      <p className="insight-muted">{[evidence.title, evidence.kind, evidence.document_id, evidence.page != null ? `стр. ${evidence.page}` : null, evidence.published_at].filter(Boolean).join(' · ')}</p>
      {typeof evidence.url === 'string' && /^https:\/\//.test(evidence.url) && <a href={evidence.url} target="_blank" rel="noopener noreferrer">Открыть документ</a>}</div>}
    </div>}
    {run && <div className="insight-trace"><h5>Проверяемый журнал · {run.status}</h5>
      <p>Этапов: {run.steps?.length ?? 0}; вызовов модели: {run.llm_calls?.length ?? 0}</p>
      {(run.steps ?? []).map((step, idx) => <p key={idx}>{idx + 1}. {step.action ?? 'Шаг'} · {JSON.stringify(step).slice(0, 300)}</p>)}
      {run.llm_calls?.map((call, idx) => <p key={idx}>{call.model} · {call.status} · расход: {call.cost_rub ?? 'неизвестен'} ₽</p>)}</div>}
  </section>;
}

/** Top-of-funnel feed: no analysis requests or model charges on page load. */
export function InsightFeed({ session, onEvent }: { session: DemoSession; onEvent: (id: string) => void }) {
  const [feed, setFeed] = useState<FeedResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const feedRequest = useRef(0);
  const reload = useCallback(async (silent = false) => {
    const current = ++feedRequest.current;
    if (!silent) { setLoading(true); setError(null); }
    try {
      const response = await analysisApi.feed(session.company_id, session.token);
      if (current === feedRequest.current) setFeed(response);
    } catch (e) {
      if (!silent && current === feedRequest.current) setError(message(e));
    } finally {
      if (!silent && current === feedRequest.current) setLoading(false);
    }
  }, [session.company_id, session.token]);
  useEffect(() => {
    setFeed(null);
    void reload();
    const refresh = () => { if (document.visibilityState !== 'hidden') void reload(true); };
    const interval = window.setInterval(refresh, 20_000);
    document.addEventListener('visibilitychange', refresh);
    return () => { feedRequest.current++; window.clearInterval(interval); document.removeEventListener('visibilitychange', refresh); };
  }, [reload]);
  return <div className="insight-feed panel">
    <div className="section-header"><div><div className="eyebrow">КОМПАНИЯ / РОЛЬ</div><h2>Аналитическая лента</h2><p>События с оценкой 4–5 из 5: учитываем деятельность, географию и масштаб.</p></div>
      <button type="button" className="ghost-button" onClick={() => void reload()} disabled={loading}>{loading ? 'Загрузка…' : 'Обновить'}</button></div>
    {error && <p className="insight-error" role="alert">{error}</p>}
    {!loading && feed?.items.length === 0 && <p className="insight-muted">Пока нет событий с подтверждённой значимостью 4–5. Загруженные публикации доступны в разделе событий; для оценки нужен разбор фактов.</p>}
    {feed?.items.map((item: FeedItem) => <article key={`${item.event_id}:${item.event_version}`} className="insight-feed-item">
      <div><strong>{item.insight?.headline || item.title || item.event_id}</strong><p className="insight-muted">Значимость {item.relevance_grade ?? 'не определена'}/5 · {item.insight?.status || 'Без карточки'}</p>
        {item.relevance_assessment?.business_channel && <p>{item.relevance_assessment.business_channel}</p>}
        {(item.reasons ?? item.relevance_reasons ?? []).slice(0, 3).map((reason, index) => <p key={index}>{relevanceDescription(reason) ?? 'Описание основания не передано сервером'}</p>)}
      </div><button type="button" className="insight-link" onClick={() => onEvent(item.event_id)}>Открыть событие →</button>
    </article>)}
    {feed?.limitations?.length ? <p className="insight-muted">Ограничения: {feed.limitations.join(', ')}</p> : null}
  </div>;
}
