import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import { analysisApi, api } from './api';
import type { Job, DemoSession } from './types';
import type { Answer, AuditRun, Claim, Evidence, FeedItem, Insight, LlmStatus } from './analysis';
import { claimReferenceTexts, relevanceDescription } from './insightPresentation';
import './insight.css';
import { DiscoveryPanel } from './DiscoveryPanel';
import { Icon } from './Icon';
import { bankLabel, analysisLabel, limitationLabel } from './presentation';
import type { FeedFilter, PersonalFeedState } from './personalFeed';
import type { ReadingState } from './readingState';
import { AccessibleTabs } from './AccessibleTabs';
import { FeedbackPanel } from './FeedbackPanel';
import { AuditTimeline, BankCitation, InsightTrust } from './InsightTrust';

function message(error: unknown): string {
  return error instanceof Error ? error.message : 'Не удалось выполнить запрос';
}

function dateLabel(value?: string | null): string {
  if (!value) return 'Не указана';
  const time = new Date(value);
  return Number.isNaN(time.getTime()) ? value : time.toLocaleString('ru-RU', { dateStyle: 'medium', timeStyle: 'short' });
}

function insightJobMessage(job: Job): string {
  if (job.status === 'failed') {
    if (['news_e5_unavailable', 'news_e5_invalid_vector', 'news_e5_invalid_threshold'].includes(job.error ?? '')) return 'Смысловая проверка недоступна. Платный анализ новости не запускался. Попробуйте позже.';
    if (job.error === 'news_e5_company_context_missing') return 'Недостаточно подтверждённых данных компании для смысловой проверки. Платный анализ не запускался.';
    if (job.error === 'news_e5_article_context_missing') return 'Нет текста новости для смысловой проверки. Платный анализ не запускался.';
    return `Не удалось завершить разбор: ${job.error || 'попробуйте ещё раз'}`;
  }
  if (job.status === 'succeeded') return job.result_type === 'insight'
    ? 'Разбор сохранён'
    : 'Новость проверена, инсайт не сформирован. Персональный разбор создаётся для новостей с оценкой 4 или 5.';
  return ({ waiting_semantic: 'Проверяем смысловую связь с компанией…', semantic_check: 'Проверяем смысловую связь с компанией…',
    waiting_facts: 'Проверяем смысловую связь, затем извлекаем факты и оцениваем значимость…',
    relevance_assessment: 'Оцениваем значимость новости для компании…',
  } as Record<string, string>)[job.stage] ?? 'Проверяем значимость и готовим персональную аналитику для оценки 4–5…';
}

const claimKinds: Record<Claim['kind'], string> = {
  news_fact: 'Факт новости', company_fact: 'Признак компании', bank_view: 'Позиция Альфа-Банка',
  model_inference: 'Вывод AI · не официальная позиция банка',
};

function BudgetStatus({ value }: { value: LlmStatus | null }) {
  if (!value) return <p className="insight-muted">Проверяем настройки аналитического шлюза…</p>;
  return <details className="insight-budget"><summary>Настройки аналитики и бюджет</summary><div role="status">
    <span>{value.configured ? 'Модели настроены' : 'Модели не настроены — генерация отключена'}</span>
    <span>Бюджет (UTC {value.budget_day_utc}): {value.spent_rub} ₽ потрачено · {value.reserved_rub} ₽ зарезервировано · лимит {value.daily_limit_rub} ₽</span>
    {!value.bank_index_present && <span>Банковский индекс не найден: подтверждённая позиция банка может быть недоступна.</span>}
  </div></details>;
}

function Claims({ value, onEvidence, kind = 'all' }: { value: Insight | Answer; kind?: Claim['kind'] | 'all'; onEvidence: (id: string, answerId?: string) => void }) {
  return <div className="insight-claims">
    {value.claims.length === 0 && <p className="insight-muted">Подтверждённых тезисов пока нет.</p>}
    {value.claims.filter(claim => kind === 'all' || claim.kind === kind).map(claim => <article className="insight-claim" key={claim.id}>
      <span className={`insight-claim-kind insight-claim-${claim.kind}`}>{claimKinds[claim.kind] ?? claim.kind}</span>
      <p>{claim.text}</p>
      {claim.conditions?.length > 0 && <p className="insight-muted">Условия: {claim.conditions.join(' · ')}</p>}
      <div className="insight-evidence-links">{claim.evidence_ids.map((id, index) => <button type="button" key={id} onClick={() => onEvidence(id, 'insight_id' in value ? value.id : undefined)}><Icon name="link" size={13}/> Источник {index + 1}</button>)}</div>
    </article>)}
  </div>;
}

/** Event-scoped analysis panel. Uses ONLY pre-existing server analysis endpoints. */
export function EventInsight({ session, eventId, onInsightReady }: { session: DemoSession; eventId: string; onInsightReady: (eventId: string, status: string) => void }) {
  const [tab, setTab] = useState<'summary' | 'evidence' | 'chat'>('summary');
  const [claimFilter, setClaimFilter] = useState<Claim['kind'] | 'all'>('all');
  const [savedLoading, setSavedLoading] = useState(true);
  const [savedError, setSavedError] = useState<string | null>(null);
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
    void analysisApi.savedEventInsight(eventId, session.token).then(({ insight: item }) => {
      if (active && item) { setInsight(item); onInsightReady(eventId, item.status); }
    }).catch(e => { if (active) setSavedError(message(e)); }).finally(() => { if (active) setSavedLoading(false); });
    return () => { active = false; };
  }, [session.company_id, session.token, eventId, onInsightReady]);

  const poll = useCallback((current: Job, setter: (j: Job) => void, resultType: 'insight' | 'answer', load: (id: string) => Promise<void>) => {
    let canceled = false;
    const next = async () => {
      try {
        const latest = await api.job(current.id, session.token);
        if (canceled) return;
        setter(latest);
        if (latest.status === 'succeeded' && latest.result_id && latest.result_type === resultType) await load(latest.result_id);
      } catch (e) { if (!canceled) setError(message(e)); }
    };
    void next();
    const timer = window.setInterval(() => { if (!canceled) void next(); }, 2500);
    return () => { canceled = true; window.clearInterval(timer); };
  }, [session.token]);

  useEffect(() => {
    if (!job || (job.status !== 'queued' && job.status !== 'running')) return;
    return poll(job, setJob, 'insight', async id => { const card = await analysisApi.insight(id, session.token); setInsight(card); onInsightReady(eventId, card.status); void analysisApi.status(session.token).then(setStatus).catch(() => { }); });
  }, [job?.id, job?.status, poll, session.token, onInsightReady]);

  useEffect(() => {
    if (!questionJob || (questionJob.status !== 'queued' && questionJob.status !== 'running')) return;
    return poll(questionJob, setQuestionJob, 'answer', async id => { setAnswer(await analysisApi.answer(id, session.token)); void analysisApi.status(session.token).then(setStatus).catch(() => { }); });
  }, [questionJob?.id, questionJob?.status, poll, session.token]);

  useEffect(() => {
    if (!job || (job.status !== 'succeeded' && job.status !== 'failed')) return;
    let active = true;
    void analysisApi.status(session.token).then(value => { if (active) setStatus(value); }).catch(() => {});
    return () => { active = false; };
  }, [job?.id, job?.status, session.token]);

  const start = async () => {
    setBusy(true); setError(null); setSavedError(null); setAnswer(null); setEvidence(null); setRun(null);
    try {
      const next = await analysisApi.createInsight(session.company_id, eventId, session.token);
      setJob(next);
      if (next.status === 'succeeded' && next.result_id && next.result_type === 'insight') { const card = await analysisApi.insight(next.result_id, session.token); setInsight(card); onInsightReady(eventId, card.status); }
    } catch (e) { setError(message(e)); } finally { setBusy(false); }
  };

  const ask = async (e: FormEvent) => {
    e.preventDefault();
    if (!insight || question.trim().length < 6) return;
    setBusy(true); setError(null); setAnswer(null);
    try {
      const next = await analysisApi.ask(insight.id, question.trim(), session.token);
      setQuestionJob(next);
      if (next.status === 'succeeded' && next.result_id && next.result_type === 'answer') setAnswer(await analysisApi.answer(next.result_id, session.token));
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

  const analyzing = busy || job?.status === 'queued' || job?.status === 'running';
  const suggestions = ['Как это влияет на мою компанию?', 'Какие данные нужны для решения?', 'Что об этом говорят аналитики банка?'];
  return <section className="insight-panel" aria-label="Персонализированная аналитика">
    {savedLoading && <div className="loading-panel" role="status"><span className="spinner"/> Загружаем сохранённый разбор…</div>}
    {savedError && <p className="insight-error" role="alert">Не удалось загрузить сохранённый разбор: {savedError}</p>}
    {statusError && <p className="insight-error" role="alert">Не удалось проверить доступность аналитики: {statusError}</p>}
    {!insight && !savedLoading && <div className="analysis-empty"><div className="empty-icon"><Icon name="activity" size={28}/></div><span className="eyebrow">ОТ СОБЫТИЯ К РЕШЕНИЮ</span><h3>Что эта новость меняет<br/>для вашей компании?</h3><p>Проверим связь с бизнесом и оценим значимость. Для оценки 4–5 подготовим разбор с доказательствами и действиями.</p><button type="button" className="insight-primary" disabled={analyzing || !status?.configured} onClick={() => void start()}>{analyzing ? 'Проверяем значимость…' : <>Разобрать влияние <Icon name="arrow" size={17}/></>}</button><small>Запуск использует общий бюджет аналитики.</small></div>}
    {job && <p className={`insight-job ${job.status === 'failed' ? 'insight-error' : ''}`} role="status">{analyzing && <span className="spinner tiny"/>} {insightJobMessage(job)}</p>}
    {error && <p className="insight-error" role="alert">{error}</p>}
    {insight && <div className="insight-result">
      <div className="analysis-meta"><span className={`analysis-state ${insight.status === 'insufficient_evidence' ? 'state-warning' : ''}`}>{analysisLabel(insight.status)}</span>{typeof insight.relevance_grade === 'number' && <span className="grade-inline">Значимость <strong>{insight.relevance_grade}/5</strong></span>}<span>Анализ: {dateLabel(insight.as_of)}</span></div>
      <h3 className="analysis-headline">{insight.headline}</h3>
      {insight.analysis_method_current === false && <p className="insight-warning">Этот разбор сохранён до обновления правил. Его можно обновить в настройках ниже.</p>}
      {insight.status === 'insufficient_evidence' && <p className="insight-warning">Доказательств недостаточно. Выводы ограничены доступными данными.</p>}
      {!!insight.verification?.issues?.length && <p className="insight-warning">Проверка выявила замечания — ознакомьтесь с ограничениями в обосновании.</p>}
      <InsightTrust insight={insight} onDetails={kind => { setClaimFilter(kind); setTab('evidence'); }}/>
      <AccessibleTabs id={`insight-sections-${eventId}`} label="Разделы аналитики" className="segmented-tabs" value={tab} onChange={setTab} items={[{ id: 'summary', label: 'Коротко', icon: 'grid' }, { id: 'evidence', label: 'Обоснование', icon: 'link' }, { id: 'chat', label: 'Спросить аналитика', icon: 'send' }]}/>
      <div id={`insight-sections-${eventId}-panel`} role="tabpanel" aria-labelledby={`insight-sections-${eventId}-${tab}`}>
      {tab === 'summary' && <div className="analysis-summary">
        <div className="impact-summary"><span className="eyebrow">ВЛИЯНИЕ НА ВАШ БИЗНЕС</span>{insight.relevance_assessment?.business_channel ? <p>{insight.relevance_assessment.business_channel}</p> : <p>{insight.claims.find(claim => claim.kind === 'model_inference')?.text || 'Подтверждённое влияние на компанию пока не установлено. Проверьте факты и ограничения в обосновании.'}</p>}</div>
        <div className="bank-summary"><Icon name="briefcase" size={19}/><div><strong>{bankLabel(insight.bank_view.status)}</strong>{insight.bank_view.status === 'found' ? <><p>{insight.claims.find(claim => claim.kind === 'bank_view')?.text || 'Откройте обоснование, чтобы прочитать подтверждённую позицию.'}</p><BankCitation insight={insight} token={session.token}/><button type="button" className="text-button" onClick={() => { setClaimFilter('bank_view'); setTab('evidence'); }}>Источники и дата материала <Icon name="arrow" size={13}/></button></> : <p>{insight.bank_view.reason || 'Интерпретация AI не является позицией банка.'}</p>}</div></div>
        {(insight.risks.length > 0 || insight.opportunities.length > 0) && <div className="impact-grid">{insight.risks.length > 0 && <div className="impact-card risk-card"><h4><Icon name="alert" size={17}/> Возможные риски</h4>{claimReferenceTexts(insight.risks, insight.claims).map((text, i) => <p key={i}>{text}</p>)}</div>}{insight.opportunities.length > 0 && <div className="impact-card opportunity-card"><h4><Icon name="activity" size={17}/> Возможности</h4>{claimReferenceTexts(insight.opportunities, insight.claims).map((text, i) => <p key={i}>{text}</p>)}</div>}</div>}
        {insight.next_steps.length > 0 && <div className="next-steps"><span className="eyebrow">ОТ АНАЛИТИКИ К ДЕЙСТВИЮ</span><h4>Что проверить дальше</h4><ol>{insight.next_steps.map((step, idx) => <li key={idx}><span>{String(idx + 1).padStart(2, '0')}</span><div><p>{step.text}</p>{step.conditions?.length > 0 && <small>При условиях: {step.conditions.join('; ')}</small>}</div></li>)}</ol></div>}
        {!!insight.missing_data?.length && <details className="analysis-details"><summary>Каких данных не хватает <span>{insight.missing_data.length}</span></summary><ul>{insight.missing_data.map((text, i) => <li key={i}>{text}</li>)}</ul></details>}
        <button type="button" className="text-button" onClick={() => setTab('evidence')}>Проверить выводы по источникам <Icon name="arrow" size={16}/></button>
      </div>}
      {tab === 'evidence' && <div className="analysis-evidence-tab"><h4>Почему это касается компании</h4>{insight.relevance_reasons.map((reason, index) => <p key={index}>{relevanceDescription(reason) ?? 'Описание основания не передано сервером'}</p>)}<h4>Факты, позиция банка и интерпретация</h4><p className="insight-muted">Тип каждого тезиса указан отдельно. Кнопка источника открывает подтверждающий фрагмент.</p><div className="claim-filters" role="group" aria-label="Тип тезисов"><button type="button" aria-pressed={claimFilter === 'all'} onClick={() => setClaimFilter('all')}>Все тезисы</button>{Object.entries(claimKinds).map(([kind, label]) => <button type="button" key={kind} aria-pressed={claimFilter === kind} onClick={() => setClaimFilter(kind as Claim['kind'])}>{label}</button>)}</div><Claims value={insight} kind={claimFilter} onEvidence={id => void showEvidence(id)}/>{claimFilter !== 'all' && !insight.claims.some(claim => claim.kind === claimFilter) && <p className="insight-muted">Тезисов этого типа в разборе нет. Это ограничение доступных данных.</p>}<details className="analysis-details"><summary>Ограничения и результат проверки</summary><p>Результат проверки: {analysisLabel(insight.verification?.status)}.</p>{[...(insight.limitations ?? []), ...(insight.verification?.issues ?? []), ...(insight.relevance_assessment?.limitations ?? [])].map((text, i) => <p key={i}>{limitationLabel(text)}</p>)}{typeof insight.relevance_selected === 'boolean' && <p>{insight.relevance_selected ? 'Событие прошло порог отбора в персональную ленту.' : 'Событие не прошло порог отбора в персональную ленту.'}</p>}</details><button type="button" className="insight-link" onClick={() => void showRun(insight.run_id)}><Icon name="list" size={15}/> Показать журнал анализа</button></div>}
      {tab === 'chat' && <div className="analysis-chat"><span className="eyebrow">ДИАЛОГ ПО ЭТОМУ СОБЫТИЮ</span><h4>Разберёмся глубже</h4><p className="insight-muted">Ответ учитывает эту карточку и профиль компании. Существенные выводы сопровождаются источниками.</p><div className="question-suggestions">{suggestions.map(text => <button type="button" key={text} onClick={() => setQuestion(text)}>{text}</button>)}</div><form className="insight-question" onSubmit={e => { void ask(e); }}><label htmlFor={`question-${eventId}`}>Ваш вопрос</label><textarea id={`question-${eventId}`} rows={3} minLength={6} maxLength={1000} value={question} onChange={e => setQuestion(e.target.value)} placeholder="Например, какие условия кредита нужно проверить?" required/><div className="question-footer"><small>Ответ использует общий бюджет аналитики.</small><button type="submit" disabled={busy || !status?.configured || question.trim().length < 6 || questionJob?.status === 'queued' || questionJob?.status === 'running'}>Задать вопрос <Icon name="send" size={15}/></button></div></form>{questionJob && <p className={questionJob.status === 'failed' ? 'insight-error' : 'insight-job'} role="status">{questionJob.status === 'failed' ? `Не удалось получить ответ: ${questionJob.error || 'ошибка обработки'}` : questionJob.status === 'succeeded' ? 'Ответ готов' : 'Аналитик готовит ответ…'}</p>}{answer && <div className="insight-answer"><h4>{answer.question}</h4><p className="insight-muted">{analysisLabel(answer.status)}</p><Claims value={answer} onEvidence={(id, answerId) => void showEvidence(id, answerId)}/>{!!answer.missing_data?.length && <div className="answer-gaps"><strong>Что ещё нужно уточнить</strong><ul>{answer.missing_data.map((text, i) => <li key={i}>{text}</li>)}</ul></div>}</div>}</div>}
      </div>
      <FeedbackPanel key={insight.id} insightId={insight.id} token={session.token}/>
    </div>}
    {(evidenceLoading || evidenceError || evidence) && <div ref={evidenceRef} className="insight-evidence-region"><div className="section-header"><strong>Подтверждающий фрагмент</strong><button type="button" className="icon-button" aria-label="Закрыть подтверждающий фрагмент" onClick={() => { evidenceRequest.current++; setEvidence(null); setEvidenceError(null); setEvidenceLoading(false); }}><Icon name="close" size={17}/></button></div>{evidenceLoading && <p className="insight-muted" role="status">Загружаем источник…</p>}{evidenceError && <p className="insight-error" role="alert">{evidenceError}</p>}{evidence && <div className="insight-evidence" aria-live="polite"><p>{evidence.text ?? 'Текст недоступен в этом типе доказательства.'}</p><p className="insight-muted">{[evidence.title, evidence.page != null ? `стр. ${evidence.page}` : null, evidence.published_at].filter(Boolean).join(' · ')}</p>{typeof evidence.url === 'string' && /^https:\/\//.test(evidence.url) && <a href={evidence.url} target="_blank" rel="noopener noreferrer">Открыть документ ↗</a>}</div>}</div>}
    {run && <AuditTimeline run={run}/>}
    <div className="analysis-settings"><BudgetStatus value={status}/>{insight && <details className="analysis-details"><summary>Обновить разбор</summary><p className="insight-muted">Повторный анализ использует общий бюджет. Сохранённый результат останется видимым до завершения.</p><button type="button" className="secondary-button" disabled={analyzing || !status?.configured} onClick={() => void start()}><Icon name="refresh" size={15}/>{analyzing ? 'Обновляем…' : 'Сформировать заново'}</button></details>}</div>
  </section>;
}

/** Top-of-funnel feed: no analysis requests or model charges on page load. */
export function InsightFeed({ session, data, reading, filter, onFilterChange, onEvent, compact = false, onAll }: {
  session: DemoSession; data: PersonalFeedState; reading: ReadingState; filter: FeedFilter; onFilterChange: (filter: FeedFilter) => void;
  onEvent: (id: string) => void; compact?: boolean; onAll?: () => void;
}) {
  const { feed, loading, error, reload } = data;
  const [search, setSearch] = useState('');
  const [sort, setSort] = useState('priority');
  const discoveryChanged = useCallback(() => { void reload(true); }, [reload]);
  const visible = useMemo(() => {
    const query = search.trim().toLocaleLowerCase('ru-RU');
    return (feed?.items ?? []).filter(item => (filter !== 'critical' || item.relevance_grade === 5) && (filter !== 'bank' || item.insight?.bank_view?.status === 'found') && (filter !== 'saved' || !!item.insight) && (filter !== 'bookmarks' || reading.isBookmarked(item.event_id)) && (filter !== 'unread' || !reading.isRead(item.event_id)) && (!query || [item.insight?.headline, item.title, item.relevance_assessment?.business_channel].some(value => value?.toLocaleLowerCase('ru-RU').includes(query))))
      .sort((a, b) => sort === 'recent' ? (Date.parse(b.insight?.as_of || '') || 0) - (Date.parse(a.insight?.as_of || '') || 0) : (b.relevance_grade ?? 0) - (a.relevance_grade ?? 0));
  }, [feed, search, filter, sort, reading.isBookmarked, reading.isRead]);
  const shown = compact ? visible.slice(0, 3) : visible;
  return <div className={`insight-feed ${compact ? 'feed-compact' : ''}`}>
    {!compact && <div className="page-title"><div className="eyebrow">ПЕРСОНАЛЬНЫЙ ВЗГЛЯД</div><h1>Важное для бизнеса<span>.</span></h1><p>События с высокой значимостью для компании {session.company_id}. Отбор учитывает ваш профиль и роль.</p></div>}
    <DiscoveryPanel key={session.token} session={session} onEvent={onEvent} onChanged={discoveryChanged}/>
    <div className="feed-section-heading"><div><div className="eyebrow">ОТОБРАНО ДЛЯ ВАС</div><h2>{compact ? 'В фокусе вашего бизнеса' : 'Персональная лента'}<span className="feed-count">{feed?.items.length ?? '—'}</span></h2></div><button type="button" className="icon-button feed-refresh" aria-label="Обновить аналитику" onClick={() => void reload()} disabled={loading}><Icon name="refresh" size={18}/></button></div>
    <div className="feed-toolbar"><div className="filter-chips" role="group" aria-label="Фильтр аналитики">{([{ id: 'all', label: 'Все важное' }, { id: 'critical', label: 'Значимость 5/5' }, { id: 'bank', label: 'С позицией банка' }, { id: 'saved', label: 'С разбором' }, { id: 'bookmarks', label: 'Избранное' }, { id: 'unread', label: 'Непрочитанные' }] as const).map(item => <button type="button" aria-pressed={filter === item.id} className={filter === item.id ? 'active' : ''} key={item.id} onClick={() => onFilterChange(item.id)}>{item.label}</button>)}</div>{!compact && <label className="feed-sort"><span className="sr-only">Порядок карточек</span><select value={sort} onChange={e => setSort(e.target.value)}><option value="priority">По значимости</option><option value="recent">По дате анализа</option></select></label>}</div>
    <details className="feed-help"><summary>Как читать значимость и работать с лентой</summary><p><strong>5/5</strong> — подтверждённая применимость и конкретное действие. <strong>4/5</strong> — существенный фактор для деятельности компании. Оценку определяет система; она не означает размер финансового риска.</p><p>Избранное и прочтение относятся к текущей компании и роли. По возможности сохраняются в этой вкладке. Фильтры работают среди доступных событий персональной ленты.</p></details>
    {!compact && <label className="feed-search"><Icon name="search" size={18}/><span className="sr-only">Поиск по персональной ленте</span><input type="search" placeholder="Найти событие или влияние на бизнес…" value={search} onChange={e => setSearch(e.target.value)}/></label>}
    {error && <div className="insight-error" role="alert">Не удалось обновить ленту: {error} <button type="button" className="text-button" onClick={() => void reload()}>Повторить</button></div>}
    {loading && !feed && <div className="feed-skeleton" role="status" aria-label="Загружаем аналитику">{[1, 2, 3].map(x => <div className="skeleton-card" key={x}><span/><span/><span/></div>)}</div>}
    {!loading && !error && feed && feed.items.length === 0 && <div className="feed-empty"><div className="empty-icon"><Icon name="activity" size={28}/></div><h3>Сначала — важное для вас</h3><p>Пока нет событий со значимостью 4–5 из 5. Запустите подбор выше: мы проверим публикации и найдём связь с вашим бизнесом.</p><span>Сохранённые публикации доступны в разделе «Все события».</span></div>}
    {!!feed?.items.length && shown.length === 0 && <div className="feed-empty"><Icon name="search" size={26}/><h3>Под этот фильтр ничего не нашлось</h3><p>Попробуйте посмотреть все важные события.</p><button className="secondary-button" type="button" onClick={() => { onFilterChange('all'); setSearch(''); }}>Сбросить фильтры</button></div>}
    <div className="feed-cards">{shown.map((item: FeedItem, index) => {
      const reasons = (item.reasons ?? item.relevance_reasons ?? []).map(relevanceDescription).filter((text): text is string => !!text);
      return <article key={`${item.event_id}:${item.event_version}`} className={`brief-card ${index === 0 && filter === 'all' && sort === 'priority' ? 'brief-card-featured' : ''}`}>
        <div className="brief-card-meta"><span className={`priority-badge ${item.relevance_grade === 5 ? 'priority-high' : ''}`}><span className="priority-bars" aria-hidden="true"><i/><i/><i/></span>{item.relevance_grade != null ? `${item.relevance_grade}/5 · ${item.relevance_grade === 5 ? 'Максимальная значимость' : 'Высокая значимость'}` : 'Значимость не определена'}</span><div className="brief-card-tools">{!reading.isRead(item.event_id) && <span className="unread-label">Новое</span>}<button type="button" className={`bookmark-button ${reading.isBookmarked(item.event_id) ? 'active' : ''}`} aria-pressed={reading.isBookmarked(item.event_id)} aria-label={`${reading.isBookmarked(item.event_id) ? 'Убрать из избранного' : 'Добавить в избранное'}: ${item.insight?.headline || item.title || 'событие'}`} onClick={() => reading.toggleBookmark(item.event_id)}><Icon name="bookmark" size={18}/></button></div></div>
        <h3><button type="button" onClick={() => onEvent(item.event_id)}>{item.insight?.headline || item.title || 'Событие без заголовка'}</button></h3>
        {item.relevance_assessment?.business_channel && <p className="brief-impact">{item.relevance_assessment.business_channel}</p>}
        {item.relevance_assessment?.required_action && <div className="brief-action"><Icon name="arrow" size={15}/><p>{item.relevance_assessment.required_action}</p></div>}
        {reasons.length > 0 && <details className="brief-reasons"><summary>Почему это касается вас</summary>{reasons.map((reason, i) => <p key={i}>{reason}</p>)}</details>}
        <div className="brief-card-footer"><div className="brief-card-labels"><span className={item.insight?.bank_view?.status === 'found' ? 'bank-available' : ''}><Icon name={item.insight?.bank_view?.status === 'found' ? 'briefcase' : 'file'} size={13}/>{item.insight?.bank_view ? bankLabel(item.insight.bank_view.status) : item.insight ? 'Разбор сохранён' : 'Готово к разбору'}</span>{item.insight?.as_of && <span>Анализ {dateLabel(item.insight.as_of)}</span>}{item.insight?.status === 'insufficient_evidence' && <span className="warning-text">Недостаточно доказательств</span>}</div><button type="button" className="brief-open" aria-label={`Открыть разбор: ${item.insight?.headline || item.title || 'событие'}`} onClick={() => onEvent(item.event_id)}><span>{item.insight ? 'Разобрать' : 'Подробнее'}</span><Icon name="arrow" size={18}/></button></div>
      </article>;
    })}</div>
    {compact && onAll && !!feed?.items.length && <button type="button" className="feed-all" onClick={onAll}>Вся персональная аналитика <Icon name="arrow" size={17}/></button>}
    {!!feed?.limitations?.length && <details className="feed-limitations"><summary>Как работает отбор</summary><ul>{feed.limitations.map((text, i) => <li key={i}>{limitationLabel(text)}</li>)}</ul></details>}
  </div>;
}
