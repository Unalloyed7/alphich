import { useEffect, useRef, useState } from 'react';
import { analysisApi, ApiError } from './api';
import type { DemoSession } from './types';
import type { LlmStatus } from './analysis';
import { Icon } from './Icon';

export interface DiscoveryRun {
  id: string; company_id: string; role: string; status: string; stage: string;
  stop_reason: string | null; target: number; min_grade: number; max_candidates: number;
  cost_rub: string; cost_unknown: boolean;
  stats: { completed: number; checked: number; below_threshold: number; failed: number;
    scanned?: number; preselected?: number; feeds_checked: number; reused: number; previously_checked?: number };
  items: Array<{ event_id: string; title: string; state: string; grade: number | null;
    insight_id?: string; error?: string }>;
}

const activeStatuses = new Set(['collecting', 'waiting_ingestion', 'analyzing', 'stopping']);
const stages: Record<string, string> = {
  queued: 'Запуск поиска', collecting_sources: 'Собираем публикации из источников',
  waiting_for_articles: 'Сохраняем собранные статьи', checking_candidates: 'Отбираем кандидатов',
  extracting_facts: 'Извлекаем и проверяем факты', generating_insight: 'Строим и проверяем анализ',
  waiting_for_budget_reservation: 'Ждём завершения другого запроса к модели',
  stop_requested: 'Останавливаемся после текущего запроса', finished: 'Поиск завершён',
};
const reasons: Record<string, string> = {
  target_reached: 'Нужное количество карточек готово.',
  sources_exhausted: 'Новых подходящих публикаций в просмотренных источниках нет. Ранее проверенные новости повторно не отбираются.',
  candidate_limit: 'Достигнут предел проверки кандидатов. Подходящих новостей оказалось меньше цели.',
  budget_exceeded: 'Достигнут доступный денежный лимит. Дальнейшие запросы к моделям остановлены.',
  time_limit: 'Закончилось время этого запуска. Уже готовые результаты сохранены.',
  user_cancelled: 'Поиск остановлен. Уже готовые результаты сохранены.',
  budget_day_changed: 'Начался новый бюджетный день. Для продолжения нужен новый запуск.',
  company_profile_changed: 'Профиль компании изменился. Запустите поиск для обновлённых данных.',
  company_unavailable: 'Компания больше недоступна.', llm_not_configured: 'Подключение к моделям недоступно.',
  invalid_rss_config: 'Не удалось загрузить настройки источников.',
};
const itemStates: Record<string, string> = { completed: 'Карточка готова', below_threshold: 'Ниже порога',
  failed: 'Обработка не завершена', pending: 'В очереди', waiting_facts: 'Проверяем факты', waiting_analysis: 'Готовим анализ' };

export function DiscoveryPanel({ session, onEvent, onChanged }: {
  session: DemoSession; onEvent: (id: string) => void; onChanged: () => void;
}) {
  const [run, setRun] = useState<DiscoveryRun | null>(null);
  const [status, setStatus] = useState<LlmStatus | null>(null);
  const [available, setAvailable] = useState(false);
  const [minGrade, setMinGrade] = useState<4 | 5>(4);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const previousResults = useRef('');
  const active = run !== null && activeStatuses.has(run.status);

  useEffect(() => {
    let alive = true;
    void Promise.allSettled([analysisApi.discoveryLatest(session.company_id, session.token), analysisApi.status(session.token)])
      .then(([discovery, llm]) => {
        if (!alive) return;
        if (llm.status === 'fulfilled') setStatus(llm.value);
        if (discovery.status === 'fulfilled') { setRun(discovery.value.run); setAvailable(true); }
        else setError(discovery.reason instanceof ApiError && discovery.reason.status === 404 ? 'Подбор новостей недоступен на этом сервере.' : 'Не удалось загрузить статус подбора. Попробуйте обновить страницу.');
        if (llm.status === 'rejected') setError('Не удалось проверить доступность аналитики. Попробуйте обновить страницу.');
      });
    return () => { alive = false; };
  }, [session.company_id, session.token]);

  useEffect(() => {
    if (!run || !active) return;
    let alive = true, fetching = false;
    const poll = async () => {
      if (fetching) return;
      fetching = true;
      try {
        const [value, llm] = await Promise.all([analysisApi.discovery(run.id, session.token), analysisApi.status(session.token)]);
        if (alive) { setRun(value); setStatus(llm); setError(null); }
      } catch (e) { if (alive) setError(e instanceof Error ? e.message : 'Не удалось обновить прогресс'); }
      finally { fetching = false; }
    };
    const timer = window.setInterval(() => void poll(), 2500);
    return () => { alive = false; window.clearInterval(timer); };
  }, [run?.id, active, session.token]);

  useEffect(() => {
    if (!run) return;
    const marker = `${run.id}:${run.stats.completed}:${run.status}`;
    if (marker !== previousResults.current && (run.stats.completed > 0 || !activeStatuses.has(run.status))) {
      previousResults.current = marker;
      onChanged();
      void analysisApi.status(session.token).then(setStatus).catch(() => {});
    }
  }, [run, onChanged, session.token]);

  const start = async () => {
    setBusy(true); setError(null);
    try { setRun(await analysisApi.startDiscovery(session.company_id, session.token, minGrade)); }
    catch (e) { setError(e instanceof Error ? e.message : 'Не удалось запустить поиск'); }
    finally { setBusy(false); }
  };
  const stop = async () => {
    if (!run) return;
    setBusy(true); setError(null);
    try { setRun(await analysisApi.stopDiscovery(run.id, session.token)); }
    catch (e) { setError(e instanceof Error ? e.message : 'Не удалось остановить поиск'); }
    finally { setBusy(false); }
  };

  return <section className="discovery-panel" aria-label="Подбор новостей для компании">
    <div className="discovery-top"><span className="discovery-icon"><Icon name="activity" size={23}/></span><div><h3>{active ? 'Ищем важное для вашего бизнеса' : 'Ваш следующий важный инсайт'}</h3><p>{active ? 'Проверяем публикации и готовим персональные разборы.' : 'Подберём и разберём до 10 новых событий для вашей компании.'}</p></div><button type="button" className="insight-primary" disabled={active || busy || !available || !status?.configured} onClick={() => void start()}>{busy ? 'Подождите…' : active ? <><span className="spinner tiny"/> Подбираем…</> : <>Найти важное <Icon name="arrow" size={16}/></>}</button></div>
    <div className="discovery-bottom"><span><Icon name="briefcase" size={13}/> {session.company_id} <span className="separator">/</span> {session.role.toUpperCase()}</span><details className="discovery-options"><summary>Настройки подбора</summary><div><label>Значимость <select value={minGrade} disabled={active || busy} onChange={e => setMinGrade(Number(e.target.value) as 4 | 5)}><option value={4}>4–5 из 5</option><option value={5}>Только 5 из 5</option></select></label><p>Проверяем до 40 кандидатов. Подходящих новостей может оказаться меньше десяти. Поиск продолжится после закрытия вкладки.</p>{status && <p>Общий бюджет на {status.budget_day_utc} (UTC): {status.spent_rub} ₽ потрачено · {status.reserved_rub} ₽ зарезервировано · лимит {status.daily_limit_rub} ₽.</p>}</div></details></div>
    <p className="discovery-cost-note">Запуск использует общий бюджет аналитики.</p>
    {status && !status.configured && <p className="insight-warning">Подбор недоступен: аналитический сервис ещё не настроен. Сохранённые разборы можно просматривать.</p>}
    {error && <p className="insight-error" role="alert">{error}</p>}
    {run && <div className="discovery-progress" aria-live="polite"><div className="discovery-progress-heading"><strong>{run.stats.completed} <span>/ {run.target} разборов</span></strong>{active && <button type="button" className="text-button" disabled={busy || run.status === 'stopping'} onClick={() => void stop()}>{run.status === 'stopping' ? 'Останавливаем…' : 'Остановить подбор'}</button>}</div><progress value={run.stats.completed} max={run.target} aria-label="Готовые аналитические карточки"/><p>{stages[run.stage] ?? 'Обрабатываем публикации'}</p>{run.stop_reason && <p className="insight-muted">{reasons[run.stop_reason] ?? 'Поиск остановлен после ошибки. Готовые результаты сохранены.'}</p>}<details className="discovery-run-details"><summary>Подробности подбора и расходы</summary><p>Источников: {run.stats.feeds_checked} · просмотрено статей: {run.stats.scanned ?? 0} · оценено: {run.stats.checked} · ниже порога: {run.stats.below_threshold} · ошибок: {run.stats.failed} · ранее проверенных: {run.stats.previously_checked ?? 0}.</p><p>Расход: {run.cost_rub} ₽{run.cost_unknown ? ' · стоимость части запросов не подтверждена' : ''}. Использовано готовых карточек: {run.stats.reused}.</p>{run.items.length > 0 && <ul className="discovery-results">{run.items.map(item => <li key={item.event_id}><button type="button" className="text-button" onClick={() => onEvent(item.event_id)}>{item.title}</button><span>{item.grade === null ? 'Оценка ожидается' : `${item.grade}/5`} · {itemStates[item.state] ?? 'В обработке'}</span></li>)}</ul>}</details></div>}
  </section>;
}
