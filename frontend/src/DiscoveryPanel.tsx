import { useEffect, useRef, useState } from 'react';
import { analysisApi } from './api';
import type { DemoSession } from './types';
import type { LlmStatus } from './analysis';

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
  const [minGrade, setMinGrade] = useState<4 | 5>(4);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const previousResults = useRef('');
  const active = run !== null && activeStatuses.has(run.status);

  useEffect(() => {
    let alive = true;
    void Promise.all([analysisApi.discoveryLatest(session.company_id, session.token), analysisApi.status(session.token)])
      .then(([value, llm]) => { if (alive) { setRun(value.run); setStatus(llm); } })
      .catch(e => { if (alive) setError(e instanceof Error ? e.message : 'Не удалось прочитать статус поиска'); });
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

  return <section className="discovery-panel" aria-label="Автономный поиск новостей">
    <h3>Найти новости для {session.company_id}</h3>
    <p>Проверим источники и разберём ещё не проверенные публикации для этой компании и роли. Ранее сохранённые карточки останутся в ленте. Поиск продолжится, если закрыть вкладку.</p>
    <div className="discovery-actions">
      <label>Отбирать <select value={minGrade} disabled={active || busy} onChange={e => setMinGrade(Number(e.target.value) as 4 | 5)}>
        <option value={4}>4–5 из 5</option><option value={5}>Только 5 из 5</option>
      </select></label>
      <button type="button" className="insight-primary" disabled={active || busy || !status?.configured} onClick={() => void start()}>
        {busy ? 'Подождите…' : active ? 'Поиск выполняется' : 'Найти и разобрать 10 новых новостей'}
      </button>
      {active && <button type="button" className="insight-link" disabled={busy || run?.status === 'stopping'} onClick={() => void stop()}>Остановить</button>}
    </div>
    {status && <p className="insight-muted">Общий бюджет на UTC-день: {status.spent_rub} ₽ учтено, {status.reserved_rub} ₽ зарезервировано, лимит {status.daily_limit_rub} ₽.
      {' '}Проверяем до 40 кандидатов за запуск; цель не гарантирует наличие десяти подходящих новостей.</p>}
    {status && !status.configured && <p className="insight-error">Модели пока не настроены — запуск недоступен.</p>}
    {error && <p className="insight-error" role="alert">{error}</p>}
    {run && <div className="discovery-progress" aria-live="polite">
      <strong>{run.stats.completed} / {run.target} карточек · порог {run.min_grade}/5</strong>
      <progress value={run.stats.completed} max={run.target} aria-label="Готовые аналитические карточки" />
      <p>{stages[run.stage] ?? 'Обработка поиска'}</p>
      <p className="insight-muted">Источников опрошено: {run.stats.feeds_checked}; просмотрено статей: {run.stats.scanned ?? 0};
        {' '}оценено: {run.stats.checked}; ниже порога: {run.stats.below_threshold}; с ошибкой: {run.stats.failed};
        {' '}пропущено ранее проверенных: {run.stats.previously_checked ?? 0}.</p>
      <p className="insight-muted">Расход этого запуска: {run.cost_rub} ₽{run.cost_unknown ? ' · стоимость части запросов ещё не подтверждена' : ''}.
        {run.stats.reused > 0 ? ` Повторно использовано готовых карточек: ${run.stats.reused}.` : ''}</p>
      {run.stop_reason && <p>{reasons[run.stop_reason] ?? 'Поиск остановлен после ошибки обработки. Уже готовые результаты сохранены.'}</p>}
      {run.items.length > 0 && <details><summary>Публикации и результаты проверки</summary>
        <ul className="discovery-results">{run.items.map(item => <li key={item.event_id}>
          <button type="button" className="insight-link" onClick={() => onEvent(item.event_id)}>{item.title}</button>
          <span>{item.grade === null ? 'Оценка ожидается' : `${item.grade}/5`} · {itemStates[item.state] ?? 'В обработке'}</span>
        </li>)}</ul>
      </details>}
    </div>}
  </section>;
}
