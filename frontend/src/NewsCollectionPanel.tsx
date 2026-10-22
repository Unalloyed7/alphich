import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react';
import { api, ApiError } from './api';
import type { DemoSession, NewsCollectionStatus, NewsSearchRun } from './types';
import './news-collection.css';

const ACTIVE = new Set<NewsSearchRun['status']>(['queued', 'collecting', 'waiting_ingestion']);
const runLabels: Record<NewsSearchRun['status'], string> = {
  queued: 'Поиск в очереди', collecting: 'Ищем и скачиваем публикации',
  waiting_ingestion: 'Создаём события из сохранённых публикаций',
  succeeded: 'Поиск завершён', partial: 'Поиск завершён с неполным результатом',
  failed: 'Поиск не завершён', cancelled: 'Поиск остановлен',
};
const stopLabels: Record<string, string> = {
  target_reached: 'Нужное количество публикаций сохранено.',
  sources_exhausted: 'В просмотренных источниках оказалось меньше новых публикаций, чем запрошено.',
  no_new_articles: 'Новых публикаций в просмотренных источниках не найдено.',
  time_limit: 'Истекло время поиска. Уже сохранённые публикации доступны ниже.',
  user_cancelled: 'Поиск остановлен. Уже сохранённые публикации доступны ниже.',
  invalid_rss_config: 'Настройки источников недоступны.',
  ingestion_failed: 'Часть публикаций не удалось преобразовать в события. Сохранённые результаты доступны ниже.',
};

function timestamp(value: number | null): string {
  return value ? new Date(value * 1000).toLocaleString('ru-RU', { dateStyle: 'short', timeStyle: 'short' }) : 'ещё нет данных';
}

function externalUrl(value: string): string | undefined {
  try {
    const url = new URL(value);
    return ['https:', 'http:'].includes(url.protocol) ? url.href : undefined;
  } catch { return undefined; }
}

export function NewsCollectionPanel({ session, onEvent, onChanged, onUnauthorized }: {
  session: DemoSession;
  onEvent: (id: string) => void;
  onChanged: () => void;
  onUnauthorized: () => void;
}) {
  const [count, setCount] = useState('10');
  const [run, setRun] = useState<NewsSearchRun | null>(null);
  const [collector, setCollector] = useState<NewsCollectionStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const mutationInFlight = useRef(false);
  const mutationVersion = useRef(0);
  const onChangedRef = useRef(onChanged);
  onChangedRef.current = onChanged;
  const previousEvents = useRef('');
  const active = run !== null && ACTIVE.has(run.status);
  const requested = Number(count);
  const validCount = count.trim() !== '' && Number.isInteger(requested) && requested >= 1 && requested <= 100;

  const message = useCallback((cause: unknown, fallback: string): string => {
    if (cause instanceof ApiError && cause.status === 401) {
      onUnauthorized();
      return 'Демосессия истекла. Подключитесь снова.';
    }
    if (cause instanceof ApiError && cause.status === 429) return 'Поиск запускался недавно. Подождите минуту перед следующим запуском.';
    if (cause instanceof ApiError && cause.status === 409) return 'Для этой компании уже выполняется поиск в другой сессии. Дождитесь его завершения.';
    return cause instanceof Error ? cause.message : fallback;
  }, [onUnauthorized]);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    void api.latestNewsSearch(session.company_id, session.token)
      .then(value => { if (alive) { setRun(value.run); setError(null); } })
      .catch(cause => { if (alive) setError(message(cause, 'Не удалось прочитать последний поиск.')); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, [session.company_id, session.token, message]);

  useEffect(() => {
    let alive = true, inFlight = false;
    const refresh = async () => {
      if (!alive || inFlight || document.visibilityState === 'hidden') return;
      inFlight = true;
      try {
        const value = await api.collectionStatus(session.token);
        if (alive) { setCollector(value); setStatusError(null); }
      } catch (cause) {
        if (alive) setStatusError(message(cause, 'Не удалось обновить состояние автосбора.'));
      } finally { inFlight = false; }
    };
    const onVisible = () => { if (document.visibilityState === 'visible') void refresh(); };
    void refresh();
    const timer = window.setInterval(() => { void refresh(); }, 15_000);
    document.addEventListener('visibilitychange', onVisible);
    return () => { alive = false; window.clearInterval(timer); document.removeEventListener('visibilitychange', onVisible); };
  }, [session.token, message]);

  useEffect(() => {
    if (!run || !active) return;
    let alive = true, inFlight = false;
    const refresh = async () => {
      if (!alive || inFlight || mutationInFlight.current || document.visibilityState === 'hidden') return;
      inFlight = true;
      const version = mutationVersion.current;
      try {
        const value = await api.newsSearch(run.id, session.token);
        if (alive && !mutationInFlight.current && version === mutationVersion.current) { setRun(value); setError(null); }
      } catch (cause) {
        if (alive && version === mutationVersion.current) setError(message(cause, 'Не удалось обновить прогресс поиска.'));
      } finally { inFlight = false; }
    };
    const onVisible = () => { if (document.visibilityState === 'visible') void refresh(); };
    const timer = window.setInterval(() => { void refresh(); }, 2_500);
    document.addEventListener('visibilitychange', onVisible);
    return () => { alive = false; window.clearInterval(timer); document.removeEventListener('visibilitychange', onVisible); };
  }, [run?.id, active, session.token, message]);

  useEffect(() => {
    if (!run) return;
    const marker = run.items.flatMap(item => item.event_id ? [item.event_id] : []).sort().join(',');
    if (marker && marker !== previousEvents.current) {
      previousEvents.current = marker;
      onChangedRef.current();
    }
  }, [run]);

  const start = async (event: FormEvent) => {
    event.preventDefault();
    if (!validCount || active || loading || mutationInFlight.current) return;
    mutationInFlight.current = true;
    mutationVersion.current += 1;
    setBusy(true); setError(null);
    try {
      const result = await api.startNewsSearch(session.company_id, requested, session.token, `news-search-${crypto.randomUUID()}`);
      setRun(result);
    } catch (cause) { setError(message(cause, 'Не удалось запустить поиск.')); }
    finally { mutationInFlight.current = false; setBusy(false); }
  };

  const stop = async () => {
    if (!run || mutationInFlight.current) return;
    mutationInFlight.current = true;
    mutationVersion.current += 1;
    setBusy(true); setError(null);
    try { setRun(await api.stopNewsSearch(run.id, session.token)); }
    catch (cause) { setError(message(cause, 'Не удалось остановить поиск.')); }
    finally { mutationInFlight.current = false; setBusy(false); }
  };

  const collectorLabel = statusError ? 'Состояние автосбора не обновлено' : collector ? ({
    running: 'Автосбор активен', stale: 'Автосбор не подтверждён свежим сигналом',
    unknown: 'Состояние автосбора пока неизвестно', disabled: 'Автосбор выключен',
  } as const)[collector.state] : 'Проверяем автосбор';

  return <section className="panel news-collection" aria-labelledby="news-collection-title">
    <div className="news-collection-heading">
      <div><div className="eyebrow">НОВЫЕ ПУБЛИКАЦИИ</div><h2 id="news-collection-title">Сбор новостей</h2></div>
      <span className={`collection-health collection-health-${statusError ? 'unknown' : collector?.state ?? 'unknown'}`} role="status">{collectorLabel}</span>
    </div>
    <p className="collection-description">Состояние круглосуточного автосбора показано выше. Для проверки можно отдельно загрузить случайные публикации из подключённых сайтов в события компании {session.company_id}.</p>
    <form className="news-collection-actions" onSubmit={event => { void start(event); }}>
      <label htmlFor="collection-count">Количество новостей
        <input id="collection-count" type="number" min="1" max="100" step="1" required value={count}
          disabled={active || busy || loading} onChange={event => setCount(event.target.value)} aria-describedby="collection-count-help" />
      </label>
      <button type="submit" className="primary-button" disabled={active || busy || loading || !validCount}>
        {busy ? 'Подождите…' : active ? 'Ищем новости…' : 'Поиск новостей'}
      </button>
      {active && <button type="button" className="ghost-button" disabled={busy} onClick={() => { void stop(); }}>Остановить</button>}
      <span id="collection-count-help" className="collection-help">От 1 до 100 · Без отбора по значимости и генерации AI-анализа</span>
    </form>
    {collector && <p className="collection-meta">
      {collector.poll_minutes ? `Проверка каждые ${collector.poll_minutes} мин.` : 'Частота проверки задана для каждого источника.'}
      {' '}Последний сигнал сборщика: {timestamp(collector.heartbeat_at)}. Новая публикация появится после ближайшей проверки и загрузки.
    </p>}
    {statusError && <p className="collection-warning" role="status">{statusError} Показано последнее известное состояние.</p>}
    {error && <p className="collection-error" role="alert">{error}</p>}
    {run && <div className="collection-progress" aria-live="polite">
      <div className="collection-progress-heading"><strong>{runLabels[run.status]}</strong><span>{run.collected_count} / {run.requested_count} сохранено</span></div>
      <progress value={run.collected_count} max={run.requested_count} aria-label="Сохранённые публикации" />
      <p className="collection-meta">Источников проверено: {run.checked_sources} из {run.total_sources}.
        {' '}Скачано сейчас: {run.newly_downloaded}; скачано ранее: {run.reused}.
        {' '}Ранее добавленные в эту компанию статьи повторно не добавляются.</p>
      {run.stop_reason && <p className="collection-result-note">{stopLabels[run.stop_reason] ?? (run.status === 'partial'
        ? 'Удалось сохранить часть запрошенных публикаций. Результаты доступны ниже.'
        : run.status === 'failed' ? 'Поиск завершился с ошибкой. Сохранённые результаты доступны ниже.' : 'Результаты поиска сохранены.')}</p>}
      {run.errors.length > 0 && <details className="collection-diagnostics"><summary>Ошибки источников: {run.errors.length}</summary><ul>{run.errors.map((value, index) => <li key={`${index}:${value}`}>{value}</li>)}</ul></details>}
      {run.items.length > 0 && <details className="collection-results" open><summary>Найденные публикации ({run.items.length})</summary>
        <ul>{run.items.map(item => <li key={item.article_id}>
          {item.event_id ? <button type="button" className="collection-event-link" onClick={() => onEvent(item.event_id!)}>{item.title}</button>
            : <span className="collection-pending-title">{item.title}</span>}
          <div className="collection-item-meta"><span>{item.event_id ? 'Событие готово' : 'Публикация сохранена · готовим событие'}</span>
            {externalUrl(item.url) && <a href={externalUrl(item.url)} target="_blank" rel="noopener noreferrer">Первоисточник ↗</a>}</div>
        </li>)}</ul>
      </details>}
    </div>}
    {collector && <details className="collection-sources"><summary>Источники и расписание ({collector.feeds.length})</summary><ul>{collector.feeds.map(feed => <li key={feed.id}>
      <div className="collection-feed-heading"><strong>{feed.id}</strong><span>каждые {Math.round(feed.poll_seconds / 60 * 10) / 10} мин.</span></div>
      <p>Проверка: {timestamp(feed.last_checked_at)} · Успешно: {timestamp(feed.last_success_at)} · Следующая: {timestamp(feed.next_poll_at)}</p>
      {feed.last_error && <p className="collection-feed-error">Последняя ошибка: {feed.last_error} · Ошибок подряд: {feed.failures}</p>}
    </li>)}</ul></details>}
  </section>;
}
