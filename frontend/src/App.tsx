import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from 'react';
import { api, analysisApi, ApiError } from './api';
import { EventInsight, InsightFeed } from './InsightUI';
import { CompanyAnalysis } from './CompanyAnalysis';
import { Icon, type IconName } from './Icon';
import { mergeRecentEvents } from './recentEvents';
import type { CompanyProfile, CompanySummary, DemoSession, Event, Job, PublicationPayload, Role, SourceDetail } from './types';

type Screen = 'dashboard' | 'publish' | 'events' | 'insights' | 'profile';
const SESSION_STORAGE_KEY = 'alphich.demo-session.v1';
const EVENTS_PAGE_SIZE = 20;
const RSS_EVENT_REFRESH_MS = 15_000;

function readSession(): DemoSession | null {
  try {
    const raw = sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (!raw) return null;
    const value = JSON.parse(raw) as DemoSession;
    return value.token && value.expires_at > Date.now() / 1000 ? value : null;
  } catch { return null; }
}

function formattedDate(date: string | null): string {
  if (!date) return 'Не указана';
  const time = new Date(date);
  return Number.isNaN(time.getTime()) ? date : time.toLocaleString('ru-RU', { dateStyle: 'medium', timeStyle: 'short' });
}

function jobLabel(job: Job): string {
  return ({ queued: 'В очереди', running: 'Обрабатывается', succeeded: 'Готово', failed: 'Ошибка' } as Record<Job['status'], string>)[job.status];
}

function Notice({ kind = 'info', children, onClose }: { kind?: 'info' | 'error' | 'success'; children: ReactNode; onClose?: () => void }) {
  return <div className={`notice notice-${kind}`} role={kind === 'error' ? 'alert' : 'status'}>
    <span className="notice-symbol"><Icon name={kind === 'error' ? 'alert' : kind === 'success' ? 'check' : 'activity'} size={17}/></span>
    <div>{children}</div>
    {onClose && <button type="button" className="icon-button" onClick={onClose} aria-label="Закрыть"><Icon name="close" size={17}/></button>}
  </div>;
}

function StatusPill({ status }: { status: Job['status'] | 'raw' | 'analyzed' | 'insight' | 'insufficient' }) {
  const labels: Record<string, string> = { queued: 'В очереди', running: 'В работе', succeeded: 'Сохранено', failed: 'Ошибка', raw: 'Ожидает разбора', analyzed: 'Факты извлечены', insight: 'Разбор готов', insufficient: 'Недостаточно данных' };
  return <span className={`status-pill status-${status}`}><span className="status-dot"/>{labels[status]}</span>;
}

const NAV: { key: Screen; label: string; icon: IconName }[] = [
  { key: 'dashboard', label: 'Обзор', icon: 'grid' },
  { key: 'insights', label: 'Моя аналитика', icon: 'activity' },
  { key: 'events', label: 'Все события', icon: 'list' },
  { key: 'profile', label: 'Моя компания', icon: 'briefcase' },
  { key: 'publish', label: 'Добавить новость', icon: 'plus' },
];

function SessionGate({ onConnect, busy, error, initialSession, onCancel }: {
  onConnect: (company: string, role: Role) => Promise<void>; busy: boolean; error: string | null; initialSession?: DemoSession | null; onCancel?: () => void;
}) {
  const [companyId, setCompanyId] = useState(initialSession?.company_id ?? 'DEMO_A');
  const [role, setRole] = useState<Role>(initialSession?.role ?? 'cfo');
  const submit = (e: FormEvent) => { e.preventDefault(); void onConnect(companyId.trim(), role); };
  return <div className="onboarding">
    <div className="onboarding-art"><div className="onboarding-wordmark">alphich<span>●</span></div><div className="art-orbit orbit-one"/><div className="art-orbit orbit-two"/><div className="art-logo">α</div><div className="art-caption"><span>ОТ СОБЫТИЯ — К РЕШЕНИЮ</span><strong>Большая картина.<br/>Ваш контекст.</strong><p>Новости × ваш бизнес × аналитика банка</p></div></div>
    <div className="onboarding-content">
      <div className="eyebrow"><span className="eyebrow-line"/> ПЕРСОНАЛЬНАЯ БИЗНЕС-АНАЛИТИКА</div>
      <h1>Новости, которые<br/><em>имеют значение.</em></h1>
      <p className="onboarding-desc">Поймите, как события влияют на ваш бизнес. Главное, возможные последствия и следующие шаги — с опорой на источники.</p>
      <form onSubmit={submit} className="connect-form">
        <div className="form-title">Откройте свой бизнес-контекст</div>
        <label htmlFor="company-id">Идентификатор компании</label>
        <input id="company-id" value={companyId} onChange={(e) => setCompanyId(e.target.value)} placeholder="DEMO_A или U_9250" required maxLength={100} spellCheck={false}/>
        <div className="quick-company"><span>Примеры:</span>{['DEMO_A', 'DEMO_B', 'U_9250'].map((id) => <button key={id} type="button" onClick={() => setCompanyId(id)}>{id}</button>)}</div>
        <label htmlFor="company-role">Ваша роль</label>
        <select id="company-role" value={role} onChange={(e) => setRole(e.target.value as Role)}><option value="cfo">Финансовый директор (CFO)</option><option value="ceo">Генеральный директор (CEO)</option></select>
        {error && <Notice kind="error">{error}</Notice>}
        <button type="submit" className="primary-button connect-button" disabled={busy || !companyId.trim()}>{busy ? 'Подключение…' : <>Перейти к аналитике <Icon name="arrow"/></>}</button>
      </form>
      {onCancel && <button type="button" className="text-button session-cancel" onClick={onCancel}>Вернуться к обзору</button>}
      <p className="helper-message">Демонстрационный доступ к обезличенному профилю. Если компания недоступна, уточните её идентификатор у команды проекта.</p>
    </div>
  </div>;
}

function PublishForm({ onPublish, busy }: { onPublish: (data: PublicationPayload) => Promise<boolean>; busy: boolean }) {
  const [title, setTitle] = useState('');
  const [text, setText] = useState('');
  const [url, setUrl] = useState('');
  const [date, setDate] = useState('');
  const [message, setMessage] = useState<string | null>(null);
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (text.trim().length < 30 || title.trim().length < 6) return;
    const payload: PublicationPayload = { title: title.trim(), text: text.trim() };
    if (url.trim()) payload.source_url = url.trim();
    if (date) payload.published_at = new Date(date).toISOString();
    const ok = await onPublish(payload);
    if (ok) { setMessage('Публикация отправлена. Статус обработки отображается в панели задач.'); setTitle(''); setText(''); setUrl(''); setDate(''); }
  };
  return <div className="panel publisher-panel">
    <form onSubmit={(e) => { void submit(e); }}>
      <div className="panel-heading"><div><div className="eyebrow">НОВЫЕ ДАННЫЕ</div><h2>Добавить публикацию</h2><p>Добавьте текст новости, чтобы сохранить событие и затем оценить его влияние на вашу компанию.</p></div><span className="panel-heading-icon"><Icon name="file" size={25}/></span></div>
      {message && <Notice kind="success" onClose={() => setMessage(null)}>{message}</Notice>}
      <div className="field"><label htmlFor="news-title">Заголовок <span>*</span></label><input id="news-title" value={title} onChange={e => { setTitle(e.target.value); setMessage(null); }} minLength={6} maxLength={300} required placeholder="Например: Банк России сохранил ключевую ставку"/></div>
      <div className="field"><label htmlFor="news-body">Текст публикации <span>*</span></label><textarea id="news-body" value={text} onChange={e => { setText(e.target.value); setMessage(null); }} minLength={30} maxLength={100000} required rows={9} placeholder="Вставьте полный текст публикации или доступный фрагмент (от 30 символов)…"/><div className="field-help"><span>Минимум 30 символов</span><span>{text.length.toLocaleString('ru-RU')} / 100 000</span></div></div>
      <div className="form-grid"><div className="field"><label htmlFor="news-url">Ссылка на источник <span className="optional">необязательно</span></label><input id="news-url" type="url" value={url} onChange={e => setUrl(e.target.value)} placeholder="https://example.org/news"/></div><div className="field"><label htmlFor="news-date">Дата публикации <span className="optional">необязательно</span></label><input id="news-date" type="datetime-local" value={date} onChange={e => setDate(e.target.value)}/></div></div>
      <div className="form-footer"><div className="footer-note"><Icon name="alert" size={16}/><span>Ссылка сохраняется как непроверенная; загрузки статьи по URL пока нет. Дата — локальное время устройства.</span></div><button className="primary-button" type="submit" disabled={busy || title.trim().length < 6 || text.trim().length < 30}>{busy ? 'Отправка…' : <>Отправить на обработку <Icon name="send" size={17}/></>}</button></div>
    </form>
  </div>;
}

function EventCard({ item, onOpen, insightStatus }: { item: Event; onOpen: (event: Event) => void; insightStatus?: string }) {
  return <button className="event-card" type="button" onClick={() => onOpen(item)}>
    <div className="event-card-top"><span className="event-tag">СОБЫТИЕ</span><span className="muted small">{formattedDate(item.published_at)}</span></div>
    <h3>{item.title}</h3>
    <p>Источник сохранён · {item.sources.length} {item.sources.length === 1 ? 'публикация' : 'публикаций'}</p>
    <div className="event-card-bottom"><StatusPill status={insightStatus ? (insightStatus === 'insufficient_evidence' ? 'insufficient' : 'insight') : item.extraction_status === 'extracted' || item.extraction_status === 'partially_extracted' ? 'analyzed' : 'raw'}/><span className="view-action">Открыть <Icon name="arrow" size={16}/></span></div>
  </button>;
}

function EventsList({ items, insightsByEvent, loading, onReload, onOpen, onPublish, hasMore = false, moreLoading = false, onMore }: {
  insightsByEvent: Record<string, string>;
  items: Event[];
  loading: boolean;
  onReload: () => void;
  onOpen: (event: Event) => void;
  onPublish: () => void;
  hasMore?: boolean;
  moreLoading?: boolean;
  onMore?: () => void;
}) {
  const [search, setSearch] = useState('');
  const normalized = search.trim().toLocaleLowerCase('ru-RU');
  const visible = items.filter(item => !normalized || [item.title, ...item.sources.map(source => source.title)]
    .some(part => part.toLocaleLowerCase('ru-RU').includes(normalized)));
  return <div className="panel event-panel">
    <div className="section-header"><div><div className="eyebrow">ЖУРНАЛ СОБЫТИЙ</div><h2>Обработанные публикации</h2></div><button type="button" className="ghost-button" onClick={onReload} disabled={loading || moreLoading}><Icon name="refresh" size={16}/> Обновить</button></div>
    {items.length > 0 && <div className="event-toolbar">
      <label htmlFor="event-search">Поиск по загруженным событиям</label>
      <input id="event-search" type="search" value={search} onChange={e => setSearch(e.target.value)} placeholder="Название публикации…" autoComplete="off"/>
      <span className="event-counter" aria-live="polite">Найдено: {visible.length} из {items.length}{hasMore ? ' · Есть более старые' : ''}</span>
    </div>}
    {loading && items.length === 0 ? <div className="empty-state"><span className="spinner"/> Загружаем события…</div> : items.length === 0 ?
      <div className="empty-state"><div className="empty-icon"><Icon name="file" size={27}/></div><h3>Пока нет событий</h3><p>Добавьте первую публикацию. После обработки она появится здесь автоматически.</p><button className="secondary-button" type="button" onClick={onPublish}><Icon name="plus" size={16}/> Добавить публикацию</button></div> : visible.length === 0 ?
      <div className="empty-state"><h3>Совпадений нет</h3><p>Поиск выполняется только среди загруженных событий. Можно загрузить более старые.</p><button type="button" className="ghost-button" onClick={() => setSearch('')}>Сбросить поиск</button></div> :
      <div className="event-list">{visible.map(item => <EventCard key={item.id} item={item} onOpen={onOpen} insightStatus={insightsByEvent[item.id]}/>)}</div>}
    {hasMore && onMore && <div className="event-pagination"><button className="secondary-button" type="button" disabled={loading || moreLoading} onClick={onMore}>{moreLoading ? 'Загружаем…' : 'Показать ещё события'}</button></div>}
  </div>;
}

function JobList({ ids, jobs, onEvent }: { ids: string[]; jobs: Record<string, Job>; onEvent: (id: string) => void }) {
  return <div className="panel jobs-panel"><div className="section-header"><div><div className="eyebrow">ОБРАБОТКА ПУБЛИКАЦИЙ</div><h2>Статус обработки</h2></div><span className="count-badge">{ids.length}</span></div>
    {ids.length === 0 ? <div className="job-empty">После добавления новости здесь появится статус её обработки.</div> :
      <div className="job-list">{ids.slice(0, 6).map(id => { const job = jobs[id]; return <div className="job-item" key={id}>
        <div className={`job-status-icon ${job?.status ?? 'queued'}`}><Icon name={job?.status === 'succeeded' ? 'check' : job?.status === 'failed' ? 'alert' : 'clock'} size={18}/></div>
        <div className="job-details"><span className="job-title">{job ? jobLabel(job) : 'Загружаем статус…'}</span>{job?.source_title && <span className="job-source-title" title={job.source_title}>{job.source_title}</span>}{job?.error && <span className="job-error">Код ошибки: {job.error}</span>}</div>
        {job?.status === 'succeeded' && job.result_id ? <button type="button" className="job-open" onClick={() => onEvent(job.result_id!)} aria-label="Открыть событие"><Icon name="arrow" size={17}/></button> : job?.status === 'running' || job?.status === 'queued' ? <span className="spinner tiny"/> : null}
      </div>; })}</div>}
    <div className="jobs-note"><span className="pulse-circle"/> Статусы обновляются автоматически</div>
  </div>;
}

function Profile({ profile }: { profile: CompanyProfile | null }) {
  if (!profile) return <div className="panel loading-panel">Загрузка профиля компании…</div>;
  const products = Object.entries(profile.products ?? {});
  const productLabels: Record<string, string> = { CL_PRODUCT_RKO_MARK: 'Расчётно-кассовое обслуживание', CL_PRODUCT_GAR_MARK: 'Банковские гарантии', CL_PRODUCT_LOA_MARK: 'Кредитные продукты', CL_PRODUCT_LEA_MARK: 'Лизинг', CL_PRODUCT_DEP_MARK: 'Депозиты', CL_PRODUCT_ZPP_MARK: 'Зарплатный проект', FACTORING_MARK: 'Факторинг' };
  return <div className="panel profile-panel">
    <div className="panel-heading"><div><div className="eyebrow">ДАННЫЕ КОМПАНИИ</div><h2>Профиль {profile.id}</h2><p>Обезличенные исходные данные. Показатели, единицы и ограничения не домысливаются.</p></div><span className="panel-heading-icon"><Icon name="briefcase" size={25}/></span></div>
    <div className="profile-facts"><div><span>Основная деятельность</span><strong>{profile.industry.okved_name || 'Не указана'}</strong><small>ОКВЭД {profile.industry.okved_code || '—'}</small></div><div><span>Регион регистрации</span><strong>{profile.region_registration || 'Не указан'}</strong><small>{profile.city_registration || 'Город не указан'}</small></div><div><span>Выручка (исходное значение)</span><strong>{profile.revenue_raw || 'Нет данных'}</strong><small>{profile.revenue_unit || 'Единица измерения не подтверждена'}</small></div><div><span>Тип клиента</span><strong>{profile.client_type || 'Не указан'}</strong><small>Версия профиля: {profile.profile_version.slice(0, 14)}</small></div></div>
    <div className="profile-column-grid"><div><h3>Банковские продукты</h3>{products.length ? <div className="feature-tags">{products.map(([key, value]) => <span className={`feature-tag ${value === true ? 'active' : ''}`} key={key}>{productLabels[key] ?? key}: {value === null ? 'неизвестно' : typeof value === 'boolean' ? value ? 'есть' : 'нет' : String(value)}</span>)}</div> : <p className="muted">Нет данных</p>}</div><div><h3>Страны в данных ФТС</h3><p>{profile.fts_countries?.length ? profile.fts_countries.join(', ') : 'Нет подтверждённого списка'}</p><h3>Доказуемые признаки</h3><p>{profile.features?.length ?? 0} идентификаторов с привязкой к версии данных</p></div></div>
    {profile.quality_flags.length > 0 && <Notice kind="info"><strong>Особенности данных:</strong> {profile.quality_flags.join(', ')}</Notice>}
    <div className="limitations"><h3>Ограничения интерпретации</h3>{profile.limitations.map((entry, index) => <p key={index}>— {entry}</p>)}</div>
  </div>;
}

function EventDrawer({ session, event, insightStatus, onInsightReady, source, loading, error, onClose, onSource }: {
  session: DemoSession; event: Event; insightStatus?: string; onInsightReady: (id: string, status: string) => void;
  source: SourceDetail | null; loading: boolean; error: string | null; onClose: () => void; onSource: (id: string) => void;
}) {
  const [tab, setTab] = useState<'analysis' | 'sources'>('analysis');
  const [activeSourceId, setActiveSourceId] = useState<string | null>(null);
  const sourcePanelRef = useRef<HTMLDivElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  useEffect(() => {
    const previousFocus = document.activeElement as HTMLElement | null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    dialogRef.current?.querySelector<HTMLButtonElement>('button')?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.stopPropagation(); closeRef.current(); }
      if (e.key !== 'Tab') return;
      const focusable = [...(dialogRef.current?.querySelectorAll<HTMLElement>('button:not(:disabled), a[href], input, textarea, select, summary, [tabindex="0"]') ?? [])].filter(el => el.getClientRects().length > 0);
      const first = focusable[0], last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last?.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first?.focus(); }
    };
    window.addEventListener('keydown', onKey);
    return () => { window.removeEventListener('keydown', onKey); document.body.style.overflow = previousOverflow; if (previousFocus?.isConnected) previousFocus.focus(); };
  }, []);
  useEffect(() => {
    if (activeSourceId && tab === 'sources') sourcePanelRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [activeSourceId, tab, loading, source, error]);
  return <div className="drawer-backdrop" onMouseDown={e => { if (e.target === e.currentTarget) onClose(); }}>
    <div ref={dialogRef} className="drawer" role="dialog" aria-modal="true" aria-labelledby="event-drawer-title">
      <header className="drawer-head"><span className="eyebrow">СОБЫТИЕ В ВАШЕМ КОНТЕКСТЕ</span><button className="icon-button" type="button" onClick={onClose} aria-label="Закрыть событие"><Icon name="close" size={22}/></button></header>
      <div className="drawer-scroll"><div className="event-drawer-meta"><StatusPill status={insightStatus ? (insightStatus === 'insufficient_evidence' ? 'insufficient' : 'insight') : event.extraction_status === 'extracted' || event.extraction_status === 'partially_extracted' ? 'analyzed' : 'raw'}/><span>{formattedDate(event.published_at)}</span></div><h2 id="event-drawer-title">{event.title}</h2>
        <div className="drawer-tabs" role="group" aria-label="Разделы события"><button type="button" className={tab === 'analysis' ? 'active' : ''} aria-pressed={tab === 'analysis'} onClick={() => setTab('analysis')}><Icon name="activity" size={17}/> Влияние на бизнес</button><button type="button" className={tab === 'sources' ? 'active' : ''} aria-pressed={tab === 'sources'} onClick={() => setTab('sources')}><Icon name="file" size={17}/> Первоисточники <span>{event.source_ids.length}</span></button></div>
        <div hidden={tab !== 'analysis'}><EventInsight session={session} eventId={event.id} onInsightReady={onInsightReady}/></div>
        {tab === 'sources' && <section className="event-sources"><h3>Что произошло</h3><p className="muted">Исходная публикация и её метаданные. Откройте источник, чтобы прочитать полный сохранённый текст.</p><div className="source-row">{event.source_ids.map((id, index) => <button className={`source-button ${activeSourceId === id ? 'active' : ''}`} type="button" key={id} aria-expanded={activeSourceId === id} aria-controls="active-source-panel" onClick={() => { setActiveSourceId(id); onSource(id); }}><Icon name="file" size={18}/><span>{event.sources[index]?.title || `Источник ${index + 1}`}</span><Icon name="chevron" size={16}/></button>)}</div>
        {activeSourceId && <div id="active-source-panel" ref={sourcePanelRef} className="source-panel" role="region" aria-label="Текст первоисточника" aria-live="polite">{loading && <div className="source-loading"><span className="spinner"/> Загружаем текст источника…</div>}{error && <Notice kind="error">{error}</Notice>}{source && source.id === activeSourceId && <section className="source-reader"><div className="source-reader-head"><div className="eyebrow">ОРИГИНАЛ ПУБЛИКАЦИИ</div><h3>{source.title}</h3><span className="muted small">Получен: {formattedDate(source.received_at)}</span></div><p className="source-full-text">{source.text}</p><div className="source-footer"><span><Icon name="alert" size={15}/>{source.url_verified ? 'Текст загружен по ссылке' : 'Ссылка не верифицирована'}</span>{source.url && /^https?:\/\//i.test(source.url) && <a href={source.url} target="_blank" rel="noopener noreferrer"><Icon name="link" size={16}/> Открыть оригинал</a>}</div></section>}</div>}
        <details className="event-metadata-details"><summary>Даты и сведения о событии</summary><div className="drawer-metadata"><div><span>Дата публикации</span><strong>{formattedDate(event.published_at)}</strong></div><div><span>Дата события</span><strong>{event.event_at ? formattedDate(event.event_at) : 'Не установлена'}</strong></div><div><span>ID события</span><strong className="mono wrap-text">{event.id}</strong></div><div><span>Статус извлечения</span><strong>{event.extraction_status}</strong></div></div></details></section>}
      </div>
    </div>
  </div>;
}

function App() {
  const [session, setSession] = useState<DemoSession | null>(readSession);
  const activeSessionToken = useRef(session?.token ?? null);
  activeSessionToken.current = session?.token ?? null;
  const [screen, setScreen] = useState<Screen>('dashboard');
  const [sessionBusy, setSessionBusy] = useState(false);
  const [sessionError, setSessionError] = useState<string | null>(null);
  const [health, setHealth] = useState<'checking' | 'online' | 'offline'>('checking');
  const [profile, setProfile] = useState<CompanyProfile | null>(null);
  const [companies, setCompanies] = useState<CompanySummary[]>([]);
  const [events, setEvents] = useState<Event[]>([]);
  const [eventsLoading, setEventsLoading] = useState(false);
  const [eventsMoreLoading, setEventsMoreLoading] = useState(false);
  const [eventsHasMore, setEventsHasMore] = useState(false);
  const [insightsByEvent, setInsightsByEvent] = useState<Record<string, string>>({});
  const [jobsError, setJobsError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobIds, setJobIds] = useState<string[]>([]);
  const [jobs, setJobs] = useState<Record<string, Job>>({});
  const [selectedEvent, setSelectedEvent] = useState<Event | null>(null);
  const [source, setSource] = useState<SourceDetail | null>(null);
  const [sourceLoading, setSourceLoading] = useState(false);
  const [sourceError, setSourceError] = useState<string | null>(null);
  const [relogin, setRelogin] = useState(false);
  const latestSourceRequest = useRef(0);

  const logOut = useCallback(() => {
    latestSourceRequest.current += 1;
    sessionStorage.removeItem(SESSION_STORAGE_KEY);
    setSession(null); setProfile(null); setEvents([]); setEventsHasMore(false); setInsightsByEvent({}); setCompanies([]); setSelectedEvent(null); setError(null); setJobIds([]); setJobs({}); setJobsError(null); setScreen('dashboard');
  }, []);

  const safeError = useCallback((e: unknown): string => {
    if (e instanceof ApiError && e.status === 401) { logOut(); return 'Демосессия истекла. Подключись снова.'; }
    return e instanceof Error ? e.message : 'Неизвестная ошибка запроса';
  }, [logOut]);

  const checkHealth = useCallback(async () => {
    try { await api.health(); setHealth('online'); } catch { setHealth('offline'); }
  }, []);
  useEffect(() => { void checkHealth(); const id = window.setInterval(() => { void checkHealth(); }, 15000); return () => clearInterval(id); }, [checkHealth]);

  const connect = async (company: string, role: Role) => {
    setSessionBusy(true); setSessionError(null);
    try {
      const next = await api.newSession(company, role);
      latestSourceRequest.current += 1;
      activeSessionToken.current = next.token;
      // Clear previous tenant/role data before rendering the new session.
      setProfile(null); setCompanies([]); setEvents([]); setEventsHasMore(false);
      setInsightsByEvent({}); setJobIds([]); setJobs({}); setJobsError(null);
      setSelectedEvent(null); setSource(null); setSourceError(null); setError(null);
      sessionStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(next));
      setSession(next); setScreen('dashboard'); setRelogin(false);
    } catch (e) { setSessionError(safeError(e)); } finally { setSessionBusy(false); }
  };

  const reloadEvents = useCallback(async (current: DemoSession) => {
    setEventsLoading(true);
    try {
      const data = await api.events(current.company_id, current.token, EVENTS_PAGE_SIZE, 0);
      if (activeSessionToken.current !== current.token) return;
      setEvents(data.items);
      setEventsHasMore(data.items.length === EVENTS_PAGE_SIZE);
    } catch (e) { if (activeSessionToken.current === current.token) setError(safeError(e)); }
    finally { if (activeSessionToken.current === current.token) setEventsLoading(false); }
  }, [safeError]);

  // RSS ingestion jobs belong to the system, not this browser session. Poll the
  // company events independently of /jobs, merging recent rows into older pages.
  useEffect(() => {
    if (!session) return;
    let active = true;
    let inFlight = false;
    const refresh = async () => {
      if (!active || inFlight || document.visibilityState === 'hidden') return;
      inFlight = true;
      try {
        const result = await api.events(session.company_id, session.token, EVENTS_PAGE_SIZE, 0);
        if (active && activeSessionToken.current === session.token) {
          setEvents(previous => mergeRecentEvents(previous, result.items));
        }
      } catch (e) {
        if (active && activeSessionToken.current === session.token && e instanceof ApiError && e.status === 401) logOut();
        // A transient polling error must not discard the last good event list.
      } finally { inFlight = false; }
    };
    const onVisible = () => { if (document.visibilityState === 'visible') void refresh(); };
    const interval = window.setInterval(() => { void refresh(); }, RSS_EVENT_REFRESH_MS);
    document.addEventListener('visibilitychange', onVisible);
    return () => { active = false; window.clearInterval(interval); document.removeEventListener('visibilitychange', onVisible); };
  }, [session, logOut]);

  const loadMoreEvents = useCallback(async () => {
    if (!session || eventsMoreLoading || eventsLoading || !eventsHasMore) return;
    setEventsMoreLoading(true);
    try {
      const data = await api.events(session.company_id, session.token, EVENTS_PAGE_SIZE, events.length);
      if (activeSessionToken.current !== session.token) return;
      setEvents(current => {
        const existing = new Set(current.map(event => event.id));
        return [...current, ...data.items.filter(event => !existing.has(event.id))];
      });
      setEventsHasMore(data.items.length === EVENTS_PAGE_SIZE);
    } catch (e) { if (activeSessionToken.current === session.token) setError(safeError(e)); }
    finally { if (activeSessionToken.current === session.token) setEventsMoreLoading(false); }
  }, [session, eventsMoreLoading, eventsLoading, eventsHasMore, events.length, safeError]);

  useEffect(() => {
    if (!session) return;
    let active = true;
    setProfile(null); setEvents([]); setEventsHasMore(false); setInsightsByEvent({}); setCompanies([]); setError(null);
    setJobIds([]); setJobs({}); setJobsError(null);
    void Promise.allSettled([
      api.profile(session.company_id, session.token),
      api.companies(session.token),
      api.events(session.company_id, session.token, EVENTS_PAGE_SIZE, 0),
      api.jobs(session.token),
      analysisApi.feed(session.company_id, session.token),
    ]).then(results => {
      if (!active) return;
      if (results[0].status === 'fulfilled') setProfile(results[0].value); else setError(safeError(results[0].reason));
      if (results[1].status === 'fulfilled') setCompanies(results[1].value.items);
      if (results[2].status === 'fulfilled') {
        setEvents(results[2].value.items);
        setEventsHasMore(results[2].value.items.length === EVENTS_PAGE_SIZE);
      } else setError(safeError(results[2].reason));
      if (results[3].status === 'fulfilled') {
        setJobIds(results[3].value.items.map(job => job.id));
        setJobs(Object.fromEntries(results[3].value.items.map(job => [job.id, job])));
      } else setJobsError(safeError(results[3].reason));
      // Saved verified cards are separate from the raw event extraction status.
      // Use existing feed references; loading the page never creates paid jobs.
      if (results[4].status === 'fulfilled') {
        setInsightsByEvent(Object.fromEntries(results[4].value.items
          .filter(item => item.insight).map(item => [item.event_id, item.insight!.status])));
      }
    });
    return () => { active = false; };
  }, [session, safeError]);

  useEffect(() => {
    if (!session) return;
    let active = true;
    let previousCompleted: string | null = null;
    const refresh = async () => {
      try {
        const result = await api.jobs(session.token);
        if (!active) return;
        setJobsError(null);
        setJobIds(result.items.map(job => job.id));
        setJobs(Object.fromEntries(result.items.map(job => [job.id, job])));
        const completedSignature = result.items.filter(job => job.status === 'succeeded').map(job => job.id).sort().join(',');
        if (previousCompleted !== null && completedSignature !== previousCompleted) {
          await reloadEvents(session);
        }
        previousCompleted = completedSignature;
      } catch (e) {
        if (!active) return;
        if (e instanceof ApiError && e.status === 401) { logOut(); return; }
        setJobsError(safeError(e));
      }
    };
    void refresh();
    const id = window.setInterval(() => { void refresh(); }, 3000);
    return () => { active = false; window.clearInterval(id); };
  }, [session, logOut, reloadEvents, safeError]);

  const publish = async (payload: PublicationPayload): Promise<boolean> => {
    if (!session) return false;
    setBusy(true); setError(null);
    try {
      const key = `web-${crypto.randomUUID()}`;
      const job = await api.publish(payload, session.token, key);
      setJobs(previous => ({ ...previous, [job.id]: job }));
      setJobIds(previous => [job.id, ...previous.filter(id => id !== job.id)].slice(0, 8));
      return true;
    } catch (e) { setError(safeError(e)); return false; }
    finally { setBusy(false); }
  };

  const openSource = useCallback(async (id: string) => {
    if (!session) return;
    const request = ++latestSourceRequest.current;
    setSourceLoading(true); setSource(null); setSourceError(null);
    try { const result = await api.source(id, session.token); if (request === latestSourceRequest.current) setSource(result); }
    catch (e) { if (request === latestSourceRequest.current) setSourceError(safeError(e)); }
    finally { if (request === latestSourceRequest.current) setSourceLoading(false); }
  }, [safeError, session]);
  const openEvent = useCallback((event: Event) => {
    // Source loading is explicit: clicking "Источник 1" must visibly open it.
    // Invalidate an outstanding response from a previously opened event.
    latestSourceRequest.current += 1;
    setSelectedEvent(event);
    setSource(null);
    setSourceLoading(false);
    setSourceError(null);
  }, []);
  const openEventId = useCallback(async (id: string) => {
    if (!session) return;
    try { const event = await api.event(id, session.token); openEvent(event); }
    catch (e) { setError(safeError(e)); }
  }, [openEvent, safeError, session]);

  const markInsightReady = useCallback((eventId: string, status: string) => {
    setInsightsByEvent(current => current[eventId] === status ? current : { ...current, [eventId]: status });
  }, []);

  const company = useMemo(() => companies.find(item => item.id === session?.company_id), [companies, session]);
  const completed = jobIds.filter(id => jobs[id]?.status === 'succeeded').length;
  const pending = jobIds.filter(id => jobs[id]?.status === 'running' || jobs[id]?.status === 'queued').length;

  if (!session || relogin) return <SessionGate onConnect={connect} busy={sessionBusy} error={sessionError} initialSession={session} onCancel={session ? () => setRelogin(false) : undefined}/>;

  return <div className="workspace">
    <aside className="sidebar">
      <button className="brand" type="button" onClick={() => setScreen('dashboard')} aria-label="Alphich, обзор"><span className="brand-icon">a<span>.</span></span><span className="brand-name">alphich<small>БИЗНЕС В КОНТЕКСТЕ</small></span></button>
      <div className="sidebar-category">ВАШЕ ПРОСТРАНСТВО</div>
      <nav className="nav-menu" aria-label="Навигация">{NAV.map(link => <button type="button" key={link.key} onClick={() => setScreen(link.key)} className={`nav-link ${screen === link.key ? 'active' : ''}`} aria-current={screen === link.key ? 'page' : undefined}><Icon name={link.icon} size={19}/><span>{link.label}</span>{screen === link.key && <span className="nav-active-bar"/>}</button>)}</nav>
      <div className="sidebar-bottom"><div className="sidebar-note"><span className="sidebar-note-dot"/>В КОНТЕКСТЕ ВАШЕГО БИЗНЕСА<strong>Видеть главное.<br/>Действовать осознанно.</strong><span>Персональный взгляд на события и их последствия.</span></div><button className="sidebar-logout" type="button" onClick={logOut}><Icon name="logout" size={18}/> Завершить сессию</button></div>
    </aside>
    <a className="skip-link" href="#page-main">К содержимому</a><main className="main" id="page-main">
      <header className="topbar"><div className="breadcrumb"><span>Alphich</span><Icon name="chevron" size={15}/><strong>{NAV.find(x => x.key === screen)?.label}</strong></div><div className="topbar-right"><span className={`online-badge online-${health}`}><span className="online-dot"/>{health === 'online' ? 'Сервис доступен' : health === 'offline' ? 'Нет связи' : 'Подключаемся'}</span><div className="user-badge"><span className="avatar">{session.role === 'cfo' ? 'Ф' : 'Г'}</span><span>{session.company_id}<small>{session.role === 'cfo' ? 'Финансовый директор' : 'Генеральный директор'}</small></span><button type="button" className="switch-session" onClick={() => setRelogin(true)}>Сменить</button></div></div></header>
      <div className="page-content">
        {error && <div className="global-error"><Notice kind="error" onClose={() => setError(null)}>{error}</Notice></div>}
        {jobsError && <div className="global-error"><Notice kind="error" onClose={() => setJobsError(null)}>Не удалось обновить историю задач: {jobsError}</Notice></div>}
        {screen === 'dashboard' && <>
          <div className="briefing-date"><span className="eyebrow">ВАШ ДЕЛОВОЙ БРИФИНГ</span><span>{new Date().toLocaleDateString('ru-RU', { day: 'numeric', month: 'long', year: 'numeric' })}</span></div>
          <section className="hero" aria-label="Обзор аналитики"><div className="hero-content"><h1>Меньше шума.<br/><em>Больше ясности.</em></h1><p>Что происходит в мире.<br/>Что это значит для вашего бизнеса.</p><div className="hero-signature"><span/> ВАЖНОЕ НАХОДИТ СВЯЗЬ</div></div><div className="signal-art" aria-hidden="true"><div className="signal-orbit signal-orbit-a"/><div className="signal-orbit signal-orbit-b"/><div className="signal-orbit signal-orbit-c"/><div className="signal-axis"/><span className="signal-core">α</span><span className="signal-point point-a"/><span className="signal-point point-b"/><span className="signal-point point-c"/><span className="signal-caption">СОБЫТИЕ → КОНТЕКСТ → РЕШЕНИЕ</span></div></section>
          <div className="context-strip"><span><Icon name="briefcase" size={16}/><strong>{session.company_id}</strong><span>{company?.okved_name || 'Профиль компании'}</span></span><button type="button" onClick={() => setScreen('profile')}>Мой бизнес-контекст <Icon name="arrow" size={15}/></button></div>
          <div className="briefing-layout"><div className="briefing-main"><InsightFeed key={session.token} session={session} onEvent={id => void openEventId(id)} compact onAll={() => setScreen('insights')}/></div><aside className="briefing-aside" aria-label="Контекст и навигация">
            <div className="focus-card"><div className="eyebrow">ВАША ОПТИКА</div><div className="focus-icon"><Icon name="briefcase" size={22}/></div><h2>{session.role === 'cfo' ? 'Финансовый фокус' : 'Стратегический фокус'}</h2><p>{session.role === 'cfo' ? 'Влияние событий на финансирование, издержки и денежные потоки компании.' : 'Изменения на рынках, возможности для бизнеса и условия для решений.'}</p><div className="focus-fact"><span>Деятельность</span><strong>{profile?.industry.okved_name || company?.okved_name || 'Не указана'}</strong></div><div className="focus-fact"><span>Регион регистрации</span><strong>{profile?.region_registration || 'Не указан'}</strong></div><button type="button" className="text-button" onClick={() => setScreen('profile')}>Посмотреть профиль <Icon name="arrow" size={16}/></button></div>
            <div className="how-card"><span className="eyebrow">ОТ НОВОСТИ К ДЕЙСТВИЮ</span><ol><li><span>01</span><div><strong>Событие</strong><p>Что произошло и где источник</p></div></li><li><span>02</span><div><strong>Влияние</strong><p>Почему это касается компании</p></div></li><li><span>03</span><div><strong>Следующий шаг</strong><p>Что проверить перед решением</p></div></li></ol></div>
            <button type="button" className="add-news-card" onClick={() => setScreen('publish')}><Icon name="plus" size={20}/><span><strong>Есть важная новость?</strong><small>Добавьте её для разбора</small></span><Icon name="arrow" size={17}/></button>
          </aside></div>
          <details className="operations-details"><summary><Icon name="list" size={17}/><span>Последние публикации и обработка</span><span className="muted">{events.length}{eventsHasMore ? '+' : ''} событий · {pending} в работе · {completed} готово</span></summary><div className="dashboard-grid"><EventsList items={events.slice(0, 5)} insightsByEvent={insightsByEvent} loading={eventsLoading} onReload={() => void reloadEvents(session)} onOpen={openEvent} onPublish={() => setScreen('publish')}/><JobList ids={jobIds} jobs={jobs} onEvent={id => void openEventId(id)}/></div></details>
        </>}
        {screen === 'publish' && <><div className="page-title"><div className="eyebrow">СОБЫТИЕ В ВАШЕМ КОНТЕКСТЕ</div><h1>Новая публикация<span>.</span></h1><p>Сохраните публикацию. Откройте готовое событие, чтобы запросить персональный разбор.</p></div><div className="wide-column"><PublishForm onPublish={publish} busy={busy}/><div className="publish-aside"><JobList ids={jobIds} jobs={jobs} onEvent={id => void openEventId(id)}/><div className="tip-panel"><Icon name="activity" size={22}/><h3>Что происходит после отправки?</h3><p>Мы сохраним текст и источник, затем подготовим событие. Откройте его и нажмите «Разобрать влияние», чтобы получить аналитику для своей компании.</p><span>Каждый существенный вывод можно проверить по источникам.</span></div></div></div></>}
        {screen === 'events' && <><div className="page-title"><div className="eyebrow">НОВОСТНОЙ КОНТЕКСТ</div><h1>События<span>.</span></h1><p>Сохранённые события компании {session.company_id}. Данные сохраняются при обновлении страницы.</p></div><EventsList items={events} insightsByEvent={insightsByEvent} loading={eventsLoading} moreLoading={eventsMoreLoading} hasMore={eventsHasMore} onMore={() => void loadMoreEvents()} onReload={() => void reloadEvents(session)} onOpen={openEvent} onPublish={() => setScreen('publish')}/></>}
        {screen === 'insights' && <InsightFeed key={session.token} session={session} onEvent={id => void openEventId(id)}/>}
        {screen === 'profile' && <><div className="page-title"><div className="eyebrow">БИЗНЕС-КОНТЕКСТ</div><h1>Анализ компании<span>.</span></h1><p>Что известно о бизнесе, какие механизмы влияния проверить и какие данные нужны для решения.</p></div><CompanyAnalysis key={session.token} session={session}/><details><summary>Исходный профиль компании</summary><Profile profile={profile}/></details></>}
        <footer className="app-footer"><span>ALPHICH · ОТ СОБЫТИЯ К РЕШЕНИЮ</span><span>Демо хакатона · Обезличенные данные</span></footer>
      </div>
    </main>
    {selectedEvent && <EventDrawer key={`${session.token}:${selectedEvent.id}`} session={session} event={selectedEvent} insightStatus={insightsByEvent[selectedEvent.id]} onInsightReady={markInsightReady} source={source} loading={sourceLoading} error={sourceError} onClose={() => { latestSourceRequest.current += 1; setSelectedEvent(null); }} onSource={id => void openSource(id)}/>}
  </div>;
}

export default App;
