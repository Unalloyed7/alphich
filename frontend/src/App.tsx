import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from 'react';
import { api, analysisApi, ApiError } from './api';
import { EventInsight, InsightFeed } from './InsightUI';
import { CompanyAnalysis } from './CompanyAnalysis';
import { mergeRecentEvents } from './recentEvents';
import type { CompanyProfile, CompanySummary, DemoSession, Event, Job, PublicationPayload, Role, SourceDetail } from './types';

type Screen = 'dashboard' | 'publish' | 'events' | 'insights' | 'profile';
type IconName = 'grid' | 'plus' | 'list' | 'briefcase' | 'chevron' | 'arrow' | 'refresh' | 'check' | 'clock' | 'link' | 'file' | 'alert' | 'close' | 'send' | 'logout' | 'activity' | 'copy' | 'search';
const SESSION_STORAGE_KEY = 'alphich.demo-session.v1';
const EVENTS_PAGE_SIZE = 20;
const RSS_EVENT_REFRESH_MS = 15_000;

function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  const paths: Record<IconName, ReactNode> = {
    grid: <><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></>,
    plus: <><path d="M12 5v14M5 12h14"/></>,
    list: <><path d="M9 6h12M9 12h12M9 18h12"/><path d="M4 6h.01M4 12h.01M4 18h.01"/></>,
    briefcase: <><rect x="2" y="7" width="20" height="14" rx="2"/><path d="M8 7V5a2 2 0 012-2h4a2 2 0 012 2v2M2 12h20"/></>,
    chevron: <path d="m9 18 6-6-6-6"/>,
    arrow: <><path d="M4 12h16M14 6l6 6-6 6"/></>,
    refresh: <><path d="M20 11a8 8 0 1 0-2 6"/><path d="M20 4v7h-7"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    clock: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
    link: <><path d="M10 13a5 5 0 0 0 7 .5l3-3a5 5 0 0 0-7-7l-2 2"/><path d="M14 11a5 5 0 0 0-7-.5l-3 3a5 5 0 0 0 7 7l2-2"/></>,
    file: <><path d="M6 2h9l5 5v15H6z"/><path d="M15 2v6h5M9 13h8M9 17h8"/></>,
    alert: <><path d="m12 3 10 18H2z"/><path d="M12 9v5M12 18h.01"/></>,
    close: <path d="M18 6 6 18M6 6l12 12"/>,
    send: <><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></>,
    logout: <><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><path d="M16 17l5-5-5-5M21 12H9"/></>,
    activity: <><path d="M3 12h4l3-7 4 14 3-7h4"/></>,
    copy: <><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/></>,
    search: <><circle cx="11" cy="11" r="7"/><path d="m16 16 5 5"/></>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

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
  const labels: Record<string, string> = { queued: 'В очереди', running: 'В работе', succeeded: 'Сохранено', failed: 'Ошибка', raw: 'Без AI-анализа', analyzed: 'Факты извлечены', insight: 'AI-карточка готова', insufficient: 'Недостаточно данных' };
  return <span className={`status-pill status-${status}`}><span className="status-dot"/>{labels[status]}</span>;
}

const NAV: { key: Screen; label: string; icon: IconName }[] = [
  { key: 'dashboard', label: 'Обзор', icon: 'grid' },
  { key: 'publish', label: 'Новая публикация', icon: 'plus' },
  { key: 'events', label: 'События', icon: 'list' },
  { key: 'insights', label: 'Аналитика', icon: 'activity' },
  { key: 'profile', label: 'Анализ компании', icon: 'briefcase' },
];

function SessionGate({ onConnect, busy, error }: {
  onConnect: (company: string, role: Role) => Promise<void>; busy: boolean; error: string | null;
}) {
  const [companyId, setCompanyId] = useState('DEMO_A');
  const [role, setRole] = useState<Role>('cfo');
  const submit = (e: FormEvent) => { e.preventDefault(); void onConnect(companyId.trim(), role); };
  return <div className="onboarding">
    <div className="onboarding-art"><div className="art-orbit orbit-one"/><div className="art-orbit orbit-two"/><div className="art-logo">a<span>.</span></div><div className="art-caption">YOUR BUSINESS.<br/>IN CONTEXT.</div></div>
    <div className="onboarding-content">
      <div className="eyebrow"><span className="eyebrow-line"/> ALPHICH / РАБОЧЕЕ ПРОСТРАНСТВО</div>
      <h1>Новости, которые<br/><em>имеют значение.</em></h1>
      <p className="onboarding-desc">Единое пространство для проверки новостного конвейера, данных компании и источников. Демо-сервис для команды хакатона.</p>
      <form onSubmit={submit} className="connect-form">
        <div className="form-title">Войти в демо-пространство</div>
        <label htmlFor="company-id">Идентификатор компании</label>
        <input id="company-id" value={companyId} onChange={(e) => setCompanyId(e.target.value)} placeholder="DEMO_A или U_9250" required maxLength={100} spellCheck={false}/>
        <div className="quick-company"><span>Примеры:</span>{['DEMO_A', 'DEMO_B', 'U_9250'].map((id) => <button key={id} type="button" onClick={() => setCompanyId(id)}>{id}</button>)}</div>
        <label htmlFor="company-role">Ваша роль</label>
        <select id="company-role" value={role} onChange={(e) => setRole(e.target.value as Role)}><option value="cfo">Финансовый директор (CFO)</option><option value="ceo">Генеральный директор (CEO)</option></select>
        {error && <Notice kind="error">{error}</Notice>}
        <button type="submit" className="primary-button connect-button" disabled={busy || !companyId.trim()}>{busy ? 'Подключение…' : <>Открыть рабочее пространство <Icon name="arrow"/></>}</button>
      </form>
      <p className="helper-message">Используй ID из `DEMO_COMPANY_IDS`. Компания должна быть предварительно импортирована в БД. Доступ предоставляется только на время демосессии.</p>
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
      <div className="panel-heading"><div><div className="eyebrow">НОВЫЕ ДАННЫЕ</div><h2>Добавить публикацию</h2><p>Вставь любую новую экономическую новость. Текст будет сохранён вместе с источником, а worker создаст событие.</p></div><span className="panel-heading-icon"><Icon name="file" size={25}/></span></div>
      {message && <Notice kind="success" onClose={() => setMessage(null)}>{message}</Notice>}
      <div className="field"><label htmlFor="news-title">Заголовок <span>*</span></label><input id="news-title" value={title} onChange={e => { setTitle(e.target.value); setMessage(null); }} minLength={6} maxLength={300} required placeholder="Например: Банк России сохранил ключевую ставку"/></div>
      <div className="field"><label htmlFor="news-body">Текст публикации <span>*</span></label><textarea id="news-body" value={text} onChange={e => { setText(e.target.value); setMessage(null); }} minLength={30} maxLength={100000} required rows={9} placeholder="Вставь полный текст публикации или доступный фрагмент (от 30 символов)…"/><div className="field-help"><span>Минимум 30 символов</span><span>{text.length.toLocaleString('ru-RU')} / 100 000</span></div></div>
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
      <div className="empty-state"><div className="empty-icon"><Icon name="file" size={27}/></div><h3>Пока нет событий</h3><p>Добавь первую публикацию. После обработки worker она появится здесь автоматически.</p><button className="secondary-button" type="button" onClick={onPublish}><Icon name="plus" size={16}/> Добавить публикацию</button></div> : visible.length === 0 ?
      <div className="empty-state"><h3>Совпадений нет</h3><p>Поиск выполняется только среди загруженных событий. Можно загрузить более старые.</p><button type="button" className="ghost-button" onClick={() => setSearch('')}>Сбросить поиск</button></div> :
      <div className="event-list">{visible.map(item => <EventCard key={item.id} item={item} onOpen={onOpen} insightStatus={insightsByEvent[item.id]}/>)}</div>}
    {hasMore && onMore && <div className="event-pagination"><button className="secondary-button" type="button" disabled={loading || moreLoading} onClick={onMore}>{moreLoading ? 'Загружаем…' : 'Показать ещё события'}</button></div>}
  </div>;
}

function JobList({ ids, jobs, onEvent }: { ids: string[]; jobs: Record<string, Job>; onEvent: (id: string) => void }) {
  return <div className="panel jobs-panel"><div className="section-header"><div><div className="eyebrow">ОЧЕРЕДЬ WORKER</div><h2>Последние задачи</h2></div><span className="count-badge">{ids.length}</span></div>
    {ids.length === 0 ? <div className="job-empty">Отправь публикацию — здесь появится задача и её статус.</div> :
      <div className="job-list">{ids.slice(0, 6).map(id => { const job = jobs[id]; return <div className="job-item" key={id}>
        <div className={`job-status-icon ${job?.status ?? 'queued'}`}><Icon name={job?.status === 'succeeded' ? 'check' : job?.status === 'failed' ? 'alert' : 'clock'} size={18}/></div>
        <div className="job-details"><span className="job-title">{job ? jobLabel(job) : 'Загружаем статус…'}</span>{job?.source_title && <span className="job-source-title" title={job.source_title}>{job.source_title}</span>}<span className="mono job-id">{id}</span>{job?.error && <span className="job-error">Код ошибки: {job.error}</span>}</div>
        {job?.status === 'succeeded' && job.result_id ? <button type="button" className="job-open" onClick={() => onEvent(job.result_id!)} aria-label="Открыть событие"><Icon name="arrow" size={17}/></button> : job?.status === 'running' || job?.status === 'queued' ? <span className="spinner tiny"/> : null}
      </div>; })}</div>}
    <div className="jobs-note"><span className="pulse-circle"/> Статусы обновляются автоматически</div>
  </div>;
}

function Profile({ profile }: { profile: CompanyProfile | null }) {
  if (!profile) return <div className="panel loading-panel">Загрузка профиля компании…</div>;
  const products = Object.entries(profile.products ?? {});
  return <div className="panel profile-panel">
    <div className="panel-heading"><div><div className="eyebrow">ДАННЫЕ КОМПАНИИ</div><h2>Профиль {profile.id}</h2><p>Обезличенные исходные данные. Показатели, единицы и ограничения не домысливаются.</p></div><span className="panel-heading-icon"><Icon name="briefcase" size={25}/></span></div>
    <div className="profile-facts"><div><span>Основная деятельность</span><strong>{profile.industry.okved_name || 'Не указана'}</strong><small>ОКВЭД {profile.industry.okved_code || '—'}</small></div><div><span>Регион регистрации</span><strong>{profile.region_registration || 'Не указан'}</strong><small>{profile.city_registration || 'Город не указан'}</small></div><div><span>Выручка (исходное значение)</span><strong>{profile.revenue_raw || 'Нет данных'}</strong><small>Единица измерения не подтверждена</small></div><div><span>Тип клиента</span><strong>{profile.client_type || 'Не указан'}</strong><small>Версия профиля: {profile.profile_version.slice(0, 14)}</small></div></div>
    <div className="profile-column-grid"><div><h3>Банковские продукты</h3>{products.length ? <div className="feature-tags">{products.map(([key, value]) => <span className={`feature-tag ${value === true ? 'active' : ''}`} key={key}>{key}: {value === null ? 'неизвестно' : typeof value === 'boolean' ? value ? 'есть' : 'нет' : String(value)}</span>)}</div> : <p className="muted">Нет данных</p>}</div><div><h3>Страны в данных ФТС</h3><p>{profile.fts_countries?.length ? profile.fts_countries.join(', ') : 'Нет подтверждённого списка'}</p><h3>Доказуемые признаки</h3><p>{profile.features?.length ?? 0} идентификаторов с привязкой к версии данных</p></div></div>
    {profile.quality_flags.length > 0 && <Notice kind="info"><strong>Особенности данных:</strong> {profile.quality_flags.join(', ')}</Notice>}
    <div className="limitations"><h3>Ограничения интерпретации</h3>{profile.limitations.map((entry, index) => <p key={index}>— {entry}</p>)}</div>
  </div>;
}

function EventDrawer({ session, event, insightStatus, onInsightReady, source, loading, error, onClose, onSource }: {
  session: DemoSession; event: Event; insightStatus?: string; onInsightReady: (id: string, status: string) => void;
  source: SourceDetail | null; loading: boolean; error: string | null; onClose: () => void; onSource: (id: string) => void;
}) {
  const [activeSourceId, setActiveSourceId] = useState<string | null>(null);
  const sourcePanelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onEscape = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onEscape);
    return () => window.removeEventListener('keydown', onEscape);
  }, [onClose]);

  // An expanded source can be below the fold in the drawer. Bring loading,
  // errors, and the original article into view after their state is rendered.
  useEffect(() => {
    if (activeSourceId) sourcePanelRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [activeSourceId, loading, source, error]);

  return <div className="drawer-backdrop" onMouseDown={e => { if (e.target === e.currentTarget) onClose(); }}>
    <div className="drawer" role="dialog" aria-modal="true" aria-label="Информация о событии">
      <header className="drawer-head"><span className="eyebrow">ПОДРОБНОСТИ СОБЫТИЯ</span><button className="icon-button" type="button" onClick={onClose} aria-label="Закрыть"><Icon name="close"/></button></header>
      <div className="drawer-scroll">
        <StatusPill status={insightStatus ? (insightStatus === 'insufficient_evidence' ? 'insufficient' : 'insight') : event.extraction_status === 'extracted' || event.extraction_status === 'partially_extracted' ? 'analyzed' : 'raw'}/><h2>{event.title}</h2>
        <div className="drawer-metadata">
          <div><span>Публикация</span><strong>{formattedDate(event.published_at)}</strong></div>
          <div><span>Событие</span><strong>{event.event_at ? formattedDate(event.event_at) : 'Дата события не установлена'}</strong></div>
          <div><span>ID события</span><strong className="mono wrap-text">{event.id}</strong></div>
          <div><span>AI-извлечение</span><strong>{event.extraction_status}</strong></div>
        </div>
        <Notice>Исходная публикация сохранена отдельно от аналитики. Подтверждённые выводы, если они сформированы, находятся в разделе ниже и имеют ссылки на доказательства.</Notice>
        <h3 className="drawer-section-title">Первоисточники ({event.source_ids.length})</h3>
        <div className="source-row">
          {event.source_ids.map((id, index) => <button
            className={`source-button ${activeSourceId === id ? 'active' : ''}`}
            type="button"
            key={id}
            aria-expanded={activeSourceId === id}
            aria-controls="active-source-panel"
            onClick={() => {
              setActiveSourceId(id);
              onSource(id);
            }}
          ><Icon name="file" size={18}/> Источник {index + 1} <span className="source-button-hint">{activeSourceId === id ? 'Открыт' : 'Просмотреть'}</span><Icon name="chevron" size={16}/></button>)}
        </div>
        {activeSourceId && <div id="active-source-panel" ref={sourcePanelRef} className="source-panel" role="region" aria-label="Текст первоисточника" aria-live="polite">
          {loading && <div className="source-loading"><span className="spinner"/> Загружаем текст источника…</div>}
          {error && <Notice kind="error">{error}</Notice>}
          {source && source.id === activeSourceId && <section className="source-reader">
            <div className="source-reader-head"><div className="eyebrow">ОРИГИНАЛ ПУБЛИКАЦИИ</div><h3>{source.title}</h3><span className="muted small">Получен: {formattedDate(source.received_at)}</span></div>
            <p className="source-full-text">{source.text}</p>
            <div className="source-footer"><span><Icon name="alert" size={15}/> {source.url_verified ? 'Текст загружен по ссылке' : 'Ссылка не верифицирована'}</span>{source.url && <a href={source.url} target="_blank" rel="noopener noreferrer"><Icon name="link" size={16}/> Открыть внешний адрес</a>}</div>
          </section>}
        </div>}
        <EventInsight session={session} eventId={event.id} onInsightReady={onInsightReady}/>
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

  if (!session || relogin) return <SessionGate onConnect={connect} busy={sessionBusy} error={sessionError}/>;

  return <div className="workspace">
    <aside className="sidebar">
      <button className="brand" type="button" onClick={() => setScreen('dashboard')} aria-label="Alphich, обзор"><span className="brand-icon">a<span>.</span></span><span className="brand-name">alphich<small>INTELLIGENCE SPACE</small></span></button>
      <div className="sidebar-category">РАБОЧЕЕ ПРОСТРАНСТВО</div>
      <nav className="nav-menu" aria-label="Навигация">{NAV.map(link => <button type="button" key={link.key} onClick={() => setScreen(link.key)} className={`nav-link ${screen === link.key ? 'active' : ''}`} aria-current={screen === link.key ? 'page' : undefined}><Icon name={link.icon} size={19}/><span>{link.label}</span>{screen === link.key && <span className="nav-active-bar"/>}</button>)}</nav>
      <div className="sidebar-bottom"><div className="sidebar-note"><span className="sidebar-note-dot"/>DEMO / P0<br/><strong>Аналитика по запросу</strong><span>Генерация доступна только с настроенным шлюзом и учётом общего бюджета.</span></div><button className="sidebar-logout" type="button" onClick={logOut}><Icon name="logout" size={18}/> Завершить сессию</button></div>
    </aside>
    <main className="main">
      <header className="topbar"><div className="breadcrumb"><span>Alphich</span><Icon name="chevron" size={15}/><strong>{NAV.find(x => x.key === screen)?.label}</strong></div><div className="topbar-right"><span className={`online-badge online-${health}`}><span className="online-dot"/>{health === 'online' ? 'API доступен' : health === 'offline' ? 'Нет связи с API' : 'Проверяем API'}</span><div className="user-badge"><span className="avatar">{session.role === 'cfo' ? 'Ф' : 'Г'}</span><span>{session.company_id}<small>{session.role === 'cfo' ? 'Финансовый директор' : 'Генеральный директор'}</small></span><button type="button" className="switch-session" onClick={() => setRelogin(true)}>Сменить</button></div></div></header>
      <div className="page-content">
        {error && <div className="global-error"><Notice kind="error" onClose={() => setError(null)}>{error}</Notice></div>}
        {jobsError && <div className="global-error"><Notice kind="error" onClose={() => setJobsError(null)}>Не удалось обновить историю задач: {jobsError}</Notice></div>}
        {screen === 'dashboard' && <>
          <div className="hero"><div className="hero-content"><div className="hero-overline"><span className="hero-dot"/> ЦЕНТР БИЗНЕС-АНАЛИТИКИ</div><h1>Контекст важнее<br/><em>заголовков.</em></h1><p>Все новые публикации, статусы обработки и подтверждающие материалы — в одном месте.</p><button type="button" className="hero-button" onClick={() => setScreen('publish')}>Добавить публикацию <Icon name="arrow" size={18}/></button></div><div className="hero-figure"><div className="hero-ring ring-a"/><div className="hero-ring ring-b"/><div className="hero-letter">α</div><div className="hero-figure-caption">DATA <span>→</span> CONTEXT <span>→</span> INSIGHT</div></div></div>
          <div className="stat-grid"><div className="stat"><span><Icon name="briefcase" size={19}/> Компания</span><strong>{session.company_id}</strong><small>{company?.okved_name || 'Профиль подключён'}</small></div><div className="stat"><span><Icon name="file" size={19}/> События</span><strong>{events.length.toString().padStart(2, '0')}</strong><small>Сохранены в PostgreSQL</small></div><div className="stat"><span><Icon name="check" size={19}/> Завершено задач</span><strong>{completed.toString().padStart(2, '0')}</strong><small>Из задач этой демосессии</small></div><div className="stat"><span><Icon name="activity" size={19}/> В обработке</span><strong>{pending.toString().padStart(2, '0')}</strong><small>Статус обновляется автоматически</small></div></div>
          <div className="dashboard-grid"><EventsList items={events.slice(0, 5)} insightsByEvent={insightsByEvent} loading={eventsLoading} onReload={() => void reloadEvents(session)} onOpen={openEvent} onPublish={() => setScreen('publish')}/><JobList ids={jobIds} jobs={jobs} onEvent={id => void openEventId(id)}/></div>
        </>}
        {screen === 'publish' && <><div className="page-title"><div className="eyebrow">NEWS INGESTION / P0</div><h1>Новая публикация<span>.</span></h1><p>Проверь полный путь от отправки текста до события в базе данных.</p></div><div className="wide-column"><PublishForm onPublish={publish} busy={busy}/><div className="publish-aside"><JobList ids={jobIds} jobs={jobs} onEvent={id => void openEventId(id)}/><div className="tip-panel"><Icon name="activity" size={22}/><h3>Что происходит после отправки?</h3><p>FastAPI сохраняет источник и создаёт задачу. Worker забирает её из PostgreSQL и создаёт событие. Статус можно отслеживать здесь.</p><span>Время обработки обычно небольшое, но зависит от состояния worker.</span></div></div></div></>}
        {screen === 'events' && <><div className="page-title"><div className="eyebrow">EVENT STORAGE / P0</div><h1>События<span>.</span></h1><p>Все завершённые публикации компании {session.company_id}. Данные сохраняются при обновлении страницы.</p></div><EventsList items={events} insightsByEvent={insightsByEvent} loading={eventsLoading} moreLoading={eventsMoreLoading} hasMore={eventsHasMore} onMore={() => void loadMoreEvents()} onReload={() => void reloadEvents(session)} onOpen={openEvent} onPublish={() => setScreen('publish')}/></>}
        {screen === 'insights' && <InsightFeed key={session.token} session={session} onEvent={id => void openEventId(id)}/>}
        {screen === 'profile' && <><div className="page-title"><div className="eyebrow">БИЗНЕС-КОНТЕКСТ</div><h1>Анализ компании<span>.</span></h1><p>Что известно о бизнесе, какие механизмы влияния проверить и какие данные нужны для решения.</p></div><CompanyAnalysis key={session.token} session={session}/><details><summary>Исходный профиль компании</summary><Profile profile={profile}/></details></>}
        <footer className="app-footer"><span>© ALPHICH / HACKATHON PROTOTYPE</span><span>Профили обезличены · Источники требуют проверки</span></footer>
      </div>
    </main>
    {selectedEvent && <EventDrawer key={`${session.token}:${selectedEvent.id}`} session={session} event={selectedEvent} insightStatus={insightsByEvent[selectedEvent.id]} onInsightReady={markInsightReady} source={source} loading={sourceLoading} error={sourceError} onClose={() => { latestSourceRequest.current += 1; setSelectedEvent(null); }} onSource={id => void openSource(id)}/>}
  </div>;
}

export default App;
