import type { CompanyProfile, CompanySummary, DemoSession, Event, Job, NewsCollectionStatus, NewsSearchRun, PublicationPayload, Role, SourceDetail } from './types';

export class ApiError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, options: RequestInit = {}, token?: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api/v1${path}`, {
      ...options,
      headers: {
        Accept: 'application/json',
        ...(options.body ? { 'Content-Type': 'application/json' } : {}),
        ...(token ? { 'X-Demo-Session': token } : {}),
        ...(options.headers ?? {}),
      },
      cache: 'no-store',
    });
  } catch {
    throw new ApiError(0, 'Нет связи с сервером. Проверьте подключение и повторите попытку.');
  }
  if (!response.ok) {
    let message = response.status === 504
      ? 'Сервер не успел подготовить данные. Повторите попытку позже.'
      : `Ошибка API: HTTP ${response.status}`;
    try {
      const payload: unknown = await response.json();
      if (payload && typeof payload === 'object' && 'detail' in payload) {
        const detail = (payload as { detail: unknown }).detail;
        const known: Record<string, string> = {
          'Company is not on the server demo allowlist': 'Компания недоступна для демонстрации. Проверьте её идентификатор.',
          'Demo session required': 'Войдите в систему.',
          'Demo session expired or invalid': 'Сессия истекла. Войдите снова.',
          llm_not_configured: 'Аналитический сервис ещё не настроен.',
          budget_exceeded: 'Достигнут общий лимит аналитики. Попробуйте позже.',
          'Not Found': 'Данные или функция недоступны на этом сервере.',
          'Company not found': 'Компания не найдена. Проверьте её идентификатор.',
          'Event not found': 'Событие не найдено или недоступно в этой сессии.',
          'Analysis result not found': 'Сохранённый анализ не найден или недоступен в этой сессии.',
          'Session does not grant access to this company': 'Эта компания недоступна в текущей сессии.',
        };
        message = typeof detail === 'string' ? known[detail] ?? detail : 'Не удалось обработать данные. Проверьте поля и повторите попытку.';
      }
    } catch { /* Use the safe default message. */ }
    throw new ApiError(response.status, message);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string; database: string }>('/health'),
  newSession: (company_id: string, role: Role) =>
    request<DemoSession>('/demo-session', { method: 'POST', body: JSON.stringify({ company_id, role }) }),
  companies: (token: string) => request<{ items: CompanySummary[] }>('/companies', {}, token),
  profile: (companyId: string, token: string) => request<CompanyProfile>(`/companies/${encodeURIComponent(companyId)}/profile`, {}, token),
  events: (companyId: string, token: string, limit = 20, offset = 0) =>
    request<{ items: Event[] }>(`/companies/${encodeURIComponent(companyId)}/events?limit=${limit}&offset=${offset}`, {}, token),
  jobs: (token: string, limit = 20) =>
    request<{ items: Job[] }>(`/jobs?limit=${limit}`, {}, token),
  publish: (payload: PublicationPayload, token: string, key: string) =>
    request<Job>('/publications', { method: 'POST', headers: { 'Idempotency-Key': key }, body: JSON.stringify(payload) }, token),
  job: (jobId: string, token: string) => request<Job>(`/jobs/${encodeURIComponent(jobId)}`, {}, token),
  event: (eventId: string, token: string) => request<Event>(`/events/${encodeURIComponent(eventId)}`, {}, token),
  source: (sourceId: string, token: string) => request<SourceDetail>(`/sources/${encodeURIComponent(sourceId)}`, {}, token),
  collectionStatus: (token: string) => request<NewsCollectionStatus>('/news-collection/status', {}, token),
  latestNewsSearch: (companyId: string, token: string) =>
    request<{ run: NewsSearchRun | null }>(`/companies/${encodeURIComponent(companyId)}/news-search`, {}, token),
  startNewsSearch: (companyId: string, count: number, token: string, key: string) =>
    request<NewsSearchRun>(`/companies/${encodeURIComponent(companyId)}/news-search`, {
      method: 'POST', headers: { 'Idempotency-Key': key }, body: JSON.stringify({ count }),
    }, token),
  newsSearch: (id: string, token: string) => request<NewsSearchRun>(`/news-search/${encodeURIComponent(id)}`, {}, token),
  stopNewsSearch: (id: string, token: string) =>
    request<NewsSearchRun>(`/news-search/${encodeURIComponent(id)}/stop`, { method: 'POST' }, token),
};

/** Analysis operations are served by the existing intelligence router.
 * Every call uses the same demo token as ingestion. No provider key reaches the browser. */
export const analysisApi = {
  feedback: (id: string, token: string) =>
    request<{ feedback: import('./analysis').InsightFeedback | null }>(`/insights/${encodeURIComponent(id)}/feedback`, {}, token),
  saveFeedback: (id: string, value: import('./analysis').FeedbackValue, comment: string, token: string) =>
    request<{ feedback: import('./analysis').InsightFeedback }>(`/insights/${encodeURIComponent(id)}/feedback`, {
      method: 'PUT', body: JSON.stringify({ value, comment }),
    }, token),
  businessContext: (companyId: string, token: string) =>
    request<import('./CompanyAnalysis').BusinessContext>(`/companies/${encodeURIComponent(companyId)}/business-context`, {}, token),
  discoveryLatest: (companyId: string, token: string) =>
    request<{ run: import('./DiscoveryPanel').DiscoveryRun | null }>(`/companies/${encodeURIComponent(companyId)}/discovery`, {}, token),
  startDiscovery: (companyId: string, token: string, minGrade: 4 | 5) =>
    request<import('./DiscoveryPanel').DiscoveryRun>(`/companies/${encodeURIComponent(companyId)}/discovery`, {
      method: 'POST', headers: { 'Idempotency-Key': `discovery-${crypto.randomUUID()}` },
      body: JSON.stringify({ target: 10, min_grade: minGrade, max_candidates: 40 }),
    }, token),
  discovery: (id: string, token: string) => request<import('./DiscoveryPanel').DiscoveryRun>(`/discovery/${encodeURIComponent(id)}`, {}, token),
  stopDiscovery: (id: string, token: string) => request<import('./DiscoveryPanel').DiscoveryRun>(`/discovery/${encodeURIComponent(id)}/stop`, { method: 'POST' }, token),
  savedEventInsight: (eventId: string, token: string) =>
    request<{ insight: import('./analysis').Insight | null }>(`/events/${encodeURIComponent(eventId)}/saved-insight`, {}, token),
  status: (token: string) => request<import('./analysis').LlmStatus>('/llm/status', {}, token),
  feed: (companyId: string, token: string) =>
    request<import('./analysis').FeedResult>(`/companies/${encodeURIComponent(companyId)}/feed`, {}, token),
  createInsight: (companyId: string, eventId: string, token: string) =>
    request<Job>(`/companies/${encodeURIComponent(companyId)}/insights`, {
      method: 'POST', headers: { 'Idempotency-Key': `insight-${crypto.randomUUID()}` },
      body: JSON.stringify({ event_id: eventId }),
    }, token),
  insight: (id: string, token: string) =>
    request<import('./analysis').Insight>(`/insights/${encodeURIComponent(id)}`, {}, token),
  ask: (id: string, question: string, token: string) =>
    request<Job>(`/insights/${encodeURIComponent(id)}/questions`, {
      method: 'POST', headers: { 'Idempotency-Key': `question-${crypto.randomUUID()}` },
      body: JSON.stringify({ question }),
    }, token),
  answer: (id: string, token: string) =>
    request<import('./analysis').Answer>(`/answers/${encodeURIComponent(id)}`, {}, token),
  evidence: (id: string, insightId: string, token: string, answerId?: string) => {
    const query = new URLSearchParams({ insight_id: insightId });
    if (answerId) query.set('answer_id', answerId);
    return request<import('./analysis').Evidence>(`/evidence/${encodeURIComponent(id)}?${query}`, {}, token);
  },
  run: (id: string, token: string) =>
    request<import('./analysis').AuditRun>(`/runs/${encodeURIComponent(id)}`, {}, token),
};
