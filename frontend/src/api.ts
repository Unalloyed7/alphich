import type { CompanyProfile, CompanySummary, DemoSession, Event, Job, PublicationPayload, Role, SourceDetail } from './types';

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
    throw new ApiError(0, 'Сервер недоступен. Убедитесь, что запущен Docker Compose.');
  }
  if (!response.ok) {
    let message = `Ошибка API: HTTP ${response.status}`;
    try {
      const payload: unknown = await response.json();
      if (payload && typeof payload === 'object' && 'detail' in payload) {
        const detail = (payload as { detail: unknown }).detail;
        message = typeof detail === 'string' ? detail : JSON.stringify(detail);
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
  events: (companyId: string, token: string) => request<{ items: Event[] }>(`/companies/${encodeURIComponent(companyId)}/events?limit=50`, {}, token),
  publish: (payload: PublicationPayload, token: string, key: string) =>
    request<Job>('/publications', { method: 'POST', headers: { 'Idempotency-Key': key }, body: JSON.stringify(payload) }, token),
  job: (jobId: string, token: string) => request<Job>(`/jobs/${encodeURIComponent(jobId)}`, {}, token),
  event: (eventId: string, token: string) => request<Event>(`/events/${encodeURIComponent(eventId)}`, {}, token),
  source: (sourceId: string, token: string) => request<SourceDetail>(`/sources/${encodeURIComponent(sourceId)}`, {}, token),
};
