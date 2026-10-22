export type Role = 'cfo' | 'ceo';

export interface DemoSession {
  token: string;
  company_id: string;
  role: Role;
  expires_at: number;
  demo_only: boolean;
}

export interface CompanySummary {
  id: string;
  okved_name: string | null;
  region: string | null;
  revenue_raw: string | null;
  revenue_unit: string | null;
  profile_version: string;
}

export interface Feature {
  id: string;
  name: string;
  value: unknown;
  source_refs: unknown[];
}

export interface CompanyProfile {
  id: string;
  role: string;
  profile_version: string;
  industry: { okved_code: string | null; okved_name: string | null };
  region_registration: string | null;
  city_registration: string | null;
  revenue_raw: string | null;
  revenue_unit: string | null;
  revenue_currency?: string | null;
  revenue_period?: string | null;
  revenue_scope?: string | null;
  client_type: string | null;
  products: Record<string, unknown>;
  fts_flag: boolean | null;
  fts_countries: string[] | null;
  features: Feature[];
  quality_flags: string[];
  limitations: string[];
}

export interface PublicationPayload {
  title: string;
  text: string;
  source_url?: string | null;
  published_at?: string | null;
}

export interface Job {
  id: string;
  status: 'queued' | 'running' | 'succeeded' | 'failed';
  stage: string;
  result_type: string;
  result_id: string | null;
  error: string | null;
  created_at?: number;
  source_title?: string;
}

export interface SourceSummary {
  id: string;
  kind: string;
  title: string;
  url: string | null;
  url_verified: boolean;
  published_at: string | null;
  received_at: string;
  fetched_at: string | null;
  content_hash: string;
  extraction_status: string;
  completeness: string;
}

export interface Event {
  id: string;
  company_id: string;
  version: number;
  title: string;
  published_at: string | null;
  event_at: string | null;
  topics: string[];
  facts: unknown[];
  topic_method: string;
  extraction_status: string;
  completeness: string;
  source_ids: string[];
  sources: SourceSummary[];
}

export interface SourceDetail extends SourceSummary {
  text: string;
  company_id: string;
  version: number;
  note: string;
}

export interface NewsSearchItem {
  article_id: string;
  source_id: string;
  job_id: string;
  event_id: string | null;
  title: string;
  url: string;
  published_at: string | null;
}

export interface NewsSearchRun {
  id: string;
  company_id: string;
  status: 'queued' | 'collecting' | 'waiting_ingestion' | 'succeeded' | 'partial' | 'failed' | 'cancelled';
  requested_count: number;
  collected_count: number;
  checked_sources: number;
  total_sources: number;
  created_at: number;
  updated_at: number;
  deadline_at: number;
  stop_reason: string | null;
  errors: string[];
  items: NewsSearchItem[];
  newly_downloaded: number;
  reused: number;
}

export interface NewsCollectionStatus {
  enabled: boolean;
  state: 'running' | 'stale' | 'unknown' | 'disabled';
  heartbeat_at: number | null;
  poll_minutes: number | null;
  feeds: Array<{
    id: string;
    url: string;
    poll_seconds: number;
    next_poll_at: number | null;
    last_checked_at: number | null;
    last_success_at: number | null;
    failures: number;
    last_error: string | null;
  }>;
  note: string;
}
