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
